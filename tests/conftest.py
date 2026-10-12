from io import BytesIO
import json

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twobridge.config import Camera, Config, DatasetConfig, TrainConfig


@pytest.fixture
def make_dataset(tmp_path):
    def make(version="v3.0", state_dim=5, action_dim=3, prompt_column=False, video=False):
        root = tmp_path / f"data_{version}_{state_dim}_{action_dim}_{prompt_column}_{video}"
        (root / "meta").mkdir(parents=True)
        features = {
            "sensors.joints": {"dtype": "float32", "shape": [state_dim]},
            "commands": {"dtype": "float32", "shape": [action_dim]},
            "observation.images.front": {"dtype": "video" if video else "image", "shape": [16, 16, 3]},
        }
        info = {
            "codebase_version": version,
            "fps": 10,
            "features": features,
            "chunks_size": 1000,
            "data_path": (
                "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                if version == "v3.0"
                else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
        (root / "meta/info.json").write_text(json.dumps(info))
        (root / "meta/tasks.jsonl").write_text(
            "\n".join(
                json.dumps({"task_index": i, "task": p})
                for i, p in enumerate(["pick object", "place object"])
            )
        )
        rows, episodes = [], []
        for ep in [0, 1]:
            episodes.append(
                {
                    "episode_index": ep,
                    "length": 4,
                    "videos/observation.images.front/chunk_index": 0,
                    "videos/observation.images.front/file_index": 0,
                    "videos/observation.images.front/from_timestamp": ep * 0.4,
                }
            )
            for frame in range(4):
                buffer = BytesIO()
                Image.new("RGB", (16, 16), (ep * 200, frame * 30, 10)).save(buffer, format="PNG")
                row = {
                    "episode_index": ep,
                    "frame_index": frame,
                    "timestamp": frame / 10,
                    "sensors.joints": [ep * 1000 + frame + dim for dim in range(state_dim)],
                    "commands": [ep * 1000 + frame * 2 + dim for dim in range(action_dim)],
                    "task_index": frame % 2,
                }
                if not video:
                    row["observation.images.front"] = {"bytes": buffer.getvalue(), "path": None}
                if prompt_column:
                    row["instruction"] = f"frame {frame}"
                rows.append(row)
        (root / "data/chunk-000").mkdir(parents=True)
        if version == "v3.0":
            # One shard contains two episodes; the second begins in this file.
            pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
            (root / "meta/episodes/chunk-000").mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
        else:
            for ep in [0, 1]:
                pq.write_table(
                    pa.Table.from_pylist(rows[4 * ep : 4 * ep + 4]),
                    root / f"data/chunk-000/episode_{ep:06d}.parquet",
                )
            (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
        if video:
            import av

            path = root / "videos/observation.images.front/chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True)
            with av.open(str(path), "w") as output:
                stream = output.add_stream("libx264", rate=10)
                stream.width = 16
                stream.height = 16
                stream.pix_fmt = "yuv420p"
                for i in range(8):
                    array = np.full((16, 16, 3), 25 if i < 4 else 220, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
        return Config(
            DatasetConfig(
                str(root),
                state_key="sensors.joints",
                action_key="commands",
                prompt_key="instruction" if prompt_column else None,
                cameras=[Camera("observation.images.front")],
                train_episodes=[0],
                val_episodes=[1],
            ),
            TrainConfig(horizon=3, steps=3, warmup_steps=0, batch_size=2, precision="float32"),
        )

    return make


@pytest.fixture
def tiny_backend(monkeypatch):
    from twobridge.backend import prepare_transformers

    prepare_transformers()
    from twobridge._vendor import gemma, gemma_pytorch

    original_pali = gemma_pytorch.PaliGemmaForConditionalGeneration
    original_gemma = gemma_pytorch.GemmaForCausalLM
    monkeypatch.setattr(gemma, "get_config", lambda _: gemma.Config(32, 2, 64, 4, 1, 8))

    def pali(config):
        config.text_config.vocab_size = 512
        config.image_token_index = 511
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_attention_heads = 4
        config.vision_config.num_hidden_layers = 1
        config.vision_config.patch_size = 56
        config.vision_config.projection_dim = 32
        return original_pali(config)

    def expert(config):
        config.vocab_size = 512
        return original_gemma(config)

    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", pali)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", expert)
    import torch

    torch.set_num_threads(2)
