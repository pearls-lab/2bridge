"""Read local LeRobot v2.0/v2.1/v3.0 without a task-specific converter.

Only numeric state/action/index columns are resident in memory. Image Parquet
shards use a small worker-local cache. Videos are decoded by timestamp with PyAV.
"""

from bisect import bisect_right
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import re

import av
import numpy as np
from PIL import Image, ImageOps
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .config import Config


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pq.read_table(path).to_pylist()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def image_transform(array, camera):
    image = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
    if camera.crop:
        x0, y0, x1, y1 = camera.crop
        if x1 > image.width or y1 > image.height:
            raise ValueError(f"Crop {camera.crop} exceeds {camera.key} dimensions {image.size}")
        image = image.crop((x0, y0, x1, y1))
    if camera.resize == "letterbox":
        image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BOX, color=(0, 0, 0))
    else:
        image = image.resize((224, 224), Image.Resampling.BOX)
    if camera.jpeg_quality:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=camera.jpeg_quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 127.5 - 1


def decode_video(path, timestamp, tolerance):
    """Nearest frame, never silently substitute a black image or wrong episode."""
    best, error = None, float("inf")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((timestamp - 1) / stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            distance = abs(float(frame.time) - timestamp)
            if distance < error:
                best, error = frame, distance
            if frame.time > timestamp + tolerance:
                break
        if best is None or error > tolerance:
            raise ValueError(f"No video frame within {tolerance:.4f}s of {timestamp:.4f}s: {path}")
        return best.to_ndarray(format="rgb24")


class LeRobotData(Dataset):
    def __init__(self, config: Config, split="train"):
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.config = config
        self.root = Path(config.dataset.root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.version = self.info.get("codebase_version")
        if self.version not in {"v2.0", "v2.1", "v3.0"}:
            raise ValueError(f"Unsupported LeRobot version {self.version!r}; expected v2.0, v2.1 or v3.0")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset fps must be finite and positive")
        d = config.dataset
        self.features = {**self.info.get("features", {}), **d.feature_overrides}
        if not self.features:
            raise ValueError(
                "Missing features in meta/info.json; provide explicit dataset.feature_overrides for a legacy export"
            )
        self.state_dim = self._dimension(d.state_key)
        self.action_dim = self._dimension(d.action_key)
        if any(x < 0 or x >= self.action_dim for x in config.training.deriv_dims):
            raise ValueError("deriv_dims must index real action dimensions")
        for a, s in d.relative_pairs:
            if a >= self.action_dim or s >= self.state_dim:
                raise ValueError("relative_pairs contains an out-of-range dimension")
        for camera in d.cameras:
            feature = self.features.get(camera.key, {})
            if feature.get("dtype") not in {"video", "image"}:
                raise ValueError(f"Camera {camera.key!r} must be a LeRobot image or video feature")
            shape = feature.get("shape", [])
            if len(shape) != 3 or (shape[0] != 3 and shape[-1] != 3):
                raise ValueError(f"Camera {camera.key!r} must contain RGB frames; shape={shape}")
            if feature.get("info", {}).get("is_depth_map", False):
                raise ValueError("Depth cameras require an explicit RGB conversion before training")
        self.tasks = {}
        for path in (self.root / "meta/tasks.jsonl", self.root / "meta/tasks.parquet"):
            if path.exists():
                for row in read_records(path):
                    task = row.get("task", row.get("__index_level_0__"))
                    if task is not None:
                        self.tasks[int(row["task_index"])] = task
        self.episode_metadata = {}
        metadata_paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
        if (self.root / "meta/episodes.jsonl").exists():
            metadata_paths += [self.root / "meta/episodes.jsonl"]
        for path in metadata_paths:
            for row in read_records(path):
                ep = int(row["episode_index"])
                if ep in self.episode_metadata:
                    raise ValueError(f"Duplicate episode metadata: {ep}")
                self.episode_metadata[ep] = row
        # Respect info.json's template, not an assumed number of files per episode.
        template = self.info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        glob = re.sub(r"\{[^}]+\}", "*", template)
        self.paths = sorted(self.root.glob(glob))
        if not self.paths:
            raise ValueError(f"No Parquet files matched data_path: {template}")
        self.file_offsets = [0]
        batches = []
        for path in self.paths:
            schema = pq.read_schema(path)
            columns = list(
                dict.fromkeys(
                    [d.state_key, d.action_key, "episode_index", "frame_index", "timestamp"]
                    + ([d.prompt_key] if d.prompt_key else ["task_index"])
                )
            )
            required = set(columns) - {"task_index"}
            if required - set(schema.names):
                raise ValueError(f"Missing columns in {path}: {sorted(required - set(schema.names))}")
            table = pq.read_table(path, columns=[c for c in columns if c in schema.names])
            batches.extend(table.to_pylist())
            self.file_offsets.append(len(batches))
        self.rows = batches
        self.episodes = {}
        for index, row in enumerate(self.rows):
            self.episodes.setdefault(int(row["episode_index"]), []).append(index)
        for ep, indices in self.episodes.items():
            indices.sort(key=lambda idx: int(self.rows[idx]["frame_index"]))
            frames = [int(self.rows[idx]["frame_index"]) for idx in indices]
            if frames != list(range(len(indices))):
                raise ValueError(
                    f"Episode {ep} has missing/duplicate frame indices or does not begin at zero"
                )
            times = np.array([self.rows[idx]["timestamp"] for idx in indices], dtype=float)
            if not np.all(np.isfinite(times)) or not np.allclose(
                times, np.arange(len(times)) / self.fps, atol=0.5 / self.fps, rtol=0
            ):
                raise ValueError(f"Episode {ep} timestamps must start at zero and follow dataset fps")
            meta = self.episode_metadata.get(ep)
            if meta is not None and int(meta["length"]) != len(indices):
                raise ValueError(f"Episode {ep} length disagrees with metadata")
        known = set(self.episodes)
        for chosen in (d.train_episodes, d.val_episodes):
            if chosen is not None and set(chosen) - known:
                raise ValueError(f"Unknown episode IDs: {sorted(set(chosen) - known)}")
        selected = (
            (sorted(known - set(d.val_episodes)) if d.train_episodes is None else d.train_episodes)
            if split == "train"
            else d.val_episodes
        )
        if not selected:
            raise ValueError(f"No episodes selected for {split}")
        self.selected_episodes = sorted(selected)
        self.samples = [
            (ep, frame) for ep in self.selected_episodes for frame in range(len(self.episodes[ep]))
        ]
        self.states = np.asarray([r[d.state_key] for r in self.rows], dtype=np.float32)
        self.actions = np.asarray([r[d.action_key] for r in self.rows], dtype=np.float32)
        if self.states.shape != (len(self.rows), self.state_dim) or self.actions.shape != (
            len(self.rows),
            self.action_dim,
        ):
            raise ValueError("State/action rows do not match feature shapes")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("Non-finite states or actions in dataset")
        # Validate every selected prompt before allocating the model.
        for ep, frame in self.samples:
            self.prompt(self.episodes[ep][frame])

    def _dimension(self, key):
        feature = self.features.get(key)
        if not feature or len(feature["shape"]) != 1 or not 1 <= feature["shape"][0] <= 32:
            raise ValueError(f"{key!r} must be a numeric vector of 1–32 dimensions")
        if feature.get("dtype") not in {"float32", "float64", "int32", "int64", "float16"}:
            raise ValueError(f"{key!r} must have a numeric dtype")
        return int(feature["shape"][0])

    def prompt(self, index):
        d = self.config.dataset
        row = self.rows[index]
        text = row.get(d.prompt_key) if d.prompt_key else self.tasks.get(int(row.get("task_index", -1)))
        if text is None:
            text = d.default_prompt
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Missing/empty instruction at episode {row['episode_index']}, frame {row['frame_index']}; "
                "supply valid per-frame task_index, prompt_key, or an explicit default_prompt"
            )
        return text

    @lru_cache(maxsize=2)
    def _image_table(self, file_index):
        keys = list(
            dict.fromkeys(
                c.key
                for c in self.config.dataset.cameras
                if self.features[c.key]["dtype"] == "image" and not c.image_path
            )
        )
        return pq.read_table(self.paths[file_index], columns=keys)

    def _image(self, index, camera):
        key = camera.key
        row = self.rows[index]
        if camera.image_path:
            values = {
                "camera_key": key,
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "episode_chunk": int(row["episode_index"]) // self.info.get("chunks_size", 1000),
            }
            with Image.open(safe_path(self.root, camera.image_path.format(**values))) as im:
                return np.asarray(im.convert("RGB"))
        ep = int(row["episode_index"])
        if self.features[key]["dtype"] == "image":
            f = bisect_right(self.file_offsets, index) - 1
            value = self._image_table(f)[key][index - self.file_offsets[f]].as_py()
            if isinstance(value, dict):
                if value.get("bytes") is not None:
                    with Image.open(BytesIO(value["bytes"])) as im:
                        return np.asarray(im.convert("RGB"))
                value = value.get("path")
            if isinstance(value, str):
                with Image.open(safe_path(self.root, value)) as im:
                    return np.asarray(im.convert("RGB"))
            raise ValueError(f"Unsupported image storage for {key}: expected bytes/path")
        template = self.info["video_path"]
        timestamp = float(row["timestamp"])
        values = {
            "video_key": key,
            "episode_index": ep,
            "episode_chunk": ep // self.info.get("chunks_size", 1000),
        }
        if self.version == "v3.0":
            meta = self.episode_metadata[ep]
            values.update(
                chunk_index=int(meta[f"videos/{key}/chunk_index"]),
                file_index=int(meta[f"videos/{key}/file_index"]),
            )
            timestamp += float(meta[f"videos/{key}/from_timestamp"])
        path = safe_path(self.root, template.format(**values))
        return decode_video(path, timestamp, tolerance=0.51 / self.fps)

    def numeric(self, sample_index):
        ep, frame = self.samples[sample_index]
        indices = self.episodes[ep]
        index = indices[frame]
        positions = np.arange(frame, frame + self.config.training.horizon)
        valid = positions < len(indices)
        selected = np.asarray(indices)[np.minimum(positions, len(indices) - 1)]
        state, action = self.states[index].copy(), self.actions[selected].copy()
        for a, s in self.config.dataset.relative_pairs:
            action[:, a] -= state[s]  # observation-time state for every action in the chunk
        return state, action, valid

    def __getitem__(self, sample_index):
        ep, frame = self.samples[sample_index]
        index = self.episodes[ep][frame]
        state, action, valid = self.numeric(sample_index)
        images = [image_transform(self._image(index, c), c) for c in self.config.dataset.cameras]
        return {
            "state": state,
            "actions": action,
            "action_mask": valid,
            "images": np.stack(images),
            "prompt": self.prompt(index),
        }

    def __len__(self):
        return len(self.samples)

    def describe(self):
        return {
            "version": self.version,
            "fps": self.fps,
            "episodes": self.selected_episodes,
            "frames": len(self),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cameras": [c.key for c in self.config.dataset.cameras],
            "parquet_files": len(self.paths),
        }
