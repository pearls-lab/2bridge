from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import sentencepiece
import torch

from twobridge import training
from twobridge.backend import build_model
from twobridge.policy import Policy


def test_training_resume_and_portable_inference(make_dataset, tiny_backend, tmp_path, monkeypatch):
    config = make_dataset()
    corpus = tmp_path / "text.txt"
    corpus.write_text(
        "\n".join(
            f"Task: pick object place object, State: {i} {i + 1} {i + 2}; Action: " for i in range(-1, 257)
        )
    )
    prefix = str(tmp_path / "tiny_tokenizer")
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=128,
        model_type="bpe",
        hard_vocab_limit=False,
        minloglevel=2,
    )
    config.training.tokenizer = prefix + ".model"
    config.training.max_token_len = 100
    config.training.save_every = 1
    config.training.eval_every = 1
    config.training.deriv_weight = 0.1
    config.training.deriv_dims = [0, 1]
    weights = tmp_path / "base.safetensors"
    torch.manual_seed(4)
    safetensors.torch.save_model(build_model(config.training, 3), str(weights))
    config.training.weights = str(weights)
    config.training.output = str(tmp_path / "uninterrupted")
    training.run(config)
    expected = safetensors.torch.load_file(
        str(Path(config.training.output) / "checkpoints/step_00000003/model.safetensors")
    )
    config.training.output = str(tmp_path / "resumed")
    original_save = training.save_checkpoint

    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        raise InterruptedError("simulated interruption after durable checkpoint")

    monkeypatch.setattr(training, "save_checkpoint", interrupt)
    with pytest.raises(InterruptedError):
        training.run(config)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    resume = Path(config.training.output) / "checkpoints/step_00000001"
    training.run(config, resume)
    final = Path(config.training.output) / "checkpoints/step_00000003"
    actual = safetensors.torch.load_file(str(final / "model.safetensors"))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    # Inference must not need the training data, original tokenizer, or initial weights.
    import shutil

    portable = tmp_path / "portable"
    shutil.copytree(final, portable)
    Path(config.training.tokenizer).unlink()
    weights.unlink()
    policy = Policy(portable, device="cpu")
    action = policy.predict(
        {
            "sensors.joints": np.zeros(5, dtype=np.float32),
            "observation.images.front": np.zeros((16, 16, 3), dtype=np.uint8),
        },
        "pick object",
        num_steps=2,
    )
    assert action.shape == (3, 3)
    assert np.isfinite(action).all()


def test_config_rejects_unknown_fields_and_resolves_paths(tmp_path):
    from twobridge.config import load_config

    p = tmp_path / "config.toml"
    p.write_text("""[dataset]
root="data"
[[dataset.cameras]]
key="observation.images.front"
[training]
weights="base/model.safetensors"
""")
    c = load_config(p)
    assert c.dataset.root == str(tmp_path / "data")
    p.write_text(p.read_text() + "lr_typo=1.0\n")
    with pytest.raises(TypeError):
        load_config(p)


def test_two_process_cpu_ddp(make_dataset, tiny_backend, tmp_path):
    import json
    import os
    import subprocess
    import sys

    config = make_dataset()
    corpus = tmp_path / "ddp_text.txt"
    corpus.write_text("\n".join(f"Task: pick object place object, State: {i}; Action: " for i in range(257)))
    prefix = str(tmp_path / "ddp_tokenizer")
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=prefix, vocab_size=128, model_type="bpe", minloglevel=2
    )
    config.training.tokenizer = prefix + ".model"
    config.training.output = str(tmp_path / "ddp_run")
    config.training.weights = str(tmp_path / "ddp_base.safetensors")
    config.training.batch_size = 1
    config.training.accumulation_steps = 2
    config.training.steps = 2
    config.training.save_every = 2
    config.training.eval_every = 1
    safetensors.torch.save_model(build_model(config.training, 3), config.training.weights)
    path = tmp_path / "ddp_config.json"
    path.write_text(json.dumps(config.to_dict()))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("ddp_worker.py")),
            str(path),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=False,  # Include both output streams in the assertion below on failure.
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = Path(config.training.output) / "checkpoints/step_00000002"
    saved = torch.load(checkpoint / "training_state.pt", weights_only=True)
    assert saved["world_size"] == 2 and len(saved["rng"]) == 2
