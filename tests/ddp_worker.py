"""Small CPU DDP integration worker (test-only dimensions)."""

import json
import sys
from pathlib import Path

import pytest
from conftest import tiny_backend

from twobridge.config import Camera, Config, DatasetConfig, TrainConfig
from twobridge.training import run

patch = pytest.MonkeyPatch()
tiny_backend.__wrapped__(patch)
raw = json.loads(Path(sys.argv[1]).read_text())
raw["dataset"]["cameras"] = [Camera(**c) for c in raw["dataset"]["cameras"]]
cfg = Config(DatasetConfig(**raw["dataset"]), TrainConfig(**raw["training"]))
run(cfg)
