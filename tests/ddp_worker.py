"""Small CPU DDP integration worker (test-only dimensions)."""

import json
import sys
import pytest

from conftest import tiny_backend
from twobridge.config import Config, DatasetConfig, TrainConfig, Camera
from twobridge.training import run

patch = pytest.MonkeyPatch()
tiny_backend.__wrapped__(patch)
raw = json.loads(open(sys.argv[1]).read())
raw["dataset"]["cameras"] = [Camera(**c) for c in raw["dataset"]["cameras"]]
cfg = Config(DatasetConfig(**raw["dataset"]), TrainConfig(**raw["training"]))
run(cfg)
