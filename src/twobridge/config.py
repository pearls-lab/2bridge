"""Strict, portable configuration. Paths are relative to the TOML file."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import tomllib


@dataclass
class Camera:
    key: str
    crop: list[int] | None = None  # x0, y0, x1, y1 in source pixels
    resize: str = "stretch"  # or letterbox
    jpeg_quality: int | None = None
    image_path: str | None = None  # optional external-image path template


@dataclass
class DatasetConfig:
    root: str
    state_key: str = "observation.state"
    action_key: str = "action"
    prompt_key: str | None = None
    default_prompt: str | None = None
    cameras: list[Camera] = field(default_factory=list)
    train_episodes: list[int] | None = None
    val_episodes: list[int] = field(default_factory=list)
    # [action dimension, observation-state dimension], explicitly opt-in.
    relative_pairs: list[list[int]] = field(default_factory=list)
    feature_overrides: dict = field(default_factory=dict)  # explicit schema for legacy exports


@dataclass
class TrainConfig:
    output: str = "runs/policy"
    weights: str = ""
    tokenizer: str = ""
    steps: int = 8000
    batch_size: int = 32  # per process
    accumulation_steps: int = 1
    horizon: int = 10
    workers: int = 0
    seed: int = 42
    lr: float = 2.5e-5
    end_lr: float = 2.5e-6
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    save_every: int = 2000
    eval_every: int = 500
    eval_batches: int = 10
    log_every: int = 10
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True
    max_token_len: int = 200
    deriv_weight: float = 0.0
    deriv_dims: list[int] = field(default_factory=list)


@dataclass
class Config:
    dataset: DatasetConfig
    training: TrainConfig

    def validate(self):
        t, d = self.training, self.dataset
        if not d.cameras or len(d.cameras) > 3:
            raise ValueError("Map 1–3 RGB cameras explicitly (select a subset if the dataset has more).")
        for cam in d.cameras:
            if cam.resize not in {"stretch", "letterbox"}:
                raise ValueError("camera.resize must be stretch or letterbox")
            if cam.crop is not None and (
                len(cam.crop) != 4
                or not all(type(x) is int for x in cam.crop)
                or not (0 <= cam.crop[0] < cam.crop[2] and 0 <= cam.crop[1] < cam.crop[3])
            ):
                raise ValueError("crop must be integer [x0,y0,x1,y1] with positive extent")
            if cam.jpeg_quality is not None and not 1 <= cam.jpeg_quality <= 100:
                raise ValueError("jpeg_quality must be in [1,100]")
        for name in (
            "steps",
            "batch_size",
            "accumulation_steps",
            "horizon",
            "save_every",
            "eval_every",
            "eval_batches",
            "log_every",
            "max_token_len",
        ):
            if type(getattr(t, name)) is not int or getattr(t, name) < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        if t.workers < 0 or not 0 <= t.warmup_steps < t.steps:
            raise ValueError("workers must be nonnegative and warmup_steps must be in [0,steps)")
        for name in ("lr", "end_lr", "weight_decay", "grad_clip", "deriv_weight"):
            if not math.isfinite(getattr(t, name)) or getattr(t, name) < 0:
                raise ValueError(f"training.{name} must be finite and nonnegative")
        if t.lr <= 0 or t.grad_clip <= 0 or t.end_lr > t.lr:
            raise ValueError("lr and grad_clip must be positive; end_lr must not exceed lr")
        if t.precision not in {"bfloat16", "float32"}:
            raise ValueError("precision must be bfloat16 or float32")
        if t.deriv_weight > 0 and (not t.deriv_dims or t.horizon < 3):
            raise ValueError("Derivative loss needs explicit deriv_dims and horizon >= 3")
        if len(t.deriv_dims) != len(set(t.deriv_dims)):
            raise ValueError("deriv_dims must be unique")
        for episodes in (d.train_episodes, d.val_episodes):
            if episodes is not None and (
                len(set(episodes)) != len(episodes) or any(type(ep) is not int or ep < 0 for ep in episodes)
            ):
                raise ValueError("Episode IDs must be unique nonnegative integers")
        if d.train_episodes is not None and set(d.train_episodes) & set(d.val_episodes):
            raise ValueError("Training and validation episodes must be disjoint")
        if any(len(pair) != 2 or any(type(x) is not int or x < 0 for x in pair) for pair in d.relative_pairs):
            raise ValueError("relative_pairs must contain [action_dim,state_dim] nonnegative integers")
        if len({p[0] for p in d.relative_pairs}) != len(d.relative_pairs):
            raise ValueError("An action dimension may be relativized only once")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    extra = set(raw) - {"dataset", "training"}
    if extra:
        raise ValueError(f"Unknown config sections: {sorted(extra)}")
    d = dict(raw["dataset"])
    d["cameras"] = [Camera(**c) for c in d.get("cameras", [])]
    config = Config(DatasetConfig(**d), TrainConfig(**raw.get("training", {})))
    for obj, names in [(config.dataset, ["root"]), (config.training, ["output", "weights", "tokenizer"])]:
        for name in names:
            value = getattr(obj, name)
            if value:
                p = Path(value).expanduser()
                setattr(obj, name, str((path.parent / p).resolve() if not p.is_absolute() else p.resolve()))
    config.validate()
    return config
