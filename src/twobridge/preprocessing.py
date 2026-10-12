"""Shared training/inference transforms; state remains part of the Pi0.5 prompt."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece
import torch

SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass
class Statistics:
    state_low: np.ndarray
    state_high: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    def normalize(self, x, kind):
        low, high = getattr(self, kind + "_low"), getattr(self, kind + "_high")
        return (2 * (x - low) / np.maximum(high - low, 1e-6) - 1).astype(np.float32)

    def unnormalize_actions(self, x):
        return (
            (x + 1) * 0.5 * np.maximum(self.action_high - self.action_low, 1e-6) + self.action_low
        ).astype(np.float32)

    def save(self, path):
        Path(path).write_text(json.dumps({k: v.tolist() for k, v in vars(self).items()}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        result = cls(
            **{k: np.asarray(v, dtype=np.float32) for k, v in json.loads(Path(path).read_text()).items()}
        )
        for kind in ("state", "action"):
            lo, hi = getattr(result, kind + "_low"), getattr(result, kind + "_high")
            if lo.ndim != 1 or hi.shape != lo.shape or not np.isfinite([lo, hi]).all() or np.any(hi < lo):
                raise ValueError(f"Invalid {kind} normalization statistics")
        return result


def compute_statistics(dataset, max_samples=8192):
    """Deterministic training-only quantiles, including valid relative-action horizons."""
    indices = np.linspace(0, len(dataset) - 1, min(max_samples, len(dataset)), dtype=int)
    states, actions = [], []
    for idx in indices:
        state, action, mask = dataset.numeric(int(idx))
        states.append(state)
        actions.append(action[mask])
    slo, shi = np.quantile(np.stack(states), [0.01, 0.99], axis=0)
    alo, ahi = np.quantile(np.concatenate(actions), [0.01, 0.99], axis=0)
    return Statistics(*(v.astype(np.float32) for v in (slo, shi, alo, ahi)))


class PromptTokenizer:
    def __init__(self, path, max_length):
        self.model = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.max_length = max_length

    def encode(self, prompt, state):
        prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        discrete = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
        text = f"Task: {prompt}, State: {' '.join(map(str, discrete))};\nAction: "
        tokens = self.model.encode(text, add_bos=True)
        if len(tokens) > self.max_length:
            raise ValueError(
                f"Prompt plus state needs {len(tokens)} tokens; increase max_token_len "
                f"(currently {self.max_length}). State tokens are never silently truncated."
            )
        n = len(tokens)
        return (
            np.asarray(tokens + [0] * (self.max_length - n), dtype=np.int64),
            np.arange(self.max_length) < n,
        )


def prepare_batch(batch, stats, tokenizer, device):
    states = np.asarray(batch["state"], dtype=np.float32)
    states = stats.normalize(states, "state")
    token_pairs = [
        tokenizer.encode(prompt, state) for prompt, state in zip(batch["prompt"], states, strict=True)
    ]
    tokens, token_masks = map(np.stack, zip(*token_pairs, strict=True))
    state = torch.zeros((len(states), 32), dtype=torch.float32, device=device)
    state[:, : states.shape[-1]] = torch.as_tensor(states, device=device)
    raw_images = torch.as_tensor(batch["images"], device=device, dtype=torch.float32).permute(0, 1, 4, 2, 3)
    images, image_masks = {}, {}
    for i, slot in enumerate(SLOTS):
        present = i < raw_images.shape[1]
        images[slot] = raw_images[:, i] if present else torch.zeros_like(raw_images[:, 0])
        image_masks[slot] = torch.full((len(states),), present, dtype=torch.bool, device=device)
    obs = SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=torch.as_tensor(tokens, device=device),
        tokenized_prompt_mask=torch.as_tensor(token_masks, device=device),
        action_mask=None,
    )
    actions = None
    if "actions" in batch:
        normalized = stats.normalize(np.asarray(batch["actions"], dtype=np.float32), "action")
        actions = torch.zeros((*normalized.shape[:-1], 32), dtype=torch.float32, device=device)
        actions[..., : normalized.shape[-1]] = torch.as_tensor(normalized, device=device)
        obs.action_mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool, device=device)
    return obs, actions


def preprocess_observation_pytorch(observation, train=False):
    # Inputs already have identical resize/crop/normalization in train and inference.
    if not train:
        return observation
    images = {}
    for key, image in observation.images.items():
        x = (image + 1) / 2
        x = x * (0.7 + torch.rand((), device=x.device) * 0.6)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * (0.6 + torch.rand((), device=x.device) * 0.8) + mean
        gray = x.mean(dim=1, keepdim=True)
        x = gray + (x - gray) * (0.5 + torch.rand((), device=x.device))
        images[key] = x.clamp(0, 1) * 2 - 1
    return SimpleNamespace(**{**vars(observation), "images": images})


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
