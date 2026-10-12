"""Load a release checkpoint and predict actions in the dataset's original units."""

import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from .backend import build_model
from .config import Camera, TrainConfig
from .dataset import image_transform
from .preprocessing import Statistics, PromptTokenizer, prepare_batch


class Policy:
    def __init__(self, checkpoint, device="cuda"):
        root = Path(checkpoint)
        self.device = torch.device(device)
        config = json.loads((root / "config.json").read_text())
        self.dataset_config = config["dataset"]
        training = TrainConfig(**config["training"])
        training.gradient_checkpointing = False
        if self.device.type == "cpu":
            training.precision = "float32"
        self.stats = Statistics.load(root / "stats.json")
        self.action_dim = len(self.stats.action_low)
        self.tokenizer = PromptTokenizer(root / "tokenizer.model", training.max_token_len)
        self.model = build_model(training, self.action_dim)
        safetensors.torch.load_model(self.model, str(root / "model.safetensors"), strict=True)
        self.model.to(self.device).eval()
        self.cameras = [Camera(**c) for c in self.dataset_config["cameras"]]

    @torch.no_grad()
    def predict(self, observation, prompt, *, num_steps=10):
        """observation uses dataset feature names; images are HWC RGB uint8.

        Returns [horizon, action_dim]. No robot commands are sent.
        Relative dimensions are restored against this observation-time state.
        """
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        state = np.asarray(observation[self.dataset_config["state_key"]], dtype=np.float32)
        if state.shape != self.stats.state_low.shape or not np.isfinite(state).all():
            raise ValueError("State shape/values disagree with the checkpoint")
        images = []
        for camera in self.cameras:
            image = np.asarray(observation[camera.key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{camera.key} must be HWC RGB uint8")
            images.append(image_transform(image, camera))
        batch = {"state": state[None], "images": np.stack(images)[None], "prompt": [prompt]}
        obs, _ = prepare_batch(batch, self.stats, self.tokenizer, self.device)
        actions = self.model.sample_actions(self.device, obs, num_steps=num_steps)[0, :, : self.action_dim]
        actions = self.stats.unnormalize_actions(actions.float().cpu().numpy())
        for a, s in self.dataset_config["relative_pairs"]:
            actions[:, a] += state[s]
        return actions
