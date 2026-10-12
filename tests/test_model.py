import numpy as np
import torch
from twobridge.config import TrainConfig
from twobridge.losses import flow_loss
from twobridge.preprocessing import Statistics, prepare_batch
from twobridge.training import learning_rate


class FakeTokenizer:
    def encode(self, prompt, state):
        return np.array([1, 2, 3, 4]), np.array([True] * 4)


def test_masked_loss_and_derivative():
    target = torch.zeros(2, 4, 32)
    pred = torch.zeros_like(target, requires_grad=True)
    with torch.no_grad():
        pred[:, :, 7:] = 1000
        pred[0, 2:, :] = 1000
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    loss = flow_loss(target, pred, target, torch.tensor([0.5, 0.5]), target, mask, 7, [0, 1, 2], 0.1)
    assert loss["loss"].item() == 0
    loss["loss"].backward()
    assert pred.grad[:, :, 7:].abs().sum() == 0
    assert pred.grad[0, 2:].abs().sum() == 0


def test_model_forward_backward_inference_and_strict_reload(tiny_backend, tmp_path):
    from twobridge.backend import build_model
    import safetensors.torch

    t = TrainConfig(
        horizon=3, precision="float32", gradient_checkpointing=True, deriv_dims=[0, 1], deriv_weight=0.1
    )
    stats = Statistics(np.zeros(5), np.ones(5), np.zeros(3), np.ones(3))
    b = {
        "state": np.zeros((1, 5), dtype=np.float32),
        "images": np.zeros((1, 1, 224, 224, 3), dtype=np.float32),
        "actions": np.ones((1, 3, 3), dtype=np.float32),
        "action_mask": np.array([[1, 1, 0]], dtype=bool),
        "prompt": ["test"],
    }
    obs, actions = prepare_batch(b, stats, FakeTokenizer(), torch.device("cpu"))
    assert not obs.image_masks["left_wrist_0_rgb"].any()
    model = build_model(t, 3)
    model.train()
    loss = model(obs, actions)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    path = str(tmp_path / "model.safetensors")
    safetensors.torch.save_model(model, path)
    restored = build_model(t, 3)
    safetensors.torch.load_model(restored, path, strict=True)
    restored.eval()
    output = restored.sample_actions(torch.device("cpu"), obs, num_steps=2)
    assert output.shape == (1, 3, 32)
    assert torch.isfinite(output).all()


def test_schedule():
    c = TrainConfig(steps=10, warmup_steps=2)
    assert learning_rate(0, c) == c.lr / 2
    assert learning_rate(1, c) == c.lr
    assert learning_rate(9, c) == c.end_lr
