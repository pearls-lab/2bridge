"""Explicit simulator/checkpoint contracts; no object poses or success predicates in S2."""

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def flatten(observation, prefix=""):
    result = {}
    for key, value in observation.items():
        key = f"{prefix}::{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, key))
        else:
            result[key] = value
    return result


class RoboCasaAdapter:
    """RoboCasa Gym PandaOmron: state16, arm action7, stationary base.

    Images from the Gym wrapper are already upright. No extra vertical flip.
    Set feature_keys to match the release checkpoint's LeRobot feature names.
    """

    action_dim = 7
    cameras = ("video.robot0_agentview_left", "video.robot0_eye_in_hand", "video.robot0_agentview_right")
    state_keys = ("state.base_position", "state.base_rotation", "state.end_effector_position_relative",
                  "state.end_effector_rotation_relative", "state.gripper_qpos")

    def __init__(self, feature_keys=None, state_key="observation.state"):
        self.feature_keys = feature_keys or ["observation.images.left", "observation.images.wrist",
                                             "observation.images.right"]
        if len(self.feature_keys) != 3:
            raise ValueError("RoboCasa requires three camera feature keys")
        self.state_key = state_key

    def images(self, obs):
        return [array(obs[key]) for key in self.cameras]

    def observation(self, obs):
        state = np.concatenate([array(obs[key]).reshape(-1) for key in self.state_keys])
        if state.shape != (16,):
            raise ValueError("Expected RoboCasa Gym's 16-dimensional state")
        return {**dict(zip(self.feature_keys, self.images(obs), strict=True)), self.state_key: state}

    def action(self, action):
        return {"action.base_motion": np.zeros(4, dtype=np.float32),
                "action.control_mode": np.zeros(1, dtype=np.float32),
                "action.end_effector_position": action[:3].astype(np.float32),
                "action.end_effector_rotation": action[3:6].astype(np.float32),
                "action.gripper_close": action[6:7].astype(np.float32)}


class BehaviorAdapter:
    """BEHAVIOR R1Pro raw OmniGibson or flattened evaluator observations.

    Default state25: base velocity3, torso4, left arm7, left fingers2,
    right arm7, right fingers2. Action23 follows the native R1Pro controller.
    These must match training preprocessing; no automatic gripper/delta conversion.
    """

    def __init__(self, feature_keys=None, state_key="observation.state", camera_keys=None,
                 proprio_key="robot_r1::proprio", state_indices=None, action_dim=23):
        self.feature_keys = feature_keys or ["observation.rgb.zed_link_camera_0",
                                             "observation.rgb.left_realsense_link_camera_0",
                                             "observation.rgb.right_realsense_link_camera_0"]
        self.camera_keys = camera_keys or [
            "robot_r1::robot_r1:zed_link:Camera:0::rgb",
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
        ]
        self.state_indices = (list(range(3)) + list(range(53, 57)) + list(range(3, 10)) + [24, 25]
                              + list(range(28, 35)) + [49, 50]) if state_indices is None else state_indices
        if len(self.camera_keys) != len(self.feature_keys) or not self.camera_keys:
            raise ValueError("Camera source and checkpoint keys must match")
        if not self.state_indices or any(type(i) is not int or i < 0 for i in self.state_indices):
            raise ValueError("state_indices must be nonnegative integers")
        self.state_key, self.proprio_key, self.action_dim = state_key, proprio_key, action_dim

    def images(self, obs):
        obs = flatten(obs)
        return [array(obs[key])[..., :3] for key in self.camera_keys]

    def observation(self, obs):
        obs = flatten(obs)
        state = array(obs[self.proprio_key])
        if state.ndim != 1:
            raise ValueError("Use one Harness per episode, without a batch dimension")
        state = state[self.state_indices].astype(np.float32)
        return {**dict(zip(self.feature_keys, self.images(obs), strict=True)), self.state_key: state}

    def action(self, action):
        # OmniGibson's native Environment.step accepts torch tensors.
        import torch

        return torch.as_tensor(action.copy(), dtype=torch.float32)
