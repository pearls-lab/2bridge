"""Simulator rollout CLI. Run in the corresponding simulator's Python environment."""

import argparse
import json
import tomllib
from pathlib import Path

from .adapters import BehaviorAdapter, RoboCasaAdapter
from .clients import FrontierPlanner, VisualMonitor
from .core import Harness, Settings


class RemotePolicy:
    """OpenPI websocket policy endpoint; imported only in simulator processes."""

    def __init__(self, host, port):
        from websockets.sync.client import connect

        from .wire import unpack

        self.socket = connect(f"ws://{host}:{port}", compression=None, max_size=32 * 1024 * 1024,
                              open_timeout=15, ping_interval=None)
        self.metadata = unpack(self.socket.recv(timeout=120))

    def predict(self, observation, prompt):
        from .wire import pack, unpack

        self.socket.send(pack({**observation, "prompt": prompt}))
        response = self.socket.recv(timeout=120)
        if isinstance(response, str):
            # The wire protocol uses text frames to report remote inference errors.
            raise RuntimeError(response)  # noqa: TRY004 -- server failure, not an invalid argument type
        return unpack(response)["actions"]

    def reset(self):
        pass  # Stateless chunk inference; Harness owns and clears all action queues.

    def close(self):
        self.socket.close()


def rollout(env, harness, task, episode, plan_path, max_steps, seed=None):
    result = env.reset(seed=seed) if seed is not None else env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    harness.reset(task, obs, episode=episode, plan_path=plan_path)
    reason = "step_limit"
    # Environment termination is recorded independently of monitor plan exhaustion.
    steps = 0
    reward = None
    environment_success = None
    for _ in range(max_steps):
        try:
            action = harness.act(obs)
        except StopIteration:
            reason = "plan_exhausted"
            break
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False
        if isinstance(info.get("success"), (bool,)):
            environment_success = info["success"]
        steps += 1
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            break
    return {"episode": episode, "steps": steps, "stop_reason": reason,
            "plan_exhausted": harness.exhausted, "environment_success": environment_success, "last_reward": float(reward) if reward is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    root = args.config.resolve().parent
    run = config["run"]
    benchmark = run["benchmark"]
    if benchmark not in {"robocasa", "behavior"}:
        raise ValueError("benchmark must be robocasa or behavior")
    adapter = (RoboCasaAdapter if benchmark == "robocasa" else BehaviorAdapter)(**config.get("adapter", {}))
    policy_config = config["policy"].copy()
    kind = policy_config.pop("kind", "checkpoint")
    if kind == "websocket":
        policy = RemotePolicy(**policy_config)
    elif kind == "checkpoint":
        from twobridge.policy import Policy

        policy_config["checkpoint"] = str(root / policy_config["checkpoint"])
        policy = Policy(**policy_config)
    else:
        raise ValueError("Unknown policy kind")
    output = root / run["output"]
    output.mkdir(parents=True, exist_ok=True)
    # Each output directory holds exactly one episode. Prevent mixing old and new traces.
    with (output / "config.json").open("x") as stream:
        json.dump(config, stream, indent=2)
    harness = Harness(policy, FrontierPlanner(**config["planner"]), VisualMonitor(**config["monitor"]),
                      adapter, Settings(**config.get("harness", {})), output / "trace.jsonl")
    if benchmark == "robocasa":
        import gymnasium as gym
        import robocasa  # noqa: F401 -- registers Gym environments

        env = gym.make("robocasa/" + run["env_task"], split=run.get("split", "target"),
                       camera_heights=512, camera_widths=512)
    else:
        import omnigibson as og

        og.gm.HEADLESS = True
        # A concrete task/scene/robot YAML from the installed BEHAVIOR version is required.
        # Official benchmark instance scheduling stays in its evaluator, see docs.
        env = og.Environment(configs=str(root / run["env_config"]))
    try:
        result = rollout(env, harness, run["task"], run["episode"], root / run["plan_cache"],
                         run.get("max_steps", 3000), run.get("seed") if benchmark == "robocasa" else None)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        env.close()
        if hasattr(policy, "close"):
            policy.close()


if __name__ == "__main__":
    main()
