"""One controller per episode; simulator success is deliberately outside this module."""

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .clients import encode_image, parse_plan


@dataclass(frozen=True)
class Settings:
    poll_interval: int = 40
    confirmations: int = 2
    image_size: int = 512
    action_horizon: int = 8
    prompt_template: str = "{task} Now: {subgoal}"

    def __post_init__(self):
        for key in ("poll_interval", "confirmations", "image_size", "action_horizon"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.prompt_template.format(task="task", subgoal="subgoal")


class Harness:
    """Call reset at episode start; act exactly once per executed simulator step.

    The adapter explicitly separates policy features from RGB-only S2 inputs.
    StopIteration means the monitor exhausted the plan, NOT benchmark success.
    Transport/model failures propagate, aborting the rollout rather than advancing.
    """

    def __init__(self, policy, planner, monitor, adapter, settings=None, trace=None):
        self.policy, self.planner, self.monitor, self.adapter = policy, planner, monitor, adapter
        self.settings = settings or Settings()
        self.trace = Path(trace) if trace else None
        self.ready = False

    def _images(self, observation):
        return tuple(encode_image(x, self.settings.image_size) for x in self.adapter.images(observation))

    def _log(self, event, **data):
        if self.trace:
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with self.trace.open("a") as stream:
                stream.write(json.dumps({"event": event, "episode": self.episode, "step": self.step,
                                         "subgoal_index": self.index, **data}) + "\n")

    def reset(self, task, observation, *, episode, plan_path=None):
        self.ready = False
        if not task.strip() or not episode:
            raise ValueError("A task and unique episode identity are required")
        self.task, self.episode = task, episode
        self.step = self.index = self.streak = 0
        self.actions = deque()
        self.reference = self._images(observation)
        if not self.reference:
            raise ValueError("At least one camera is required")
        fingerprint = hashlib.sha256(json.dumps(self.reference).encode()).hexdigest()
        identity = {"task": task, "episode": episode, "initial_images_sha256": fingerprint,
                    "image_size": self.settings.image_size}
        path = Path(plan_path) if plan_path else None
        if path and path.exists():
            record = json.loads(path.read_text())
            if any(record.get(k) != v for k, v in identity.items()):
                raise ValueError("Cached plan does not match this task, episode and initial images")
            self.plan = parse_plan(json.dumps(record["subgoals"]))
        else:
            self.plan = parse_plan(json.dumps(self.planner.plan(task, self.reference)))
            record = {**identity, "subgoals": self.plan,
                      "planner_model": getattr(self.planner, "model", "custom")}
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive creation prevents silently replacing a frozen evaluation plan.
                with path.open("x") as stream:
                    json.dump(record, stream, indent=2)
        reset = getattr(self.policy, "reset", None)
        if reset:
            reset()
        self.ready = True
        self._log("reset", plan=record, settings=asdict(self.settings))

    @property
    def exhausted(self):
        return self.ready and self.index == len(self.plan)

    def act(self, observation):
        if not self.ready:
            raise RuntimeError("Call reset before act")
        if self.exhausted:
            raise StopIteration("Plan exhausted; evaluate success independently")
        if self.step and self.step % self.settings.poll_interval == 0:
            current = self._images(observation)
            started = time.monotonic()
            try:
                complete, raw = self.monitor.judge(self.plan[self.index], self.reference, current)
            except Exception as exc:
                self.ready = False
                self._log("monitor_error", error=type(exc).__name__)
                raise
            self.streak = self.streak + 1 if complete is True else 0
            self._log("poll", complete=complete is True, raw=raw, streak=self.streak,
                      elapsed_seconds=time.monotonic() - started)
            if self.streak >= self.settings.confirmations:
                self.index += 1
                self.streak = 0
                self.reference = current
                self.actions.clear()  # Never execute old-subgoal actions after a transition.
                self._log("advance", exhausted=self.exhausted)
                if self.exhausted:
                    raise StopIteration("Plan exhausted; evaluate success independently")
        if not self.actions:
            prompt = self.settings.prompt_template.format(task=self.task, subgoal=self.plan[self.index])
            actions = np.asarray(self.policy.predict(self.adapter.observation(observation), prompt))
            if (actions.ndim != 2 or not len(actions) or actions.shape[1] != self.adapter.action_dim
                    or not np.isfinite(actions).all()):
                raise ValueError("Policy must return finite [horizon, adapter.action_dim] actions")
            self.actions.extend(actions[:self.settings.action_horizon].copy())
        action = self.adapter.action(self.actions.popleft())
        self.step += 1
        return action
