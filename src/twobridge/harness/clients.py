"""Dependency-light vision clients. Credentials come from environment variables."""

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np
from PIL import Image, ImageOps

SYSTEM = (
    "You are a robot task progress judge. You see reference camera images from "
    "the START of a robot's current subtask and camera images of the scene NOW. "
    "Decide whether the subtask has been completed by comparing the two. "
    "Calibration note: judging absolute object placement from a single view is "
    "unreliable; judge the CHANGE between start and now. Respond with exactly "
    "one word: yes or no."
)
HINT = (
    " (judge by the visible state change: if this subtask were completed, the "
    "NOW images should show its end result compared to the START images — e.g. "
    "objects in new locations, doors/lids opened or closed, appliances switched "
    "on or off)"
)
PLANNER_SYSTEM = (
    'Plan executable robot subgoals from the task and initial camera images. '
    'Return only JSON: {"subgoals": ["instruction", ...]}. '
    'Use visible objects and explicit destinations. Include sustained actions such as rinsing '
    'when required. Do not claim access to hidden simulator state. The plan is executed once '
    'in order; each step must have a visually assessable completion condition.'
)


def encode_image(array, size=512):
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Vision inputs must be HWC RGB uint8")
    image = ImageOps.pad(Image.fromarray(array), (size, size), method=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def post(url, payload, *, key_env=None, timeout=120, headers=None):
    headers = {"Content-Type": "application/json", **(headers or {})}
    if key_env:
        headers.setdefault("Authorization", "Bearer " + os.environ[key_env])
    request = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_plan(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    value = json.loads(text)
    goals = value.get("subgoals") if isinstance(value, dict) else value
    if not isinstance(goals, list) or not 1 <= len(goals) <= 100:
        raise ValueError("Plan must contain 1–100 subgoal strings")
    if any(not isinstance(s, str) or not s.strip() or len(s) > 2000 for s in goals):
        raise ValueError("Invalid subgoal")
    return tuple(s.strip() for s in goals)


def completion_text(subgoal):
    return (f"The robot's current subtask is: {subgoal}{HINT}\n"
            "Comparing the start images to the current images, has the robot "
            "completed this subtask? Answer with exactly one word: yes or no.")


def parse_yes(raw):
    # Invalid/verbose output is not a confirmation. Never bool("false").
    return raw.strip().lower() == "yes"


def image_blocks(images):
    return [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + x}} for x in images]


class FrontierPlanner:
    def __init__(self, url, model, provider="chat", key_env=None, timeout=180):
        if provider not in {"chat", "anthropic"}:
            raise ValueError("provider must be chat or anthropic")
        self.url, self.model, self.provider = url, model, provider
        self.key_env, self.timeout = key_env, timeout

    def plan(self, task, images):
        if self.provider == "anthropic":
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                        "data": x}} for x in images]
            content.append({"type": "text", "text": task})
            headers = {"anthropic-version": "2023-06-01"}
            if self.key_env:
                headers["x-api-key"] = os.environ[self.key_env]
            result = post(self.url, {"model": self.model, "max_tokens": 4096,
                          "system": PLANNER_SYSTEM, "messages": [{"role": "user", "content": content}]},
                          timeout=self.timeout, headers=headers)
            raw = "".join(x["text"] for x in result["content"] if x["type"] == "text")
        else:
            result = post(self.url, {"model": self.model, "max_tokens": 4096,
                          "messages": [{"role": "system", "content": PLANNER_SYSTEM},
                                       {"role": "user", "content": image_blocks(images)
                                        + [{"type": "text", "text": task}]}]},
                          timeout=self.timeout, key_env=self.key_env)
            raw = result["choices"][0]["message"]["content"]
        return parse_plan(raw)


class VisualMonitor:
    def __init__(self, url, model="qwen3.6-27b", protocol="legacy", key_env=None, timeout=120):
        if protocol not in {"legacy", "chat"}:
            raise ValueError("protocol must be legacy or chat")
        self.url, self.model, self.protocol = url, model, protocol
        self.key_env, self.timeout = key_env, timeout

    def judge(self, subgoal, start, current):
        if not start or len(start) != len(current):
            raise ValueError("START and NOW must have matching ordered camera views")
        if self.protocol == "legacy":
            payload = {"query_type": "completion", "images": [*start, *current],
                       "n_ref": len(start), "subgoal": subgoal + HINT}
        else:
            content = [{"type": "text", "text": "START images:"}, *image_blocks(start),
                       {"type": "text", "text": "NOW images:"}, *image_blocks(current),
                       {"type": "text", "text": completion_text(subgoal)}]
            payload = {"model": self.model, "max_tokens": 8, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": content}]}
        result = post(self.url, payload, timeout=self.timeout, key_env=self.key_env)
        if self.protocol == "legacy":
            raw = result["raw"]
            return result.get("complete") is True and parse_yes(raw), raw
        raw = result["choices"][0]["message"]["content"]
        return parse_yes(raw), raw
