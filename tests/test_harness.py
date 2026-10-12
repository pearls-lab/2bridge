import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import numpy as np
import pytest

from twobridge.harness import Harness, Settings
from twobridge.harness.adapters import BehaviorAdapter, RoboCasaAdapter
from twobridge.harness.clients import FrontierPlanner, VisualMonitor, encode_image, parse_plan, parse_yes
from twobridge.harness.monitor_server import make_handler
from twobridge.harness.runner import rollout


class Adapter:
    action_dim = 2

    def images(self, obs):
        return [obs["rgb"]]

    def observation(self, obs):
        return {"state": obs["state"]}

    def action(self, value):
        return value


class Planner:
    def __init__(self):
        self.calls = 0

    def plan(self, task, images):
        self.calls += 1
        return ["open door", "place apple"]


class Monitor:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def judge(self, goal, start, now):
        self.calls.append((goal, start, now))
        answer = next(self.answers)
        return answer, "yes" if answer else "no"


class Policy:
    def __init__(self):
        self.prompts = []

    def predict(self, obs, prompt):
        self.prompts.append(prompt)
        return np.full((20, 2), len(self.prompts), dtype=np.float32)


def obs(value=0):
    return {"rgb": np.full((8, 8, 3), value, dtype=np.uint8), "state": np.zeros(2),
            "secret_simulator_success": True}


def test_frozen_plan_confirmations_reference_and_chunk_flush(tmp_path):
    planner, policy, monitor = Planner(), Policy(), Monitor([True, False, True, True, True, True])
    harness = Harness(policy, planner, monitor, Adapter(), Settings(2, 2, 16, 20), tmp_path / "trace")
    harness.reset("task", obs(), episode="i0", plan_path=tmp_path / "plan")
    for step in range(12):
        action = harness.act(obs(step))
        assert action[0] == (1 if step < 8 else 2)
    with pytest.raises(StopIteration):
        harness.act(obs(12))
    assert harness.exhausted
    assert planner.calls == 1
    assert len(policy.prompts) == 2
    assert all(call[1] == monitor.calls[0][1] for call in monitor.calls[:4])
    assert monitor.calls[4][1] == monitor.calls[3][2]
    events = [json.loads(line) for line in (tmp_path / "trace").read_text().splitlines()]
    assert [e["step"] for e in events if e["event"] == "advance"] == [8, 12]
    harness.reset("task", obs(), episode="i0", plan_path=tmp_path / "plan")
    assert planner.calls == 1 and not harness.exhausted and harness.step == 0
    with pytest.raises(ValueError, match="Cached plan"):
        harness.reset("task", obs(1), episode="i0", plan_path=tmp_path / "plan")
    with pytest.raises(RuntimeError):
        harness.act(obs())


@pytest.mark.parametrize("raw", ["yes indeed", "yesterday", "<think>yes</think>", "false", "", "no"])
def test_strict_judgment(raw):
    assert not parse_yes(raw)


def test_parse_validation():
    assert parse_yes(" YES ")
    assert parse_plan('```json\n{"subgoals":["open"]}\n```') == ("open",)
    for text in ['[]', '[""]', '[1]', '{}']:
        with pytest.raises(ValueError):
            parse_plan(text)
    with pytest.raises(ValueError):
        Settings(poll_interval=0)


def test_adapter_contracts():
    casa = RoboCasaAdapter()
    raw = dict(zip(casa.state_keys, [np.zeros(n) for n in [3, 4, 3, 4, 2]], strict=True))
    raw.update({key: obs()["rgb"] for key in casa.cameras})
    assert casa.observation(raw)["observation.state"].shape == (16,)
    assert np.array_equal(casa.action(np.arange(7))["action.gripper_close"], [6])
    behavior = BehaviorAdapter()
    raw = {key: np.zeros((8, 8, 4), dtype=np.uint8) for key in behavior.camera_keys}
    raw[behavior.proprio_key] = np.arange(57)
    assert behavior.observation(raw)["observation.state"].shape == (25,)
    nested = {"robot_r1": {k.removeprefix("robot_r1::"): v for k, v in raw.items()}}
    np.testing.assert_equal(behavior.observation(raw), behavior.observation(nested))
    assert behavior.action(np.arange(23)).shape == (23,)


@pytest.fixture
def server():
    running = []

    def start(handler):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        running.append((httpd, thread))
        return f"http://127.0.0.1:{httpd.server_port}"

    yield start
    for httpd, thread in running:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_legacy_monitor_http(server):
    class Judge:
        def judge(self, data):
            assert data["n_ref"] == 1
            assert "visible state change" in data["subgoal"]
            return {"complete": True, "raw": "yes"}

    url = server(make_handler(Judge()))
    monitor = VisualMonitor(url)
    image = encode_image(obs()["rgb"], 16)
    assert monitor.judge("open", [image], [image]) == (True, "yes")


def test_chat_planner_and_nonthinking_monitor_http(server):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            text = '{"subgoals":["open"]}' if len(requests) == 1 else "yes"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": text}}]}).encode())

    url = server(Handler)
    image = encode_image(obs()["rgb"], 16)
    assert FrontierPlanner(url, "frontier").plan("task", [image]) == ("open",)
    assert VisualMonitor(url, protocol="chat").judge("open", [image], [image])[0]
    assert requests[1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "state" not in requests[0]


def test_rollout_exhaustion_is_not_success():
    class Env:
        def reset(self):
            return obs(), {}

        def step(self, action):
            return obs(), 0, False, False, {"success": False}

    h = Harness(Policy(), Planner(), Monitor([True] * 4), Adapter(), Settings(1, 1, 16, 8))
    result = rollout(Env(), h, "task", "i0", None, 10)
    assert result["stop_reason"] == "plan_exhausted" and result["steps"] == 2
    assert "success" not in result


def test_monitor_failure_aborts_without_advancing():
    class Broken:
        def judge(self, *args):
            raise TimeoutError("offline")

    h = Harness(Policy(), Planner(), Broken(), Adapter(), Settings(1, 1, 16, 8))
    h.reset("task", obs(), episode="i0")
    h.act(obs())
    with pytest.raises(TimeoutError):
        h.act(obs())
    assert h.index == 0
    with pytest.raises(RuntimeError):
        h.act(obs())


def test_websocket_policy_transport():
    import asyncio
    from twobridge.harness.policy_server import make_handler as policy_handler
    from twobridge.harness.runner import RemotePolicy
    from websockets.asyncio.server import serve

    ready = threading.Event()
    state = {}
    policy = Policy()
    policy.action_dim = 2

    async def run():
        async with serve(policy_handler(policy), "127.0.0.1", 0) as ws:
            state["port"] = ws.sockets[0].getsockname()[1]
            state["loop"] = asyncio.get_running_loop()
            state["stop"] = asyncio.Event()
            ready.set()
            await state["stop"].wait()

    thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    thread.start()
    assert ready.wait(5)
    client = None
    try:
        client = RemotePolicy("127.0.0.1", state["port"])
        actions = client.predict({"state": np.zeros(2), "rgb": obs()["rgb"]}, "open")
        assert actions.shape == (20, 2)
        assert policy.prompts == ["open"]
    finally:
        if client:
            client.close()
        state["loop"].call_soon_threadsafe(state["stop"].set)
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_anthropic_planner_request(server, monkeypatch):
    monkeypatch.setenv("TEST_FRONTIER_KEY", "test-key")
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            assert self.headers["x-api-key"] == "test-key"
            assert self.headers["anthropic-version"] == "2023-06-01"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"content": [
                {"type": "text", "text": '{"subgoals":["rinse the apple"]}'}]}).encode())

    planner = FrontierPlanner(server(Handler), "frontier", provider="anthropic", key_env="TEST_FRONTIER_KEY")
    assert planner.plan("Wash apple", [encode_image(obs()["rgb"], 16)]) == ("rinse the apple",)
    assert captured["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_wire_rejects_object_arrays():
    import msgpack
    from twobridge.harness.wire import pack, unpack

    with pytest.raises(ValueError):
        pack(np.array([object()]))
    with pytest.raises(ValueError):
        unpack(msgpack.packb({b"__ndarray__": True, b"dtype": "O", b"data": b"", b"shape": [0]}))
