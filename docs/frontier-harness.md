# Frontier planner, local monitor, VLA policy

The release includes a deployable-observation harness for RoboCasa Gym (PandaOmron)
and BEHAVIOR-1K / OmniGibson (R1Pro). A frontier model creates an ordered plan from
the task and initial RGB views **once per episode**. A separate local Qwen monitor
compares the current subgoal's start views with current views. The policy executes
`<task> Now: <subgoal>`. No object poses, simulator predicates, oracle subgoal indices,
or task-success flags enter the planner or monitor.

This implements planner–monitor–policy, rather than a two-model replanning loop.
The planner model ID is explicitly configured; no provider/model availability is
assumed by the release. Native Anthropic Messages and OpenAI-compatible chat
completion endpoints are supported. The local monitor supports the original Qwen
HTTP completion protocol and compatible chat completion servers.

## Processes and installation

Use separate policy, monitor, and simulator environments. The policy pins
Transformers 4.53.2; **do not install its full dependencies into the Qwen or
OmniGibson environment**. The simulator only needs the harness source, NumPy,
Pillow, and (for remote inference) `websockets>=15,<16` and `msgpack>=1,<2`.

In the policy environment, from the release root:

```bash
python -m pip install -e '.[harness]'
python -m twobridge.harness.policy_server \
  --checkpoint /path/to/checkpoint --device cuda:0 --port 8000
```

The policy server returns action chunks in the checkpoint dataset's original
units, including restoration of configured relative action dimensions. It uses the
OpenPI websocket wire format without requiring an OpenPI checkout. An existing
compatible policy server can also be used, provided its input features, action
space, prompt formatting, and preprocessing agree with the adapter.

In a separate Qwen-compatible environment (model support must be supplied by that
environment's Transformers version), expose this release's `src` on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/2bridge/src
python -m twobridge.harness.monitor_server \
  --model-path /path/to/Qwen3.6-27B --device cuda:0 --port 8001
```

This loads `AutoModelForImageTextToText` in bfloat16, disables thinking via
`enable_thinking=False`, and generates at most eight tokens greedily. It returns
`complete: true` only for an exact case-insensitive `yes`. The health endpoint is
`/healthz`; model inference is serialized while health requests remain responsive.
Both shipped servers bind to localhost by default; use an SSH tunnel for remote
simulator hosts or explicitly choose the listening interface.

An already running compatible vLLM monitor can instead use:

```toml
[monitor]
protocol = "chat"
url = "http://127.0.0.1:8001/v1/chat/completions"
model = "your-served-qwen-model-name"
timeout = 120
```

The client sends `chat_template_kwargs.enable_thinking=false`, START images first,
then NOW images, with the visible-state-change hint inside the subgoal text. Ensure
your serving backend honors this chat-template option. The legacy protocol remains
compatible with `qwen36_instructor_server.py`.

## RoboCasa

Copy and edit `configs/harness_robocasa.toml`. Set a frontier model ID available to
your account and export `ANTHROPIC_API_KEY`. Set the task instruction to the actual
scene's task; the fruit-bowl sentence in the example is illustrative. Set policy
feature keys to the checkpoint's dataset keys. The sample slash-separated keys
match the historical OpenPI RoboCasa endpoint; a release checkpoint may instead
use dotted LeRobot keys.

Run inside the installed RoboCasa Gym environment:

```bash
export PYTHONPATH=/path/to/2bridge/src
MUJOCO_GL=egl python -m twobridge.harness.runner /path/to/harness_robocasa.toml
```

Defaults reproduce the documented observation/polling settings: 512px camera
observations, polling every 40 executed control steps, two consecutive yes answers.
The start references change only when the cursor advances. A no answer resets the
confirmation streak; errors abort the rollout. Old action chunks are cleared on
advance, including when the switch occurs midway through a chunk.

The adapter expects the Gym wrapper's three upright cameras and state16
`[base position3, base quaternion4, relative EEF position3, relative EEF quaternion4,
gripper joints2]`. It sends arm action7 as position3/rotation3/gripper1 and zeroes
the base and control-mode fields, matching the stationary-base research setup.
It does not support mobile-base policies without a different action adapter.

## BEHAVIOR-1K

Copy and edit `configs/harness_behavior.toml`. Provide an OmniGibson YAML for the
specific task, scene instance, R1Pro robot and controllers from your installation;
scene assets and licensed benchmark data are not distributed here. The standalone
runner constructs `omnigibson.Environment(configs=...)`. It is a rollout entry point,
not a replacement for the official benchmark's instance scheduler/scoring.

```bash
export PYTHONPATH=/path/to/2bridge/src
python -m twobridge.harness.runner /path/to/harness_behavior.toml
```

The adapter accepts native nested OmniGibson observations or the evaluator's
flattened `robot_r1::...` keys. Its defaults select head/left-wrist/right-wrist RGB,
project raw proprioception into state25, and return native action23 as a tensor.
State25 is base velocity3, torso4, left arm7, left fingers2, right arm7, right
fingers2. **The dataset/checkpoint must use that same projection and native action
representation.** Configure `state_indices`, camera/feature keys and `action_dim`
for another supported layout. A historical server that performs its own raw-state
projection needs the full raw proprio vector instead; explicitly list its indices
and verify that server's contract. These adapters do not make arbitrary trained
robot embodiments interchangeable.

BEHAVIOR's sample uses the existing serving code's 100-step polling default, two
confirmations and a 16-action execution horizon. Configure sensor resolution in
the scene YAML to capture at least 512px if matching high-resolution monitoring;
resizing a 224px evaluator frame cannot restore missing detail. The harness pads
all planner/monitor views to 512px while leaving policy preprocessing to the
checkpoint. Use one controller per environment instance; batched observation
arrays are intentionally rejected.

For the official evaluator, create a `Harness` with `BehaviorAdapter` and attach it
to the evaluator's policy/reset lifecycle after its normal instance loading. The
integration itself is ordinary Python:

```python
from twobridge.harness import Harness, Settings
from twobridge.harness.adapters import BehaviorAdapter

# planner, monitor and policy implement the same interfaces used by runner.py.
harness = Harness(policy, planner, monitor, BehaviorAdapter(),
                  Settings(poll_interval=100, confirmations=2, action_horizon=16),
                  trace="runs/task-i003/trace.jsonl")
# Obtain initial_obs from the evaluator AFTER loading and resetting instance 3.
harness.reset(task_instruction, initial_obs, episode="task/instance-3",
              plan_path="plans/task-i003.json")
# Once per executed control step:
action = harness.act(current_obs)
# Pass action to the evaluator's normal env.step path. StopIteration means the
# monitor exhausted the plan; record the evaluator's own task-success metric.
```

Use the official evaluator's seeding and `load_task_instance` procedures for scored
runs. Do not replace those with task-only plans or assume a YAML scene reset is the
same instance as an official evaluation episode.

## Frozen plans, traces and outcomes

The first reset writes a plan cache containing task, episode identity, planner model,
image size, an initial-image SHA256 and the immutable ordered subgoals. Subsequent
resets reuse it without calling the frontier API. A mismatched task, episode, or
initial image fingerprint is rejected. Share the cache across experiment arms
only when they begin from identical observations. Cache files are exclusively
created and never overwritten by the harness.

Each run gets a fresh output directory containing the effective config, append-only
JSONL poll/advance/error trace, and result JSON. Credentials are environment-variable
names in the config, never literal keys. Existing run directories containing a
config are rejected to prevent mixing traces. Relative paths resolve from the
TOML file. Poll logs include raw monitor output, streaks, latency and step indices.
Monitor failures propagate rather than counting as yes or silently continuing.

`plan_exhausted`, environment termination, and `environment_success` are distinct.
The last field is only copied from a boolean `info['success']`, otherwise null;
it is never inferred from the monitor, reward, termination, or elapsed time. For
BEHAVIOR benchmark reporting, use the official evaluator's scoring output.
The final subgoal is polled, and an exhausted plan stops action execution.

## Validation scope

Automated tests cover frozen cache reuse/mismatch, image-reference changes,
confirmation streaks, chunk invalidation, final-plan exhaustion, reset isolation,
strict yes/no parsing, transport failures, adapter layouts, real local HTTP requests,
and websocket action transport. They do not establish benchmark task success or
frontier/Qwen judgment accuracy. No paid frontier calls are required by tests.
