# Policy training and inference

Training guide for [2Bridge](../README.md). Run the commands below from the repository root.

This guide covers fine-tuning and running a pretrained **π0.5** policy on robot
demonstrations stored in LeRobot format. Configure the dataset, camera streams,
robot state and action fields, language instructions, image preprocessing, and
episode splits in a TOML file.

Start with the [generic configuration](../configs/generic.toml) for your task.
The [crossword configuration](../configs/crossword.toml) is an example
from the paper.

## Install

Use Python **3.11 or 3.12** in a dedicated environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,harness]'
python -m pytest -q
```

PyTorch 2.7.1, torchvision 0.22.1 and Transformers 4.53.2 are pinned because the
OpenPI backend uses matching adaptive-normalization implementations. Those modules
are bundled and loaded **within the training process**; installation does not
patch your `site-packages`. Use a dedicated process for this policy rather than
mixing it with unrelated Transformers models. JAX and LeRobot's Python package
are not dependencies: the loader reads the LeRobot storage format directly.

Full fine-tuning needs a CUDA GPU with substantial VRAM; reduce the per-device
batch size and use accumulation if necessary. `batch_size × accumulation_steps ×
world_size` is the effective batch size. CPU float32 is supported for verification,
but full π0.5 training is not practical on ordinary CPUs.

## Prepare your dataset and pretrained assets

1. Put a LeRobot dataset on disk, including `meta/info.json`, task metadata,
   Parquet data, and image/video files. Existing datasets are read without rewriting.
2. Obtain a **compatible OpenPI π0.5 PyTorch** checkpoint (`model.safetensors`)
   and the matching PaliGemma SentencePiece tokenizer. Follow the
   [OpenPI PyTorch setup/conversion instructions](https://github.com/Physical-Intelligence/openpi).
   JAX checkpoints and Hugging Face LeRobot policy checkpoints have different layouts
   and cannot be loaded directly. Weights and tokenizer are not bundled here.
3. Create a configuration for your dataset:

   ```bash
   cp configs/generic.toml configs/my_dataset.toml
   ```

   Set the dataset, checkpoint, and tokenizer paths, and use the feature names from
   your dataset's `meta/info.json`. Configure the camera streams and held-out
   episode IDs. Paths resolve relative to the TOML file.

Optional explicit Hub download (no upload occurs):

```bash
python -m pip install -e '.[hub]'
2bridge download --repo-id YOUR_ORG/YOUR_DATASET \
  --revision DATASET_COMMIT_SHA --root datasets/my_dataset
```

Supported dataset formats and inputs:

| Contract | Support |
|---|---|
| Storage versions | LeRobot v2.0, v2.1 and v3.0 |
| Parquet layout | Per-episode files or shared shards, including episode boundaries inside a shard |
| Visual storage | Embedded image bytes, dataset-relative image paths, or timestamp-indexed video |
| State and actions | Configurable numeric vector fields, each 1–32 dimensions |
| Camera inputs | Explicitly map 1–3 RGB streams; choose a subset if your dataset has more |
| Instructions | Per-frame `task_index`, an explicit string column, or an explicitly configured fixed prompt |
| Action semantics | Recorded values preserved; optional explicitly paired action/state relative dimensions |
| Dataset access | Local complete datasets, including explicitly downloaded Hub snapshots |

Missing camera slots are masked. Extra unused dataset features are ignored.
Depth, point clouds, audio, non-vector actions, dimensions above 32, streaming,
LeRobot v1 datasets, and newer unrecognized storage versions require an adapter. Frame indices
must start at zero and be contiguous within each episode, with relative timestamps
at the declared FPS. The numeric index is loaded into RAM; this is not a streaming
loader for arbitrarily large corpora. Video is decoded on demand with PyAV.

## Inspect and train

```bash
2bridge inspect --config configs/my_dataset.toml --samples 8
2bridge train --config configs/my_dataset.toml
```

`inspect` validates schema, instructions, splits and numeric values, computes
training-only normalization, and decodes evenly spaced sample frames. It also checks
tokenization when the configured tokenizer exists. It never loads model weights.
For a complete visual decoding preflight, set `--samples` to the training frame count.

Set `val_episodes` to held-out episode IDs. If `train_episodes` is omitted, training
uses all remaining episodes. Explicit train/validation overlap is rejected.
Hold out complete scenes/layouts or collection sessions where appropriate;
random frames from the same demonstration are not an independent validation set.
Validation reports flow/derivative loss, not task success.

The default recipe has horizon 10, 8,000 steps, batch 32, 500 warmup steps and
cosine LR from 2.5e-5 to 2.5e-6. State and actions use training-split q01/q99
normalization estimated deterministically from at most 8,192 evenly spaced
training samples. Action statistics use valid horizon targets. State is retained
in the **discrete π0.5 prompt tokens**. Long prompts raise an error instead of
silently truncating state. Inputs use RGB; neither BGR swapping nor undistortion
happens implicitly. Geometric augmentation is off; training uses photometric
augmentation. Crop/resize/JPEG settings are shared with inference.

The optional I1a loss matches first/second finite differences of the flow model's
implied clean action, weighted by `(1-t)`. Enable it with `deriv_weight` and explicit
`deriv_dims`. Padded dimensions and targets beyond episode ends are excluded from
both flow and derivative losses. Relative actions are offsets from the state at
observation time, not cumulative deltas between successive actions.

Multi-GPU launch:

```bash
torchrun --standalone --nproc_per_node=2 -m twobridge.cli train \
  --config configs/my_dataset.toml
```

Single-device and torchrun/DDP use the same loop.

## Checkpoints and resume

Each `runs/.../checkpoints/step_XXXXXXXX/` contains:

- `model.safetensors`: model weights, loaded strictly.
- `training_state.pt`: optimizer, step, sampler position and per-rank Torch RNG.
- `config.json`, `dataset.json`, `manifest.json`: resolved configuration and provenance.
- `stats.json`, `tokenizer.model`: the complete preprocessing assets needed for inference.

Checkpoints are atomically renamed into place only after writing. Interrupted
`.tmp` directories are never offered as complete checkpoints. Existing output
runs are not overwritten. All completed checkpoints are retained; allow enough
disk space for both weights and optimizer state.

```bash
2bridge train --config configs/my_dataset.toml \
  --resume runs/my_policy/checkpoints/step_00008000
```

Resume the latest complete checkpoint with the original config, data, tokenizer
and world size. The data fingerprint checks numeric records, selected prompts,
episode metadata and Parquet file size/mtime; it does not hash entire video files.
Do not change video assets during a run or resume. Resume is for continuation to
the originally configured step count, not schedule extension. For a new fine-tune,
start a new output directory and use a previous `model.safetensors` as initialization.

## Inference

```python
from twobridge.policy import Policy

policy = Policy("runs/my_policy/checkpoints/step_00008000", device="cuda")
actions = policy.predict(
    {
        "observation.state": measured_state,          # float vector, original units
        "observation.images.front": front_rgb,       # HWC uint8 RGB
        "observation.images.wrist": wrist_rgb,
    },
    prompt="Place the object in the bowl",
)
# actions.shape == (horizon, dataset_action_dim), in the recorded action units.
```

Use the same feature names as the training config. A copied checkpoint directory
is sufficient: inference does not read the training dataset or original pretrained
checkpoint. Relative dimensions are restored to absolute commands against the
provided measured state. The library returns action arrays; it does not control hardware.

## Crossword example

`configs/crossword.toml` preserves the historical six-joint, three-image-slot
recipe as one example. Already converted crossword datasets can be used directly.
For raw datasets, explicitly configure the two cameras and an optional third crop
(e.g. duplicate the overhead key with `crop = [x0,y0,x1,y1]`); choose coordinates
for the actual camera placement. No historical crop, frame trimming, calibration,
plan-CSV relabeling, marker drawing, or board geometry is applied automatically.
Ensure the dataset's per-frame task annotations describe what was demonstrated.

## Frontier harness: RoboCasa and BEHAVIOR-1K

The release also includes a **frontier planner → local Qwen monitor → VLA policy**
harness. Plans are frozen per episode. The monitor compares subgoal-start and
current RGB views, and advances only after consecutive confirmations. Simulator
success is tracked separately from monitor plan exhaustion.

See [setup and execution](frontier-harness.md),
[RoboCasa configuration](../configs/harness_robocasa.toml), and
[BEHAVIOR configuration](../configs/harness_behavior.toml). The harness supplies
benchmark adapters, HTTP planner/monitor clients, a standalone Qwen monitor
server, an OpenPI-compatible policy server, cached plans, and JSONL traces.
Policy, monitor and simulator can run in separate Python environments.
