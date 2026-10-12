# Dataset mapping

The dataset root must contain `meta/info.json`. Inspect its `features` dictionary
and configure `state_key`, `action_key` and camera keys verbatim. Do not rename
files or relabel every dataset into a RoboCasa schema.

For example, a seven-dimensional action robot with sixteen proprioceptive values
works by setting its vector feature names; neither dimension needs to be six and
the state dimension does not have to equal the action dimension. Up to 32 values
in each vector are retained, normalized, then zero-padded to the π0.5 model width.
Select at most three RGB views explicitly so no camera is dropped silently.

## Language

By default, each frame's own `task_index` is looked up in `meta/tasks.jsonl` or
`meta/tasks.parquet`. The task is not inferred from an episode's first instruction,
so primitive/atomic/composite labels can change within the same episode. A named
`prompt_key` takes precedence and must identify a string Parquet column. Missing
instructions fail validation unless `default_prompt` is explicitly supplied.

For parquet task metadata, either a `task` column or the pandas string-index column
`__index_level_0__` is accepted with an integer `task_index` column.

## Storage

v2 per-episode path templates are resolved from `info.json`; v3 data files can
contain multiple episodes. Every row is indexed by its episode/frame metadata.
The loader clamps horizon reads at episode boundaries and marks repeated tail
targets invalid for loss calculation.

For v3 video, the episode's `videos/<key>/chunk_index`, `file_index`, and
`from_timestamp` in `meta/episodes/**/*.parquet` select the video shard and time
origin. The frame's relative timestamp is added to that origin. The nearest
decoded frame must be within 0.51 / FPS seconds; missing/corrupt video raises an
error instead of silently returning an unrelated frame. RGB image features use
embedded image bytes or dataset-relative image paths.

The expected layouts follow the
[LeRobotDataset v3 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).
The metadata adapter also supports v2.0 and v2.1. This release reads full local
snapshots; Hub streaming and Lance storage are not supported.

## Optional image transforms

Each `[[dataset.cameras]]` can specify:

- `resize`: `stretch` (square squash) or `letterbox` (preserve aspect ratio).
- `crop`: source-pixel `[x0, y0, x1, y1]`, applied before resizing.
- `jpeg_quality`: optional JPEG round trip after resize, disabled by default.

Transforms are saved with the checkpoint and used by `Policy.predict` too. These
are new release transform semantics; historical JPEG-before-resize conversions
are not reproduced by adding `jpeg_quality`. Already converted images should not
be recompressed. Camera names do not imply mirroring, undistortion or handedness.

## Relative action pairs

`relative_pairs = [[0, 2], [1, 3]]` subtracts state dimensions 2 and 3 from action
dimensions 0 and 1 respectively, for every action in a horizon. Unlisted dimensions
(such as gripper or velocity channels) remain unchanged. This requires semantically
compatible state/action units. Inference adds the observation-time state back.

## Common errors

- **Unknown feature**: check `meta/info.json`, not the camera's physical nickname.
- **No tasks**: provide the correct task metadata, a string `prompt_key`, or an
  intentional fixed `default_prompt`.
- **No video frame near timestamp**: check FPS, v3 video offsets and missing shards.
- **Dimension above 32**: select/transform the state and action representation
  explicitly before training; the model cannot represent arbitrary-width vectors.
- **Prompt too long**: increase `max_token_len`; prompt/state tokens are not truncated.
- **Resume mismatch**: resume with the original config and unchanged dataset, or
  initialize a new run from the saved weights.
