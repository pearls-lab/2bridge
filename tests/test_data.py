import numpy as np
import pytest
from twobridge.dataset import LeRobotData
from twobridge.preprocessing import compute_statistics


@pytest.mark.parametrize("version", ["v2.0", "v2.1", "v3.0"])
@pytest.mark.parametrize("dims", [(5, 3), (16, 7), (32, 32)])
def test_layouts_boundaries_and_train_only_stats(make_dataset, version, dims):
    config = make_dataset(version, *dims)
    data = LeRobotData(config)
    assert data[0]["images"].shape == (1, 224, 224, 3)
    assert data[1]["prompt"] == "place object"
    final = data[3]
    assert final["action_mask"].tolist() == [True, False, False]
    assert np.all(final["actions"] == data.actions[3])
    assert final["actions"].max() < 1000
    stats = compute_statistics(data)
    assert stats.action_high.max() < 1000
    assert stats.state_high.max() < 1000
    assert LeRobotData(config, "val")[0]["state"][0] == 1000


def test_v3_shared_video_timestamp_offsets(make_dataset):
    c = make_dataset(video=True)
    first, second = LeRobotData(c)[0], LeRobotData(c, "val")[0]
    assert first["images"].mean() < -0.7
    assert second["images"].mean() > 0.6


def test_prompt_column_and_relative_origin(make_dataset):
    c = make_dataset(prompt_column=True)
    c.dataset.relative_pairs = [[0, 2]]
    d = LeRobotData(c)
    assert d[1]["prompt"] == "frame 1"
    np.testing.assert_allclose(d[1]["actions"][:, 0], [-1, 1, 3])


def test_unknown_episode_overlap_and_unknown_feature(make_dataset):
    c = make_dataset()
    c.dataset.train_episodes = [1]
    with pytest.raises(ValueError, match="disjoint"):
        c.validate()
    c.dataset.train_episodes = [12]
    with pytest.raises(ValueError, match="Unknown episode"):
        LeRobotData(c)
    c.dataset.train_episodes = [0]
    c.dataset.state_key = "wrong"
    with pytest.raises(ValueError, match="numeric vector"):
        LeRobotData(c)


def test_missing_prompt_fails_instead_of_fallback(make_dataset):
    c = make_dataset()
    from pathlib import Path

    (Path(c.dataset.root) / "meta/tasks.jsonl").write_text("")
    with pytest.raises(ValueError, match="instruction"):
        LeRobotData(c)
    c.dataset.default_prompt = "perform the demonstrated task"
    assert LeRobotData(c)[0]["prompt"] == c.dataset.default_prompt


def test_bad_crop_and_feature_shape(make_dataset):
    c = make_dataset()
    c.dataset.cameras[0].crop = [0, 0, 200, 200]
    with pytest.raises(ValueError, match="exceeds"):
        LeRobotData(c)[0]
    c.dataset.cameras[0].crop = [5, 0, 1, 2]
    with pytest.raises(ValueError, match="crop"):
        c.validate()


def test_tasks_parquet_index_and_per_frame_instructions(make_dataset):
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq

    c = make_dataset()
    root = Path(c.dataset.root)
    (root / "meta/tasks.jsonl").unlink()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"task_index": 0, "__index_level_0__": "reach"},
                {"task_index": 1, "__index_level_0__": "release"},
            ]
        ),
        root / "meta/tasks.parquet",
    )
    d = LeRobotData(c)
    assert d[0]["prompt"] == "reach" and d[1]["prompt"] == "release"


def test_explicit_legacy_schema_and_external_images(make_dataset):
    from pathlib import Path
    from PIL import Image
    import json

    c = make_dataset("v2.0")
    root = Path(c.dataset.root)
    info = json.loads((root / "meta/info.json").read_text())
    c.dataset.feature_overrides = info.pop("features")
    (root / "meta/info.json").write_text(json.dumps(info))
    c.dataset.cameras[0].image_path = "images/{episode_index}/{frame_index}.png"
    (root / "images/0").mkdir(parents=True)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(root / "images/0/0.png")
    sample = LeRobotData(c)[0]
    assert sample["images"][0, :, :, 0].mean() == 1
    assert sample["images"][0, :, :, 1].mean() == -1
