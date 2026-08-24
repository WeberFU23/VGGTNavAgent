import numpy as np

from mapping.diagnostic_snapshot import (
    load_frame_snapshot,
    save_frame_snapshot,
)


def test_mapping_snapshot_round_trip(tmp_path):
    frames = []
    for frame_id, count in ((3, 5), (7, 8)):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = frame_id
        frames.append({
            "frame_id": frame_id,
            "pose": pose,
            "points": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
            "rows": np.arange(count, dtype=np.int32),
        })
    path = tmp_path / "snapshot.npz"
    save_frame_snapshot(path, frames, metadata={"revision": 9})
    loaded, metadata = load_frame_snapshot(path)
    assert metadata == {"revision": 9}
    assert [frame["frame_id"] for frame in loaded] == [3, 7]
    for expected, actual in zip(frames, loaded):
        assert np.array_equal(expected["pose"], actual["pose"])
        assert np.array_equal(expected["points"], actual["points"])
        assert np.array_equal(expected["rows"], actual["rows"])


def test_mapping_snapshot_rejects_empty_input(tmp_path):
    try:
        save_frame_snapshot(tmp_path / "empty.npz", [])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty snapshots must be rejected")
