"""Portable, pickle-free snapshots for occupancy diagnostics.

The mapping server exposes per-frame VGGT points, camera poses and source image
rows.  This module stores that response in a compact NPZ so occupancy and
top-down rendering can be changed and replayed without rerunning Habitat or
VGGT-SLAM.  Simulator depth and poses are intentionally not part of this format;
they remain separate evaluation-only truth.
"""

import json

import numpy as np


SNAPSHOT_VERSION = 1


def save_frame_snapshot(path, frames, metadata=None):
    """Save MappingClient.get_frame_points() output to ``path``."""
    frame_ids = []
    poses = []
    offsets = [0]
    point_parts = []
    row_parts = []
    for frame in frames or []:
        points = np.asarray(frame["points"], dtype=np.float32).reshape(-1, 3)
        rows = np.asarray(frame["rows"], dtype=np.int32).reshape(-1)
        pose = np.asarray(frame["pose"], dtype=np.float32).reshape(4, 4)
        if len(points) != len(rows):
            raise ValueError("frame points and rows must have equal length")
        frame_ids.append(int(frame["frame_id"]))
        poses.append(pose)
        point_parts.append(points)
        row_parts.append(rows)
        offsets.append(offsets[-1] + len(points))
    if not frame_ids:
        raise ValueError("cannot save an empty mapping snapshot")
    payload = {
        "version": np.asarray([SNAPSHOT_VERSION], dtype=np.int32),
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "poses": np.stack(poses).astype(np.float32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "points": np.concatenate(point_parts).astype(np.float32),
        "rows": np.concatenate(row_parts).astype(np.int32),
        "metadata_json": np.asarray(
            [json.dumps(metadata or {}, ensure_ascii=False, default=str)]),
    }
    np.savez_compressed(path, **payload)


def load_frame_snapshot(path):
    """Load and validate a snapshot, returning ``(frames, metadata)``."""
    with np.load(path, allow_pickle=False) as data:
        version = int(np.asarray(data["version"]).reshape(-1)[0])
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"unsupported mapping snapshot version: {version}")
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        poses = np.asarray(data["poses"], dtype=np.float32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        points = np.asarray(data["points"], dtype=np.float32)
        rows = np.asarray(data["rows"], dtype=np.int32)
        metadata_text = str(
            np.asarray(data["metadata_json"]).reshape(-1)[0])
    if poses.shape != (len(frame_ids), 4, 4):
        raise ValueError("snapshot poses do not match frame ids")
    if offsets.shape != (len(frame_ids) + 1,) or offsets[0] != 0:
        raise ValueError("snapshot offsets are malformed")
    if offsets[-1] != len(points) or len(points) != len(rows):
        raise ValueError("snapshot point payload is truncated")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("snapshot offsets must be monotonic")
    frames = []
    for index, frame_id in enumerate(frame_ids):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        frames.append({
            "frame_id": int(frame_id),
            "pose": poses[index].copy(),
            "points": points[lo:hi].copy(),
            "rows": rows[lo:hi].copy(),
        })
    metadata = json.loads(metadata_text)
    if not isinstance(metadata, dict):
        raise ValueError("snapshot metadata must be a JSON object")
    return frames, metadata
