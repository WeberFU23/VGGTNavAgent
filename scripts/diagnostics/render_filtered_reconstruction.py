"""Render a cleaned, metrically scaled VGGT-SLAM point cloud.

The raw map is useful for preserving every confident VGGT prediction, but a
direct transparent scatter plot makes surfaces hard to read: repeated frames
overdraw the same wall while isolated points fill the empty volume.  This tool
keeps the raw map untouched and creates a separate diagnostic product by:

1. rotating the map into the gravity-aligned navigation frame;
2. converting map units to metres;
3. cropping to the explored floor and trajectory neighbourhood;
4. averaging points and RGB values inside metric voxels;
5. rejecting voxels supported by too few source points.

It writes a binary RGB PLY, clear fixed-view renders, height-sliced bird's-eye
views and a JSON report.  Filtering is diagnostic only and never changes the
mapping server or the occupancy grid used by the agent.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agents import navigator as nav
from mapping.client import MappingClient
from runtime_paths import run_debug_path


VIEWS = {
    "oblique_front": (28, 45),
    "oblique_back": (24, 135),
    "side": (5, 0),
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Filter and render a mapping-server RGB point cloud")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--output-dir",
        default=run_debug_path("diagnostics", "filtered_reconstruction"))
    parser.add_argument("--label", default="filtered_reconstruction")
    parser.add_argument("--max-points", type=int, default=1_200_000)
    parser.add_argument("--plot-points", type=int, default=250_000)
    parser.add_argument("--meters-per-unit", type=float, required=True)
    parser.add_argument("--camera-height-m", type=float, default=1.5)
    parser.add_argument("--crop-margin-m", type=float, default=4.0)
    parser.add_argument("--min-height-m", type=float, default=-0.20)
    parser.add_argument("--max-height-m", type=float, default=2.70)
    parser.add_argument("--voxel-size-m", type=float, default=0.035)
    parser.add_argument("--min-voxel-points", type=int, default=2)
    return parser.parse_args()


def _voxel_average(points, colors, voxel_size, min_points):
    """Average geometry/RGB per voxel and drop weakly supported voxels."""
    origin = points.min(axis=0)
    cells = np.floor((points - origin) / voxel_size).astype(np.int32)
    unique, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True)
    supported = counts >= max(1, int(min_points))
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    color_sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    denom = counts[:, None]
    return (sums[supported] / denom[supported]).astype(np.float32), \
        np.clip(color_sums[supported] / denom[supported], 0, 255) \
        .astype(np.uint8), counts, supported


def _write_binary_ply(path, points, colors):
    """Write a compact PLY without requiring Open3D."""
    vertex = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n")
    with open(path, "wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertex.tobytes())


def _sample(points, colors, limit):
    if len(points) <= limit:
        return points, colors
    indices = np.random.default_rng(0).choice(
        len(points), size=limit, replace=False)
    return points[indices], colors[indices]


def _bounds(points):
    low, high = np.percentile(points, [0.25, 99.75], axis=0)
    span = np.maximum(high - low, 0.1)
    return low - 0.03 * span, high + 0.03 * span


def _style_3d(axis, bounds, title, view):
    low, high = bounds
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_box_aspect(np.maximum(high - low, 0.1))
    axis.view_init(elev=view[0], azim=view[1])
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("height above floor (m)")
    axis.set_title(title)
    axis.grid(True, alpha=0.16)


def _render_3d(path, title, points, colors, trajectory, bounds, view):
    figure = plt.figure(figsize=(12, 9), dpi=170)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=colors,
        s=1.25, alpha=0.96, linewidths=0, rasterized=True)
    axis.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color="#ff1744", linewidth=2.0)
    _style_3d(axis, bounds, title, view)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="#f2f2f2")
    plt.close(figure)


def _render_birdseye(path, title, points, colors, trajectory, bounds):
    figure, axis = plt.subplots(figsize=(11, 10), dpi=180)
    axis.scatter(
        points[:, 0], points[:, 1], c=colors,
        s=1.15, alpha=0.97, linewidths=0, rasterized=True)
    axis.plot(trajectory[:, 0], trajectory[:, 1],
              color="#ff1744", linewidth=2.2)
    axis.scatter(trajectory[:1, 0], trajectory[:1, 1],
                 color="#00a86b", s=36, label="start")
    axis.scatter(trajectory[-1:, 0], trajectory[-1:, 1],
                 color="#e3a000", s=36, label="end")
    axis.set_xlim(bounds[0][0], bounds[1][0])
    axis.set_ylim(bounds[0][1], bounds[1][1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title(title)
    axis.grid(True, alpha=0.15)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="#f2f2f2")
    plt.close(figure)


def _render_height_slices(path, points, colors, bounds):
    slices = (
        ("floor: -0.15 to 0.15 m", -0.15, 0.15),
        ("low objects: 0.15 to 0.85 m", 0.15, 0.85),
        ("walls/objects: 0.85 to 2.20 m", 0.85, 2.20),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=170)
    for axis, (title, low_z, high_z) in zip(axes, slices):
        mask = (points[:, 2] >= low_z) & (points[:, 2] < high_z)
        axis.scatter(points[mask, 0], points[mask, 1], c=colors[mask],
                     s=1.1, alpha=0.98, linewidths=0, rasterized=True)
        axis.set_xlim(bounds[0][0], bounds[1][0])
        axis.set_ylim(bounds[0][1], bounds[1][1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title}\n{int(mask.sum()):,} voxels")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.grid(True, alpha=0.13)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="#f2f2f2")
    plt.close(figure)


def main():
    args = _parse_args()
    if args.meters_per_unit <= 0 or args.voxel_size_m <= 0:
        raise ValueError("scale and voxel size must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    client = MappingClient(args.host, args.port, timeout=240.0)
    try:
        state = client.get_state()
        raw_points, raw_colors = client.get_map_points(args.max_points)
        poses, frame_ids = client.get_all_poses()
    finally:
        client.close()
    finite = np.isfinite(raw_points).all(axis=1)
    raw_points, raw_colors = raw_points[finite], raw_colors[finite]
    if not len(raw_points) or poses is None or len(poses) < 5:
        raise RuntimeError("mapping server returned insufficient map data")

    poses = np.asarray(poses, dtype=np.float64)
    alignment = nav.gravity_alignment(
        poses, cam_up=nav.mount_compensated_cam_up())
    points = (raw_points @ alignment.T) * args.meters_per_unit
    trajectory = (poses[:, :3, 3] @ alignment.T) * args.meters_per_unit
    floor_z = float(np.median(trajectory[:, 2]) - args.camera_height_m)
    points[:, 2] -= floor_z
    trajectory[:, 2] -= floor_z

    margin = args.crop_margin_m
    low_xy = trajectory[:, :2].min(axis=0) - margin
    high_xy = trajectory[:, :2].max(axis=0) + margin
    keep = np.isfinite(points).all(axis=1)
    keep &= np.all(points[:, :2] >= low_xy, axis=1)
    keep &= np.all(points[:, :2] <= high_xy, axis=1)
    keep &= points[:, 2] >= args.min_height_m
    keep &= points[:, 2] <= args.max_height_m
    cropped_points, cropped_colors = points[keep], raw_colors[keep]
    filtered_points, filtered_colors, voxel_counts, supported = \
        _voxel_average(
            cropped_points, cropped_colors, args.voxel_size_m,
            args.min_voxel_points)
    if not len(filtered_points):
        raise RuntimeError("all points were rejected by diagnostic filtering")

    plot_points, plot_colors_u8 = _sample(
        filtered_points, filtered_colors, args.plot_points)
    plot_colors = plot_colors_u8.astype(np.float32) / 255.0
    bounds = _bounds(filtered_points)
    prefix = os.path.join(args.output_dir, args.label)
    _write_binary_ply(prefix + "_filtered.ply",
                      filtered_points, filtered_colors)
    for name, view in VIEWS.items():
        _render_3d(
            prefix + f"_{name}.png", f"{args.label} — {name}",
            plot_points, plot_colors, trajectory, bounds, view)
    _render_birdseye(
        prefix + "_birdseye.png",
        f"{args.label} — filtered RGB point cloud",
        plot_points, plot_colors, trajectory, bounds)
    _render_height_slices(
        prefix + "_height_slices.png", plot_points, plot_colors, bounds)

    report = {
        "raw_points": int(len(raw_points)),
        "cropped_points": int(len(cropped_points)),
        "candidate_voxels": int(len(voxel_counts)),
        "retained_voxels": int(supported.sum()),
        "rejected_sparse_voxels": int((~supported).sum()),
        "poses": int(len(poses)),
        "frame_ids": int(len(frame_ids or [])),
        "submaps": int(state.get("num_submaps", 0)),
        "meters_per_unit": args.meters_per_unit,
        "floor_z_before_recentering_m": floor_z,
        "voxel_size_m": args.voxel_size_m,
        "min_voxel_points": args.min_voxel_points,
        "height_range_m": [args.min_height_m, args.max_height_m],
        "crop_margin_m": args.crop_margin_m,
    }
    with open(prefix + "_report.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
