"""Rebuild occupancy and top-down diagnostics from a fixed VGGT snapshot."""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agents import navigator as nav
from agents.map_render import render_topdown
from mapping.diagnostic_snapshot import load_frame_snapshot
from runtime_paths import run_debug_path


def _report(grid, frame_count):
    traversed = np.asarray(grid.traversed, dtype=bool)
    return {
        "frames": int(frame_count),
        "shape": list(grid.free.shape),
        "resolution_map_units": float(grid.res),
        "unit_per_m": float(getattr(grid, "unit_per_m", 0.0)),
        "floor_z": float(getattr(grid, "floor_z", np.nan)),
        "floor_model": getattr(grid, "floor_model", None),
        "source_frame_count": int(getattr(grid, "source_frame_count",
                                          frame_count)),
        "source_point_count": int(getattr(grid, "source_point_count", 0)),
        "retained_voxel_count": int(getattr(
            grid, "retained_voxel_count", 0)),
        "voxel_size_m": float(getattr(grid, "voxel_size_m", 0.0)),
        "min_voxel_views": int(getattr(grid, "min_voxel_views", 0)),
        "free_cells": int(grid.free.sum()),
        "obstacle_cells": int(grid.obstacle.sum()),
        "geometry_observed_cells": int(grid.geometry_observed.sum()),
        "traversed_cells": int(traversed.sum()),
        "traversed_unknown_cells": int((
            traversed & ~grid.free & ~grid.obstacle).sum()),
        "traversed_obstacle_conflicts": int((
            traversed & grid.obstacle).sum()),
        "traversed_raw_obstacle_conflicts": int((
            traversed & grid.raw_obstacle).sum()),
        "ground_vote_cells": int((grid.ground_votes > 0).sum()),
        "ground_vote_max": int(grid.ground_votes.max(initial=0)),
        "obstacle_vote_cells": int((grid.obstacle_votes > 0).sum()),
        "obstacle_vote_max": int(grid.obstacle_votes.max(initial=0)),
        "start_cell": grid.start_cell,
        "start_seed_cell": grid.start_seed_cell,
        "start_seed_distance_cells": grid.start_seed_distance_cells,
        "connectivity_filtered": bool(grid.connectivity_filtered),
    }


def _font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _vote_panel(values, channel):
    values = np.asarray(values, dtype=np.float64)
    scaled = np.log1p(values)
    peak = float(scaled.max(initial=0.0))
    if peak > 0:
        scaled /= peak
    image = np.zeros(values.shape + (3,), dtype=np.uint8)
    image[..., channel] = np.round(255 * scaled).astype(np.uint8)
    return image


def _render_evidence_layers(grid, out_path, panel_side=640):
    """Render evidence arrays separately so classification cannot hide causes."""
    geometry = np.repeat(
        (np.asarray(grid.geometry_observed, dtype=np.uint8) * 210)[..., None],
        3, axis=2)
    traversed = np.zeros(grid.free.shape + (3,), dtype=np.uint8)
    path = np.asarray(grid.traversed, dtype=bool)
    traversed[path] = (145, 205, 225)
    traversed[path & grid.obstacle] = (235, 65, 165)
    panels = [
        ("ground votes (log scale)", _vote_panel(grid.ground_votes, 1)),
        ("obstacle votes (log scale)", _vote_panel(
            grid.obstacle_votes, 0)),
        ("geometry observed", geometry),
        ("traversed; magenta=obstacle conflict", traversed),
    ]
    font = _font()
    label_h = 28
    rendered = []
    for title, array in panels:
        source = Image.fromarray(array)
        scale = min(panel_side / max(source.width, 1),
                    panel_side / max(source.height, 1))
        size = (max(1, int(round(source.width * scale))),
                max(1, int(round(source.height * scale))))
        image = source.resize(size, Image.Resampling.NEAREST)
        panel = Image.new("RGB", (panel_side, panel_side + label_h), "white")
        panel.paste(image, ((panel_side - image.width) // 2,
                            label_h + (panel_side - image.height) // 2))
        ImageDraw.Draw(panel).text((8, 6), title, fill=(20, 20, 20), font=font)
        rendered.append(panel)
    canvas = Image.new(
        "RGB", (panel_side * 2, (panel_side + label_h) * 2), "white")
    for index, panel in enumerate(rendered):
        canvas.paste(panel, ((index % 2) * panel_side,
                             (index // 2) * (panel_side + label_h)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument(
        "--out", default=run_debug_path(
            "diagnostics", "occupancy_snapshot.png"))
    parser.add_argument("--report")
    parser.add_argument("--layers-out")
    parser.add_argument("--identity-alignment", action="store_true")
    parser.add_argument(
        "--scale-m-per-unit", type=float,
        help="已知 SLAM 尺度；用于约束地板位于相机下方约 1.5m")
    parser.add_argument("--floor-band-m", type=float, default=0.12)
    parser.add_argument("--obs-low-m", type=float, default=0.15)
    parser.add_argument("--obs-high-m", type=float, default=1.8)
    parser.add_argument("--robot-radius-m", type=float, default=0.25)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--min-voxel-views", type=int, default=3)
    args = parser.parse_args()

    frames, _metadata = load_frame_snapshot(args.snapshot)
    poses = np.stack([frame["pose"] for frame in frames])
    align_R = np.eye(3) if args.identity_alignment else nav.gravity_alignment(
        poses, cam_up=nav.mount_compensated_cam_up())
    unit_per_m = None
    if args.scale_m_per_unit is not None:
        if not np.isfinite(args.scale_m_per_unit) or \
                args.scale_m_per_unit <= 0:
            raise SystemExit("--scale-m-per-unit must be positive")
        unit_per_m = 1.0 / args.scale_m_per_unit
    grid = nav.OccupancyGrid.from_frame_points(
        frames, align_R, unit_per_m=unit_per_m,
        floor_band_m=args.floor_band_m, obs_low_m=args.obs_low_m,
        obs_high_m=args.obs_high_m, robot_radius_m=args.robot_radius_m,
        voxel_size_m=args.voxel_size_m,
        min_voxel_views=args.min_voxel_views)
    if grid is None:
        raise SystemExit("occupancy construction failed")
    trajectory = poses[:, :3, 3] @ align_R.T
    latest = poses[-1]
    px, py, yaw = nav.pose_to_yaw_2d(latest, align_R)
    png = render_topdown(
        grid, trajectory=[tuple(point[:2]) for point in trajectory],
        pose=(px, py, yaw), step="snapshot", map_revision="offline")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fp:
        fp.write(png)
    report_path = args.report or os.path.splitext(args.out)[0] + ".json"
    layers_path = args.layers_out or \
        os.path.splitext(args.out)[0] + "_layers.png"
    _render_evidence_layers(grid, layers_path)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(_report(grid, len(frames)), fp, indent=2,
                  ensure_ascii=False)
    print(f"saved {args.out}")
    print(f"saved {layers_path}")
    print(f"saved {report_path}")


if __name__ == "__main__":
    main()
