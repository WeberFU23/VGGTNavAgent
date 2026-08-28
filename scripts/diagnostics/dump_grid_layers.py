"""对 5556 端口上已建好的地图直接分析栅格层并导出可视化（不重放）。

用法：python scripts/diagnostics/dump_grid_layers.py --port 5556 --out layers.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from mapping.client import MappingClient
from agents import navigator as nav
from agents import skeleton as skel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--out", default="/root/autodl-tmp/grid_layers.png")
    args = ap.parse_args()

    client = MappingClient(host="127.0.0.1", port=args.port)
    frames = client.get_frame_points(stride=3)
    pose_by_frame = {int(f["frame_id"]): np.asarray(f["pose"], float)
                     for f in frames}
    poses = np.stack([pose_by_frame[f] for f in sorted(pose_by_frame)])
    align_R = nav.gravity_alignment(
        poses, cam_up=nav.mount_compensated_cam_up())
    grid = nav.OccupancyGrid.from_frame_points(frames, align_R)

    free = np.asarray(grid.free, bool)
    obstacle = np.asarray(grid.obstacle, bool)
    observed = np.asarray(grid.observed, bool)
    geom = np.asarray(getattr(grid, "geometry_observed", observed), bool)
    print(f"unit_per_m={grid.unit_per_m:.3f} res={grid.res:.4f}")
    print(f"cells: free={free.sum()} obstacle={obstacle.sum()} "
          f"observed={observed.sum()} geometry_observed={geom.sum()} "
          f"total={free.size}")
    print(f"floor_z={grid.floor_z:.3f} start_cell={grid.start_cell} "
          f"seed={grid.start_seed_cell} "
          f"connectivity_filtered={grid.connectivity_filtered}")
    print(f"source_points={grid.source_point_count} "
          f"retained_voxels={grid.retained_voxel_count}")

    raw, layers = skel.frontier_clusters(grid, min_size=5,
                                         return_layers=True)
    boundary = np.asarray(layers.get("unified", []), bool)
    print(f"boundary={boundary.sum()} clusters={len(raw)}")

    # 可视化：白=未观测, 灰=已观测非自由, 绿=free, 红=obstacle, 蓝=frontier
    img = np.full((*free.shape, 3), 255, dtype=np.uint8)
    img[geom] = (200, 200, 200)
    img[free] = (60, 200, 60)
    img[obstacle] = (220, 50, 50)
    img[boundary] = (50, 50, 240)
    # 相机轨迹
    traj = np.asarray(grid.traversed, bool) if hasattr(
        grid, "traversed") else None
    if traj is not None:
        img[traj & free] = (20, 60, 20)
    from PIL import Image
    im = Image.fromarray(img)
    im = im.resize((img.shape[1] * 4, img.shape[0] * 4),
                   Image.NEAREST)
    im.save(args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
