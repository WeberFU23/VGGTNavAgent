"""从运行中的 mapping server 拉点云+位姿，构建自由空间栅格并渲染 PNG。

用法（habitat 环境，eval 跑完、server 仍持图时）::

    PYTHONPATH=/path/to/vggt_nav_agent python scripts/diagnostics/check_freespace.py \
        --port 5555

配色：白=未知，绿=几何可行走，红=膨胀障碍，浅蓝=走过但几何未确认，
洋红=走过但当前被判为障碍，蓝=轨迹中心线，黄=最新相机位置。
"""

import argparse
import os

import numpy as np

from agents import navigator as nav
from mapping.client import MappingClient
from mapping.diagnostic_snapshot import save_frame_snapshot
from runtime_paths import run_debug_path


def render(grid, cam_centers_aligned, out_png, size=1024):
    from PIL import Image

    h, w = grid.free.shape
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[grid.free] = (120, 220, 120)
    img[grid.obstacle] = (220, 80, 80)
    traversed = np.asarray(grid.traversed, dtype=bool)
    img[traversed & ~grid.free & ~grid.obstacle] = (145, 205, 225)
    img[traversed & grid.obstacle] = (235, 65, 165)

    cc = np.floor((cam_centers_aligned[:, :2] - grid.origin) / grid.res) \
        .astype(np.int64)
    ok = (cc[:, 0] >= 0) & (cc[:, 0] < w) & (cc[:, 1] >= 0) & (cc[:, 1] < h)
    cc = cc[ok]
    img[cc[:, 1], cc[:, 0]] = (60, 60, 230)
    if len(cc):
        x, y = int(cc[-1][0]), int(cc[-1][1])
        img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = (250, 220, 40)

    # 缩放到 size 并保持纵横比
    scale = (size - 10) / max(h, w)
    nh, nw = max(int(h * scale), 1), max(int(w * scale), 1)
    img = np.asarray(
        Image.fromarray(img).resize((nw, nh), Image.NEAREST))
    Image.fromarray(img).save(out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--out", default=run_debug_path("diagnostics", "freespace.png"))
    parser.add_argument("--max-points", type=int, default=800000)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument(
        "--scale-m-per-unit", type=float,
        help="已知 SLAM 尺度；用于约束全局地板峰搜索")
    parser.add_argument(
        "--snapshot-out",
        help="同时保存可离线重复构建 occupancy 的 VGGT frame snapshot")
    args = parser.parse_args()

    client = MappingClient(host=args.host, port=args.port)
    poses, _ = client.get_all_poses()
    if poses is None or len(poses) < 5:
        print("位姿不足")
        return
    poses = np.asarray(poses, dtype=np.float64)
    align_R = nav.gravity_alignment(
        poses, cam_up=nav.mount_compensated_cam_up())
    cam_centers = poses[:, :3, 3] @ align_R.T
    unit_per_m = None
    if args.scale_m_per_unit is not None:
        if not np.isfinite(args.scale_m_per_unit) or \
                args.scale_m_per_unit <= 0:
            raise SystemExit("--scale-m-per-unit must be positive")
        unit_per_m = 1.0 / args.scale_m_per_unit

    grid = None
    frames = client.get_frame_points(stride=args.stride)
    print(f"逐帧点: {len(frames)} 帧, "
          f"首帧 {frames[0]['points'].shape if frames else None}")
    if frames:
        if args.snapshot_out:
            os.makedirs(os.path.dirname(os.path.abspath(
                args.snapshot_out)), exist_ok=True)
            save_frame_snapshot(
                args.snapshot_out, frames,
                metadata={
                    "stride": args.stride,
                    "snapshot_revision":
                        client.last_frame_snapshot_revision,
                })
            print(f"已保存离线 snapshot: {args.snapshot_out}")
        grid = nav.OccupancyGrid.from_frame_points(
            frames, align_R, unit_per_m=unit_per_m)
        if grid is not None:
            print("使用融合点云全局地板栅格")
    if grid is None:
        pts, _ = client.get_map_points(max_points=args.max_points)
        print(f"回退全局点云 {len(pts)} 点, 关键帧 {len(poses)}")
        if len(pts) < 1000:
            print("点云不足")
            return
        pts_aligned = np.asarray(pts, dtype=np.float64) @ align_R.T
        grid = nav.OccupancyGrid.build(
            pts_aligned, cam_centers, unit_per_m=unit_per_m)
    if grid is None:
        print("栅格构建失败")
        return
    n_free = int(grid.free.sum())
    n_obs = int(grid.obstacle.sum())
    print(f"栅格 {grid.free.shape}, res={grid.res:.3f} 单位/格, "
          f"unit_per_m={grid.unit_per_m:.3f}, floor_z={grid.floor_z:.3f}")
    print(f"自由格 {n_free} ({n_free / grid.free.size * 100:.1f}%), "
          f"障碍格 {n_obs}")
    traversed = np.asarray(grid.traversed, dtype=bool)
    print(
        f"轨迹证据 {int(traversed.sum())} 格；"
        f"未确认 free={int((traversed & ~grid.free & ~grid.obstacle).sum())}，"
        f"与 obstacle 冲突={int((traversed & grid.obstacle).sum())}；"
        f"起点 seed={grid.start_seed_cell}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    render(grid, cam_centers, args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
