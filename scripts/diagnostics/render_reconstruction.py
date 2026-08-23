"""从运行中的 mapping server 导出并渲染彩色 VGGT-SLAM 点云。

脚本只读取服务状态，不修改地图。输出 PLY、四个固定视角 PNG 和一张
2x2 总览图，适合在无显示器的远程服务器上检查重影、分层和漂移。
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapping.client import MappingClient
from agents import navigator as nav
from runtime_paths import run_debug_path


VIEWS = {
    "top": (90, -90),
    "oblique_front": (28, 45),
    "oblique_back": (24, 135),
    "side": (5, 0),
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render a mapping-server point cloud from fixed views")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--output-dir",
        default=run_debug_path("diagnostics", "reconstruction"))
    parser.add_argument("--label", default="reconstruction")
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--plot-points", type=int, default=180_000)
    return parser.parse_args()


def _aligned_coordinates(xyz, alignment):
    """Raw VGGT world -> gravity-aligned coordinates with z pointing up."""
    xyz = np.asarray(xyz, dtype=np.float32)
    return xyz @ np.asarray(alignment, dtype=np.float32).T


def _robust_bounds(points):
    low, high = np.percentile(points, [1.0, 99.0], axis=0)
    span = np.maximum(high - low, 1e-3)
    return low - 0.05 * span, high + 0.05 * span


def _style_axis(axis, low, high, title, elev, azim):
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_box_aspect(np.maximum(high - low, 1e-3))
    axis.view_init(elev=elev, azim=azim)
    axis.set_xlabel("aligned X")
    axis.set_ylabel("aligned Y")
    axis.set_zlabel("aligned Z (up)")
    axis.set_title(title)
    axis.grid(True, alpha=0.18)


def _draw(axis, points, colors, trajectory, bounds, title, view):
    low, high = bounds
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=0.22, alpha=0.72, linewidths=0,
        rasterized=True,
    )
    if len(trajectory):
        axis.plot(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color="#ff2d55", linewidth=1.8, label="camera trajectory")
        axis.scatter(
            trajectory[:1, 0], trajectory[:1, 1], trajectory[:1, 2],
            color="#00ff99", s=20, label="start")
        axis.scatter(
            trajectory[-1:, 0], trajectory[-1:, 1], trajectory[-1:, 2],
            color="#ffd60a", s=20, label="end")
    _style_axis(axis, low, high, title, *view)


def _write_ply(path, points, colors):
    try:
        import open3d as o3d
    except ImportError:
        return False
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return bool(o3d.io.write_point_cloud(path, cloud, write_ascii=False))


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    client = MappingClient(args.host, args.port, timeout=180.0)
    try:
        state = client.get_state()
        points, colors_u8 = client.get_map_points(max_points=args.max_points)
        poses, _frame_ids = client.get_all_poses()
    finally:
        client.close()

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors_u8[finite].astype(np.float32) / 255.0
    if not len(points):
        raise RuntimeError("mapping server returned an empty point cloud")

    trajectory_world = (
        np.asarray(poses, dtype=np.float32)[:, :3, 3]
        if poses is not None else np.zeros((0, 3), dtype=np.float32)
    )
    if len(trajectory_world) < 5:
        raise RuntimeError("至少需要 5 个相机位姿才能可靠估计重力方向")
    alignment = nav.gravity_alignment(
        np.asarray(poses, dtype=np.float64),
        cam_up=nav.mount_compensated_cam_up(),
    )
    display_points = _aligned_coordinates(points, alignment)
    display_trajectory = _aligned_coordinates(trajectory_world, alignment)
    bounds = _robust_bounds(display_points)

    if len(display_points) > args.plot_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(
            len(display_points), size=args.plot_points, replace=False)
        plot_points = display_points[indices]
        plot_colors = colors[indices]
    else:
        plot_points, plot_colors = display_points, colors

    for name, view in VIEWS.items():
        figure = plt.figure(figsize=(10, 8), dpi=160)
        axis = figure.add_subplot(111, projection="3d")
        _draw(axis, plot_points, plot_colors, display_trajectory,
              bounds, f"{args.label} — {name}", view)
        figure.tight_layout()
        figure.savefig(
            os.path.join(args.output_dir, f"{args.label}_{name}.png"),
            bbox_inches="tight")
        plt.close(figure)

    # 真正的二维鸟瞰图：沿重力方向压平，不使用 3D 透视投影。
    figure, axis = plt.subplots(figsize=(10, 10), dpi=180)
    axis.scatter(
        plot_points[:, 0], plot_points[:, 1],
        c=plot_colors, s=0.28, alpha=0.72, linewidths=0,
        rasterized=True,
    )
    axis.plot(
        display_trajectory[:, 0], display_trajectory[:, 1],
        color="#ff2d55", linewidth=2.0)
    axis.scatter(
        display_trajectory[:1, 0], display_trajectory[:1, 1],
        color="#00b86b", s=30, label="start")
    axis.scatter(
        display_trajectory[-1:, 0], display_trajectory[-1:, 1],
        color="#e0a800", s=30, label="end")
    axis.set_xlim(bounds[0][0], bounds[1][0])
    axis.set_ylim(bounds[0][1], bounds[1][1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("aligned X")
    axis.set_ylabel("aligned Y")
    axis.set_title(f"{args.label} — gravity-aligned bird's-eye projection")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(
        os.path.join(args.output_dir, f"{args.label}_birdseye.png"),
        bbox_inches="tight")
    plt.close(figure)

    figure = plt.figure(figsize=(16, 13), dpi=150)
    for index, (name, view) in enumerate(VIEWS.items(), 1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        _draw(axis, plot_points, plot_colors, display_trajectory,
              bounds, name, view)
    figure.suptitle(
        f"{args.label}: {len(points):,} sampled map points, "
        f"{len(trajectory_world)} pose entries, "
        f"{state.get('num_submaps', 0)} submaps")
    figure.tight_layout()
    figure.savefig(
        os.path.join(args.output_dir, f"{args.label}_overview.png"),
        bbox_inches="tight")
    plt.close(figure)

    ply_path = os.path.join(args.output_dir, f"{args.label}.ply")
    # PLY 也保存旋正后的坐标，方便 CloudCompare/Open3D 默认视角查看。
    ply_written = _write_ply(ply_path, display_points, colors)
    print({
        "label": args.label,
        "points": len(points),
        "poses": len(trajectory_world),
        "submaps": state.get("num_submaps", 0),
        "ply": ply_path if ply_written else None,
        "output_dir": args.output_dir,
    })


if __name__ == "__main__":
    main()
