"""从运行中的 mapping server 拉取全局点云，导出 PLY 并渲染三视图 PNG。

用法（在 habitat 环境，需 numpy + pillow）::

    PYTHONPATH=/path/to/vggt_nav_agent python scripts/diagnostics/dump_pointcloud.py \
        --port 5555 --max-points 500000

输出:
    <out>.ply          点云（可用 MeshLab/CloudCompare 打开）
    <out>_xy.png       俯视 (x-z 平面取两轴, 视坐标系而定)
    <out>_xz.png / <out>_yz.png / <out>_xy.png 三个正交投影
"""

import argparse
import os

import numpy as np

from agents import navigator as nav
from mapping.client import MappingClient
from runtime_paths import run_debug_path


def save_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def render_projection(points, colors, axes, out_png, size=1024):
    """正交投影散点图。axes: 取 points 的哪两维作为 (横, 纵)。"""
    from PIL import Image

    xy = points[:, list(axes)]
    # 用 1%~99% 分位数裁剪离群点，避免个别野点压缩视野
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (size - 20) / span.max()
    pix = ((xy - lo) * scale).astype(np.int32)
    pix[:, 1] = size - 1 - pix[:, 1]  # 翻转纵轴

    img = np.full((size, size, 3), 255, dtype=np.uint8)
    valid = (pix[:, 0] >= 0) & (pix[:, 0] < size) & \
            (pix[:, 1] >= 0) & (pix[:, 1] < size)
    pix, cols = pix[valid], colors[valid]
    img[pix[:, 1], pix[:, 0]] = cols
    Image.fromarray(img).save(out_png)


def render_density(points, colors, out_png, size=1024):
    """俯视密度图：2D 直方图，颜色=格内平均色，亮度=log 密度。"""
    from PIL import Image

    xy = points[:, [0, 1]]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (size - 20) / span.max()
    pix = ((xy - lo) * scale).astype(np.int32)
    valid = (pix[:, 0] >= 0) & (pix[:, 0] < size) & \
            (pix[:, 1] >= 0) & (pix[:, 1] < size)
    pix, cols = pix[valid], colors[valid]

    count = np.zeros((size, size), dtype=np.float32)
    csum = np.zeros((size, size, 3), dtype=np.float32)
    np.add.at(count, (pix[:, 1], pix[:, 0]), 1)
    for ch in range(3):
        np.add.at(csum[:, :, ch], (pix[:, 1], pix[:, 0]), cols[:, ch])
    occ = count > 0
    mean_col = np.zeros_like(csum)
    mean_col[occ] = csum[occ] / count[occ, None]
    bright = np.zeros_like(count)
    bright[occ] = np.log1p(count[occ]) / np.log1p(count.max())
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[occ] = (mean_col[occ] * (0.4 + 0.6 * bright[occ, None])).astype(np.uint8)
    img = img[::-1]  # 翻转纵轴
    Image.fromarray(img).save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument(
        "--out", default=run_debug_path("diagnostics", "pointcloud"))
    ap.add_argument("--max-points", type=int, default=500000)
    ap.add_argument("--radius", type=float, default=15.0,
                    help="距离过滤半径(米, 相对任一相机中心)")
    args = ap.parse_args()

    client = MappingClient(host=args.host, port=args.port)
    client.wait_idle(timeout=120.0)
    points, colors = client.get_map_points(max_points=args.max_points)
    poses, _ = client.get_all_poses()
    client.close()
    print(f"获取点云 {len(points)} 点")
    if len(points) == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_ply(args.out + ".ply", points, colors)

    if poses is not None and len(poses) >= 5:
        # 复用导航链路的安装角补偿和 PCA 修正，诊断图与规划坐标一致。
        R = nav.gravity_alignment(
            poses, cam_up=nav.mount_compensated_cam_up())
        points_a = points @ R.T
        render_projection(points_a, colors, (0, 1), args.out + "_top.png")
        render_projection(points_a, colors, (0, 2), args.out + "_front.png")
        render_projection(points_a, colors, (1, 2), args.out + "_side.png")
        print("已导出重力对齐视图: _top/_front/_side.png")

        # 距离过滤 + 密度俯视图：抑制野点，突出房间结构
        centers = poses[:, :3, 3] @ R.T
        keep = np.zeros(len(points_a), dtype=bool)
        for c in centers[:: max(1, len(centers) // 50)]:
            keep |= np.linalg.norm(points_a - c, axis=1) < args.radius
        pf, cf = points_a[keep], colors[keep]
        print(f"距离过滤({args.radius}m): {len(pf)}/{len(points_a)} 点保留")
        render_density(pf, cf, args.out + "_top_density.png")
    else:
        print("位姿不足，跳过重力对齐")
    render_projection(points, colors, (0, 1), args.out + "_xy.png")
    render_projection(points, colors, (0, 2), args.out + "_xz.png")
    render_projection(points, colors, (1, 2), args.out + "_yz.png")
    print("已导出:", args.out + ".ply 及三视图 PNG")


if __name__ == "__main__":
    main()
