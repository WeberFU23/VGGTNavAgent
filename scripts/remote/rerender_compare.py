import io
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
import open3d as o3d
from PIL import Image

from agents.map_render import render_pointcloud_topdown, _COLOR_POINTCLOUD_BACKGROUND

PLY = "/root/autodl-tmp/runs_TEEsav_dense_topdown_20260824/pointcloud/TEEsav_dense_topdown_filtered.ply"

pcd = o3d.io.read_point_cloud(PLY)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
z = points[:, 2]
print("z percentiles:", np.percentile(z, [1, 5, 25, 50, 75, 95, 99]).round(3))
for lo, hi in [(-0.2, -0.05), (-0.05, 0.05), (0.05, 0.3), (0.3, 1.0),
               (1.0, 1.8), (1.8, 2.2), (2.2, 2.7)]:
    print(f"  z in [{lo:+.2f},{hi:+.2f}): {((z >= lo) & (z < hi)).sum()}")


def stats(png):
    img = np.asarray(Image.open(io.BytesIO(png)))
    non_bg = (img != _COLOR_POINTCLOUD_BACKGROUND).any(axis=-1)
    return int(non_bg.sum()), img.shape


def old_raster(image, points, colors, px, py, floor_z=None, units_per_m=0.0):
    """旧版：2.7m 上限由调用处保证；权重随高度单调增至 3x。"""
    if len(points) == 0:
        return
    height, width = image.shape[:2]
    flat = py * width + px
    if floor_z is not None and units_per_m > 0:
        height_m = (points[:, 2] - float(floor_z)) / float(units_per_m)
        weights = 1.0 + 2.0 * np.clip(height_m, 0.0, 2.2) / 2.2
    else:
        weights = np.ones(len(points), dtype=np.float64)
    size = height * width
    weight_sum = np.bincount(flat, weights=weights, minlength=size)
    occupied_flat = weight_sum > 0
    raster = np.zeros((size, 3), dtype=np.float64)
    for channel in range(3):
        raster[:, channel] = np.bincount(
            flat, weights=colors[:, channel] * weights, minlength=size)
    raster[occupied_flat] /= weight_sum[occupied_flat, None]
    raster = np.clip(np.rint(raster), 0, 255).astype(np.uint8).reshape(
        height, width, 3)
    occupied = occupied_flat.reshape(height, width)
    image[occupied] = raster[occupied]
    source_mask = occupied.copy()
    source_rgb = raster.copy()
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                   (1, 0), (-1, 1), (0, 1), (1, 1)):
        sy0, sy1 = max(0, -dy), min(height, height - dy)
        sx0, sx1 = max(0, -dx), min(width, width - dx)
        dy0, dy1 = sy0 + dy, sy1 + dy
        dx0, dx1 = sx0 + dx, sx1 + dx
        source = source_mask[sy0:sy1, sx0:sx1]
        destination = occupied[dy0:dy1, dx0:dx1]
        fill = source & ~destination
        if fill.any():
            image[dy0:dy1, dx0:dx1][fill] = source_rgb[sy0:sy1, sx0:sx1][fill]
            destination[fill] = True


# 旧版：2.7m 上限 + 旧权重（exec 一个改了上限的模块副本，替换其 rasterizer）
src = open("/root/autodl-tmp/vggt_nav_agent/agents/map_render.py").read()
src_old = src.replace("(z <= float(floor_z) + 2.20 * units_per_m)",
                      "(z <= float(floor_z) + 2.70 * units_per_m)")
ns = {}
exec(compile(src_old, "map_render_old", "exec"), ns)
ns["_rasterize_orthographic_rgb"] = old_raster
png_old = ns["render_pointcloud_topdown"](
    points, colors, floor_z=0.0, unit_per_m=1.0,
    min_image_side=768, max_image_side=1024)

png_new = render_pointcloud_topdown(
    points, colors, floor_z=0.0, unit_per_m=1.0,
    min_image_side=768, max_image_side=1024)

n_old, shape = stats(png_old)
n_new, _ = stats(png_new)
print(f"OLD non-bg: {n_old} / {shape[0]*shape[1]}")
print(f"NEW non-bg: {n_new} / {shape[0]*shape[1]}")
open("/root/autodl-tmp/rerender_old_weights.png", "wb").write(png_old)
open("/root/autodl-tmp/rerender_new_weights.png", "wb").write(png_new)
print("saved both renders")
