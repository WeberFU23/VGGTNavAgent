import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
import open3d as o3d
from agents.map_render import render_pointcloud_topdown

PLY = "/root/autodl-tmp/runs_TEEsav_dense_topdown_20260824/pointcloud/TEEsav_dense_topdown_filtered.ply"
OUT = "/root/autodl-tmp/rerender_new_weights.png"

pcd = o3d.io.read_point_cloud(PLY)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)  # 0..1 float
print("points:", points.shape, "z range:", np.percentile(points[:, 2], [0, 1, 50, 99, 100]))

png = render_pointcloud_topdown(
    points, colors, floor_z=0.0, unit_per_m=1.0,
    min_image_side=768, max_image_side=1024)
open(OUT, "wb").write(png)
print("saved", OUT)
