"""离线诊断：重放 episode RGB 后检查 frontier 供给在哪一环断掉。

用法（远端 habitat 环境，先起不带语义的映射服务）：
    python -m mapping.server --port 5556 --no-semantic ...
    python scripts/diagnostics/diagnose_frontier_supply.py \
        <rgb_dir> --port 5556 --reset-map

输出：栅格统计、原始 frontier 簇数量/尺寸/分布，以及沿轨迹多个位置
施加与 nav_agent 相同的过滤（过近/不可达）后的剩余数量。
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from mapping.client import MappingClient
from agents import navigator as nav
from agents import skeleton as skel
from scripts.diagnostics.replay_rgb_sequence import list_rgb_frames
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rgb_dir")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--reset-map", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    client = MappingClient(host="127.0.0.1", port=args.port)
    paths = list_rgb_frames(args.rgb_dir)
    if args.max_frames > 0:
        paths = paths[: args.max_frames]
    print(f"replay {len(paths)} frames -> :{args.port}")
    if args.reset_map:
        client.reset_map()
    client.set_episode("frontier-diag")
    t0 = time.time()
    for i, p in enumerate(paths):
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        client.feed_frame(rgb)
        if i % 100 == 0:
            print(f"  fed {i}/{len(paths)} ({time.time()-t0:.0f}s)",
                  flush=True)
    client.flush_map()
    # 等后端清空
    for _ in range(120):
        st = client.get_state()
        if not st.get("busy") and not st.get("queued_keyframes"):
            break
        time.sleep(5)
    print(f"replay done in {time.time()-t0:.0f}s, state={st}")

    frames = client.get_frame_points(stride=3)
    print("frames:", len(frames))
    pose_by_frame = {int(f["frame_id"]): np.asarray(f["pose"], float)
                     for f in frames}
    fids = sorted(pose_by_frame)
    poses = np.stack([pose_by_frame[f] for f in fids])
    align_R = nav.gravity_alignment(
        poses, cam_up=nav.mount_compensated_cam_up())
    grid = nav.OccupancyGrid.from_frame_points(frames, align_R)
    print("grid unit_per_m:", grid.unit_per_m)
    free = np.asarray(grid.free, bool)
    obs = np.asarray(grid.obstacle, bool)
    geom = np.asarray(getattr(grid, "geometry_observed",
                              getattr(grid, "observed", None)), bool)
    print(f"grid cells: free={free.sum()} obstacle={obs.sum()} "
          f"geometry_observed={geom.sum()} total={free.size}")

    raw, layers = skel.frontier_clusters(grid, min_size=5,
                                         return_layers=True)
    boundary = np.asarray(layers.get("unified", []), bool)
    print(f"frontier boundary cells={boundary.sum()}, "
          f"raw clusters={len(raw)}")
    for c in raw:
        print(f"  cluster world=({c['world'][0]:.2f},{c['world'][1]:.2f}) "
              f"size={c.get('size')} reason={c.get('reason')} "
              f"geo_gain={c.get('geometry_gain')} "
              f"sem_gain={c.get('semantic_gain')}")

    # 沿轨迹多位置施加与 nav_agent 相同的过滤
    scale = 1.0 / grid.unit_per_m if grid.unit_per_m > 0 else 1.0
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        pose = poses[int(frac * (len(poses) - 1))]
        cur = pose[:3, 3] @ align_R.T
        too_near = unreachable = ok = 0
        for c in raw:
            d = math.hypot(c["world"][0] - cur[0],
                           c["world"][1] - cur[1]) * scale
            if d < 1.0:
                too_near += 1
                continue
            path = grid.astar(cur[:2], c["world"])
            if path is None or len(path) < 2:
                unreachable += 1
            else:
                ok += 1
        print(f"pos {frac:.0%}: too_near={too_near} "
              f"unreachable={unreachable} valid={ok}")


if __name__ == "__main__":
    main()
