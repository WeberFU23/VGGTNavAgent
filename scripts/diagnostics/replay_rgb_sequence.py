"""Replay a fixed RGB directory through a running mapping server.

The server map is reset only when ``--reset-map`` is explicitly provided. This
keeps the tool safe around an active deployment while allowing deterministic
keyframe/SLAM ablations from RGB frames previously saved by MappingAgent.
"""

import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from mapping.client import MappingClient
from runtime_paths import run_debug_path


_RGB_FRAME_NAME = re.compile(r"^rgb_[0-9]+\.(?:jpe?g|png)$", re.IGNORECASE)


def list_rgb_frames(root):
    """Return only raw recorded observations, excluding diagnostic overlays."""
    root = Path(root)
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and _RGB_FRAME_NAME.fullmatch(path.name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--episode-id", default="offline-replay")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reset-map", action="store_true")
    parser.add_argument(
        "--report", default=run_debug_path(
            "diagnostics", "rgb_replay_state.json"))
    args = parser.parse_args()

    root = Path(args.rgb_dir)
    paths = list_rgb_frames(root)
    if args.max_frames > 0:
        paths = paths[:args.max_frames]
    if not paths:
        raise SystemExit(f"no RGB images found in {root}")

    client = MappingClient(host=args.host, port=args.port)
    try:
        if args.reset_map:
            client.reset_map()
        client.set_episode(args.episode_id)
        for index, path in enumerate(paths):
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            info = client.feed_frame(rgb)
            if info.get("busy"):
                client.wait_idle(timeout=60.0)
            if index % 20 == 0:
                print(f"frame {index + 1}/{len(paths)}: {path.name}")
        client.flush_map()
        state = client.get_state()
    finally:
        client.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fp:
        json.dump(state, fp, indent=2, ensure_ascii=False)
    print(f"replayed {len(paths)} RGB frames; state -> {args.report}")


if __name__ == "__main__":
    main()
