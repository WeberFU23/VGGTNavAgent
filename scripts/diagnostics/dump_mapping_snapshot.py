"""Export the current mapping server state for deterministic offline replay."""

import argparse
import os
import time

from mapping.client import MappingClient
from mapping.diagnostic_snapshot import save_frame_snapshot
from runtime_paths import run_debug_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument(
        "--out", default=run_debug_path(
            "diagnostics", "mapping_snapshot.npz"))
    args = parser.parse_args()

    client = MappingClient(host=args.host, port=args.port)
    try:
        state = client.get_state()
        frames = client.get_frame_points(stride=args.stride)
        revision = client.last_frame_snapshot_revision
    finally:
        client.close()
    if not frames:
        raise SystemExit("mapping server returned no frame points")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_frame_snapshot(args.out, frames, metadata={
        "created_unix": time.time(),
        "stride": args.stride,
        "server_state": state,
        "snapshot_revision": revision,
    })
    print(f"saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
