import glob
import json
import sys

root = sys.argv[1]
for path in sorted(glob.glob(root + "/*/debug_output/*/mapping/diagnostics/*_queries.jsonl")):
    ep = path.split("/")[-1].replace("_queries.jsonl", "")
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        cmd = r.get("cmd")
        if cmd in ("instantiate_pixels", "point_pixels", "point_frame", "prepare_pixels"):
            pts = r.get("points") or []
            for p in pts:
                px = p.get("pixel")
                if px:
                    print(ep, r.get("t"), cmd, "f%s" % r.get("frame_id"),
                          str(r.get("text"))[:30], px, "found=", p.get("found"))
