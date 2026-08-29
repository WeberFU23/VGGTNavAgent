import sys
import cv2

# args: frames_dir out_dir fid x y [fid x y ...]
frames_dir, out_dir = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
for i in range(0, len(rest), 3):
    fid, x, y = int(rest[i]), float(rest[i + 1]), float(rest[i + 2])
    p = "%s/rgb_%06d.jpg" % (frames_dir, fid)
    img = cv2.imread(p)
    if img is None:
        print("missing", p)
        continue
    s = img.shape[1] / 518.0
    for (px, py), col in [((x, y), (0, 0, 255)), ((y, x), (0, 255, 0))]:
        cx, cy = int(px * s), int(py * s)
        cv2.drawMarker(img, (cx, cy), col, cv2.MARKER_TILTED_CROSS, 60, 6)
    cv2.imwrite("%s/check_%06d.jpg" % (out_dir, fid), img)
    print("saved", fid)
