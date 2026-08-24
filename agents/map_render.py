"""Bird's-eye renderers for VLM input and occupancy diagnostics.

``render_pointcloud_topdown`` is the production decision image: reconstructed
RGB points plus agent/frontier/target markers only. ``render_topdown`` retains
the richer occupancy-layer visualization for offline debugging and tests; the
decision path must not send that diagnostic coloring or trajectory to the VLM.
"""

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_COLOR_GEOMETRY_UNKNOWN = (185, 185, 185)
_COLOR_GEOMETRY_UNCERTAIN = (105, 120, 132)
_COLOR_FREE_INSPECTED = (250, 250, 250)
_COLOR_FREE_UNINSPECTED = (255, 232, 160)
_COLOR_OBSTACLE = (20, 20, 20)
_COLOR_TRAVERSED_UNKNOWN = (145, 205, 225)
_COLOR_TRAVERSED_CONFLICT = (235, 65, 165)
_COLOR_RAW_FRONTIER = (35, 200, 210)
_COLOR_TRAJECTORY = (220, 40, 40)
_COLOR_TRAJECTORY_HISTORY = (150, 105, 105)
_COLOR_POSE = (40, 80, 220)
_COLOR_VIEW = (95, 155, 235)
_COLOR_INSTANCE = (30, 160, 60)
_COLOR_FRONTIER = (150, 60, 200)
_COLOR_ACTIVE_TARGET = (245, 145, 25)
_COLOR_POINTCLOUD_BACKGROUND = (248, 248, 248)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _reason_code(reason):
    return {"geometry": "G", "semantic": "S", "both": "B"}.get(
        str(reason or "").lower(), "?")


def _rasterize_orthographic_rgb(image, points, colors, px, py, floor_z=None,
                                units_per_m=0.0):
    """Aggregate gravity-aligned 3D evidence into a strict XY orthographic view.

    Every output location depends only on X/Y. Height is used solely to weight
    RGB samples that land in the same pixel, so it can never introduce camera
    perspective or move geometry in the image. A one-pixel dilation makes thin
    sampled surfaces legible without filling unsupported free space.
    """
    if len(points) == 0:
        return
    height, width = image.shape[:2]
    flat = py * width + px
    if floor_z is not None and units_per_m > 0:
        height_m = (points[:, 2] - float(floor_z)) / float(units_per_m)
        # Prefer object/wall evidence over floor samples when they share a
        # top-down pixel, while preventing high ceiling points from dominating.
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

    # Expand only into an immediately adjacent blank pixel. Use the original
    # occupancy mask for every offset so dilation cannot grow recursively.
    source_mask = occupied.copy()
    source_rgb = raster.copy()
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                   (1, 0), (-1, 1), (0, 1), (1, 1)):
        src_y0, src_y1 = max(0, -dy), min(height, height - dy)
        src_x0, src_x1 = max(0, -dx), min(width, width - dx)
        dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
        dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
        source = source_mask[src_y0:src_y1, src_x0:src_x1]
        destination = occupied[dst_y0:dst_y1, dst_x0:dst_x1]
        fill = source & ~destination
        if fill.any():
            target_rgb = image[dst_y0:dst_y1, dst_x0:dst_x1]
            candidate_rgb = source_rgb[src_y0:src_y1, src_x0:src_x1]
            target_rgb[fill] = candidate_rgb[fill]
            destination[fill] = True


def render_pointcloud_topdown(points, colors, pose=None, instances=None,
                              frontiers=None, active_target=None,
                              crop_center=None, crop_radius=None,
                              floor_z=None, unit_per_m=None,
                              min_image_side=512, max_image_side=1024,
                              max_plot_points=600000):
    """Render the decision VLM's compact RGB point-cloud bird's-eye view.

    The base layer is only reconstructed RGB geometry.  It deliberately has no
    occupancy/semantic region colors and no trajectory: the VLM sees the map
    evidence itself plus selectable frontier IDs, target-instance IDs, the
    active target and the current directed pose.  All coordinates must already
    be in the same gravity-aligned SLAM frame.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("point-cloud colors must match points")
    valid = np.isfinite(points).all(axis=1)
    if np.issubdtype(colors.dtype, np.floating):
        finite_colors = np.isfinite(colors).all(axis=1)
        colors = np.nan_to_num(colors, nan=0.0)
        if colors.size and float(np.max(colors)) <= 1.0:
            colors = colors * 255.0
        valid &= finite_colors
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    units_per_m = float(unit_per_m or 0.0)
    if floor_z is not None and units_per_m > 0:
        z = points[:, 2]
        valid &= (z >= float(floor_z) - 0.20 * units_per_m) & \
                 (z <= float(floor_z) + 2.70 * units_per_m)
    points, colors = points[valid], colors[valid]
    if len(points) == 0:
        return None

    # Deterministic sampling bounds render cost while retaining evidence from
    # the complete map rather than favoring recent frames.
    limit = max(int(max_plot_points), 1)
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
        points, colors = points[indices], colors[indices]

    marker_xy = []
    if pose is not None:
        marker_xy.append(np.asarray(pose[:2], dtype=np.float64))
    for item in list(instances or []) + list(frontiers or []):
        if item.get("xy") is not None:
            marker_xy.append(np.asarray(item["xy"], dtype=np.float64)[:2])
    if active_target is not None and active_target.get("xy") is not None:
        marker_xy.append(np.asarray(active_target["xy"], dtype=np.float64)[:2])

    if crop_center is not None and crop_radius is not None:
        center = np.asarray(crop_center, dtype=np.float64)[:2]
        radius = max(float(crop_radius), 1e-6)
        lo, hi = center - radius, center + radius
    else:
        lo, hi = np.percentile(points[:, :2], [0.5, 99.5], axis=0)
        if marker_xy:
            marker_array = np.stack(marker_xy)
            lo = np.minimum(lo, marker_array.min(axis=0))
            hi = np.maximum(hi, marker_array.max(axis=0))
        span = np.maximum(hi - lo, 1e-6)
        margin = max(0.05 * float(max(span)),
                     0.25 * units_per_m if units_per_m > 0 else 0.25)
        lo, hi = lo - margin, hi + margin

    in_view = ((points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) &
               (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1]))
    points, colors = points[in_view], colors[in_view]
    span = np.maximum(hi - lo, 1e-6)
    content_limit = max(64, int(max_image_side) - 24)
    scale = content_limit / float(max(span))
    content_w = max(32, int(math.ceil(span[0] * scale)))
    content_h = max(32, int(math.ceil(span[1] * scale)))
    canvas_w = max(int(min_image_side), content_w + 24)
    canvas_h = max(int(min_image_side), content_h + 24)
    ox = (canvas_w - content_w) // 2
    oy = (canvas_h - content_h) // 2
    image = np.full((canvas_h, canvas_w, 3),
                    _COLOR_POINTCLOUD_BACKGROUND, dtype=np.uint8)

    def to_px(xy):
        xy = np.asarray(xy, dtype=np.float64)[:2]
        x = int(round(ox + (xy[0] - lo[0]) * scale))
        # Image rows grow downward; map +Y is shown upward.
        y = int(round(oy + (hi[1] - xy[1]) * scale))
        return x, y

    if len(points):
        px = np.rint(ox + (points[:, 0] - lo[0]) * scale).astype(np.int64)
        py = np.rint(oy + (hi[1] - points[:, 1]) * scale).astype(np.int64)
        inside = ((px >= 0) & (px < canvas_w) &
                  (py >= 0) & (py < canvas_h))
        _rasterize_orthographic_rgb(
            image, points[inside], colors[inside], px[inside], py[inside],
            floor_z=floor_z, units_per_m=units_per_m)

    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    font = _font(14)
    marker = 9

    def label(xy, text, color):
        draw.text(xy, str(text), fill=color, font=font,
                  stroke_width=2, stroke_fill=(255, 255, 255))

    # Target evidence points are green circles; reported targets are crossed.
    for instance in instances or []:
        x, y = to_px(instance["xy"])
        draw.ellipse([x - marker, y - marker, x + marker, y + marker],
                     outline=_COLOR_INSTANCE, width=3)
        if instance.get("reported"):
            draw.line([(x - marker, y - marker),
                       (x + marker, y + marker)], fill=_COLOR_INSTANCE, width=2)
            draw.line([(x - marker, y + marker),
                       (x + marker, y - marker)], fill=_COLOR_INSTANCE, width=2)
        label((x + marker + 3, y - marker), f"t{instance['id']}",
              _COLOR_INSTANCE)

    # Only selectable frontiers are drawn; raw frontier boundaries are omitted.
    for frontier in frontiers or []:
        x, y = to_px(frontier["xy"])
        draw.polygon([(x, y - marker), (x + marker, y),
                      (x, y + marker), (x - marker, y)],
                     fill=_COLOR_FRONTIER, outline=(80, 25, 120))
        label((x + marker + 3, y - marker), frontier["id"], _COLOR_FRONTIER)

    if active_target is not None and active_target.get("xy") is not None:
        x, y = to_px(active_target["xy"])
        radius = marker + 4
        star = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            rr = radius if index % 2 == 0 else radius * 0.45
            star.append((x + rr * math.cos(angle),
                         y + rr * math.sin(angle)))
        draw.polygon(star, fill=_COLOR_ACTIVE_TARGET, outline=(160, 80, 0))
        label((x + radius + 3, y - radius),
              f"ACTIVE {active_target.get('id', '')}".strip(),
              _COLOR_ACTIVE_TARGET)

    if pose is not None:
        x, y = to_px(pose[:2])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        length = 28
        tip = (x + length * math.cos(yaw),
               y - length * math.sin(yaw))
        draw.line([(x, y), tip], fill=_COLOR_POSE, width=4)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=_COLOR_POSE,
                     outline=(255, 255, 255), width=1)
        label((x + 9, y + 2), "AGENT", _COLOR_POSE)

    # Marker-only legend; there is intentionally no region/occupancy legend.
    legend = "blue=agent   purple=fN frontier   green=tN target   orange=active"
    draw.text((8, 6), legend, fill=(25, 25, 25), font=_font(12),
              stroke_width=2, stroke_fill=(255, 255, 255))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_topdown(grid, trajectory=None, pose=None, instances=None,
                   frontiers=None, pixels_per_cell=4, active_target=None,
                   crop_center=None, crop_radius=None, frontier_layers=None,
                   frontier_stats=None,
                   step=None, map_revision=None, min_image_side=512,
                   max_image_side=1536, show_legend=True,
                   recent_trajectory_points=30):
    """渲染俯视标注地图，返回 PNG bytes。

    ``frontier_layers['unified']`` 是未过滤的青色边界；``frontiers`` 是
    经过聚类、A* 和冷却过滤后可供 VLM 选择的紫色候选。局部 crop 超出
    全局栅格的区域按几何未知处理。最终画布限制最大边长，避免尺度漂移
    产生的狭长全局地图被原样发送给决策 VLM。
    """
    free = np.asarray(grid.free, dtype=bool)
    obstacle = np.asarray(grid.obstacle, dtype=bool)
    geometry = np.asarray(getattr(
        grid, "geometry_observed", getattr(grid, "observed", free | obstacle)),
        dtype=bool)
    semantic_enabled = bool(getattr(
        grid, "semantic_coverage_enabled", False))
    semantic = np.asarray(getattr(
        grid, "semantic_inspected", geometry), dtype=bool)
    traversed = np.asarray(getattr(
        grid, "traversed", np.zeros_like(free)), dtype=bool)
    h, w = free.shape

    x0, y0, x1, y1 = 0, 0, w, h
    if crop_center is not None and crop_radius is not None:
        cx, cy = grid.world_to_cell(crop_center)
        radius_cells = max(1, int(math.ceil(
            float(crop_radius) / max(float(grid.res), 1e-12))))
        x0, y0 = cx - radius_cells, cy - radius_cells
        x1, y1 = cx + radius_cells + 1, cy + radius_cells + 1

    view_h, view_w = y1 - y0, x1 - x0
    ppc = max(1, int(pixels_per_cell))
    if min_image_side:
        ppc = max(ppc, int(math.ceil(
            float(min_image_side) / max(max(view_h, view_w), 1))))
    if max_image_side and max(view_h, view_w) * ppc > max_image_side:
        # 先降低栅格放大倍数以减少中间图像内存；不足一个像素/栅格时，
        # 仍由末尾的等比缩放兜底。
        ppc = max(1, int(max_image_side) // max(view_h, view_w))

    img = np.full(
        (view_h, view_w, 3), _COLOR_GEOMETRY_UNKNOWN, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x0, sy0 - y0
        dx1, dy1 = dx0 + sx1 - sx0, dy0 + sy1 - sy0
        local_free = free[sy0:sy1, sx0:sx1]
        local_obstacle = obstacle[sy0:sy1, sx0:sx1]
        local_geometry = geometry[sy0:sy1, sx0:sx1]
        local_semantic = semantic[sy0:sy1, sx0:sx1]
        local_traversed = traversed[sy0:sy1, sx0:sx1]
        patch = img[dy0:dy1, dx0:dx1]
        patch[local_geometry] = _COLOR_GEOMETRY_UNCERTAIN
        if semantic_enabled:
            patch[local_free & ~local_semantic] = _COLOR_FREE_UNINSPECTED
            patch[local_free & local_semantic] = _COLOR_FREE_INSPECTED
        else:
            patch[local_free] = _COLOR_FREE_INSPECTED
        patch[local_obstacle] = _COLOR_OBSTACLE
        # Traversed is deliberately visualized only where it disagrees with
        # occupancy. It never changes free/obstacle/frontier computation.
        patch[local_traversed & ~local_free & ~local_obstacle] = \
            _COLOR_TRAVERSED_UNKNOWN
        patch[local_traversed & local_obstacle] = \
            _COLOR_TRAVERSED_CONFLICT

        raw = None if frontier_layers is None else frontier_layers.get(
            "unified")
        if raw is not None:
            raw = np.asarray(raw, dtype=bool)
            if raw.shape == free.shape:
                patch[raw[sy0:sy1, sx0:sx1]] = _COLOR_RAW_FRONTIER

    pil = Image.fromarray(img).resize(
        (view_w * ppc, view_h * ppc), Image.Resampling.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(pil)
    marker = max(6, min(13, ppc))
    line_width = max(2, min(4, ppc // 3))
    font = _font(max(12, min(18, ppc)))
    small_font = _font(12)
    label_boxes = []

    def to_px(xy):
        cx, cy = grid.world_to_cell(xy)
        return ((cx - x0) * ppc + ppc // 2,
                (cy - y0) * ppc + ppc // 2)

    def label(xy, text, color, priority=False):
        """尽量避开已有标签；普通实例拥挤时宁可只保留标记。"""
        text = str(text)
        x, y = xy
        offsets = ((0, 0), (0, 16), (0, -16), (12, 0), (-36, 0))
        chosen = None
        for dx, dy in offsets:
            candidate = (x + dx, y + dy)
            box = draw.textbbox(candidate, text, font=font,
                                stroke_width=2)
            padded = (box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2)
            overlap = any(not (padded[2] < old[0] or padded[0] > old[2]
                                  or padded[3] < old[1]
                                  or padded[1] > old[3])
                          for old in label_boxes)
            if not overlap:
                chosen = (candidate, padded)
                break
        if chosen is None:
            if not priority:
                return False
            candidate = (x, y)
            box = draw.textbbox(candidate, text, font=font,
                                stroke_width=2)
            chosen = (candidate, box)
        draw.text(chosen[0], text, fill=color, font=font,
                  stroke_width=2, stroke_fill=(255, 255, 255))
        label_boxes.append(chosen[1])
        return True

    if trajectory:
        points = [to_px(p) for p in trajectory]
        if len(points) >= 2:
            split = max(0, len(points) - max(
                int(recent_trajectory_points), 2))
            if split >= 2:
                draw.line(points[:split + 1],
                          fill=_COLOR_TRAJECTORY_HISTORY, width=1)
            draw.line(points[split:], fill=_COLOR_TRAJECTORY,
                      width=line_width)

    # 过滤后候选统一画紫色菱形；后缀 G/S/B 只解释候选来源，不分裂 id。
    for frontier in frontiers or []:
        x, y = to_px(frontier["xy"])
        r = marker
        draw.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)],
                     fill=_COLOR_FRONTIER, outline=(80, 25, 120))
        text = f"{frontier['id']}:{_reason_code(frontier.get('reason'))}"
        label((x + r + 3, y - r), text, _COLOR_FRONTIER)

    for instance in instances or []:
        x, y = to_px(instance["xy"])
        r = marker
        draw.ellipse([x - r, y - r, x + r, y + r],
                     outline=_COLOR_INSTANCE, width=line_width)
        if instance.get("reported"):
            draw.line([(x - r, y - r), (x + r, y + r)],
                      fill=_COLOR_INSTANCE, width=line_width)
            draw.line([(x - r, y + r), (x + r, y - r)],
                      fill=_COLOR_INSTANCE, width=line_width)
        label((x + r + 3, y - r), str(instance["id"]), _COLOR_INSTANCE)

    if active_target is not None and active_target.get("xy") is not None:
        x, y = to_px(active_target["xy"])
        r = marker + 2
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            radius = r if index % 2 == 0 else r * 0.45
            points.append((x + radius * math.cos(angle),
                           y + radius * math.sin(angle)))
        draw.polygon(points, fill=_COLOR_ACTIVE_TARGET, outline=(160, 80, 0))
        label((x + r + 3, y - r),
              f"TARGET {active_target.get('id', '')}".strip(),
              _COLOR_ACTIVE_TARGET, priority=True)

    if pose is not None:
        x, y = to_px(pose[:2])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        view_length = max(24, marker * 3)
        for angle in (yaw - math.radians(60), yaw + math.radians(60)):
            endpoint = (x + view_length * math.cos(angle),
                        y + view_length * math.sin(angle))
            draw.line([(x, y), endpoint], fill=_COLOR_VIEW, width=2)
        tip = (x + view_length * math.cos(yaw),
               y + view_length * math.sin(yaw))
        draw.line([(x, y), tip], fill=_COLOR_POSE, width=line_width)
        r = marker // 2 + 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_COLOR_POSE)
        label((x + r + 3, y + 1), "YOU", _COLOR_POSE, priority=True)

    # 地图尺度来自 VGGT 相机高度尺规；没有可靠尺规时不画伪精确比例尺。
    unit_per_m = float(getattr(grid, "unit_per_m", 0.0) or 0.0)
    if unit_per_m > 0 and grid.res > 0:
        bar_px = int(round(unit_per_m / float(grid.res) * ppc))
        if 12 <= bar_px <= max(12, pil.width // 3):
            bx, by = 18, pil.height - 22
            draw.line([(bx, by), (bx + bar_px, by)],
                      fill=(255, 255, 255), width=5)
            draw.line([(bx, by), (bx + bar_px, by)],
                      fill=(20, 20, 20), width=2)
            draw.line([(bx, by - 4), (bx, by + 4)], fill=(20, 20, 20), width=2)
            draw.line([(bx + bar_px, by - 4), (bx + bar_px, by + 4)],
                      fill=(20, 20, 20), width=2)
            draw.text((bx, by - 18), "1 m", fill=(20, 20, 20),
                      font=small_font, stroke_width=2,
                      stroke_fill=(255, 255, 255))

    if show_legend:
        pose_status = "n/a"
        if pose is not None:
            pose_cell = grid.world_to_cell(pose[:2])
            if not grid.in_bounds(pose_cell):
                pose_status = "outside"
            else:
                px, py = pose_cell
                if obstacle[py, px]:
                    pose_status = "OBSTACLE"
                elif free[py, px]:
                    pose_status = "free"
                elif geometry[py, px]:
                    pose_status = "uncertain"
                else:
                    pose_status = "unknown"
        entries = [
            (_COLOR_FREE_INSPECTED, "free + semantically inspected"),
            (_COLOR_FREE_UNINSPECTED, "free + semantic inspection needed"),
            (_COLOR_GEOMETRY_UNCERTAIN, "geometry seen, occupancy uncertain"),
            (_COLOR_GEOMETRY_UNKNOWN, "geometry unseen"),
            (_COLOR_OBSTACLE, "obstacle / inflated obstacle"),
            (_COLOR_TRAVERSED_UNKNOWN, "traversed but geometry not free"),
            (_COLOR_TRAVERSED_CONFLICT, "traversed / obstacle CONFLICT"),
            (_COLOR_RAW_FRONTIER, "raw unified frontier boundary"),
            (_COLOR_FRONTIER, "selectable frontier fN:G/S/B"),
        ]
        row_h, box = 16, 10
        panel_h = 8 + len(entries) * row_h + 50
        # 图例放在地图外部，不能遮住真实边界或候选。
        canvas_w = max(pil.width, 540)
        canvas = Image.new("RGB", (canvas_w, pil.height + panel_h),
                           (245, 245, 245))
        canvas.paste(pil, ((canvas_w - pil.width) // 2, panel_h))
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, panel_h - 1), (canvas_w, panel_h - 1)],
                  fill=(80, 80, 80), width=1)
        title = f"step={step if step is not None else '?'}  " \
                f"map_rev={map_revision if map_revision is not None else '?'}  " \
                f"view={'local' if crop_center is not None else 'global'}  " \
                f"pose_cell={pose_status}"
        draw.text((10, 8), title, fill=(20, 20, 20), font=small_font)
        stats = dict(frontier_stats or {})
        status = (
            f"frontier raw={stats.get('raw_clusters', '?')}  "
            f"reachable={stats.get('reachable', '?')}  "
            f"selectable={stats.get('selectable', '?')}  "
            f"cooldown={stats.get('filtered_cooldown', '?')}")
        draw.text((10, 24), status, fill=(20, 20, 20), font=small_font)
        draw.text((10, 40), "map axes: +X right, +Y down; blue wedge = heading",
                  fill=(20, 20, 20), font=small_font)
        for index, (color, text) in enumerate(entries):
            yy = 59 + index * row_h
            draw.rectangle([10, yy, 10 + box, yy + box],
                           fill=color, outline=(60, 60, 60))
            draw.text((25, yy - 2), text, fill=(20, 20, 20),
                      font=small_font)
        pil = canvas

    # 保持宽高比限制发送给 VLM 的最大尺寸；极窄地图再用外部留白补齐
    # 最小画布，不拉伸地图内容。
    if max_image_side and max(pil.size) > int(max_image_side):
        pil.thumbnail((int(max_image_side), int(max_image_side)),
                      Image.Resampling.LANCZOS)
    if min_image_side and (pil.width < min_image_side or
                           pil.height < min_image_side):
        canvas = Image.new(
            "RGB", (max(pil.width, int(min_image_side)),
                    max(pil.height, int(min_image_side))),
            _COLOR_GEOMETRY_UNKNOWN)
        canvas.paste(pil, ((canvas.width - pil.width) // 2,
                           (canvas.height - pil.height) // 2))
        pil = canvas

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
