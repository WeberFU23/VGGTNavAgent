"""决策 VLM 使用的俯视地图渲染器。

底图同时表达几何重建和语义检查状态；叠加层显示统一 frontier 的原始
边界、经过可达性/冷却过滤的候选、实例、轨迹、当前位姿和活动目标。
调用方应传入生成 frontier 时的同一 OccupancyGrid 快照。
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
_COLOR_RAW_FRONTIER = (35, 200, 210)
_COLOR_TRAJECTORY = (220, 40, 40)
_COLOR_TRAJECTORY_HISTORY = (150, 105, 105)
_COLOR_POSE = (40, 80, 220)
_COLOR_VIEW = (95, 155, 235)
_COLOR_INSTANCE = (30, 160, 60)
_COLOR_FRONTIER = (150, 60, 200)
_COLOR_ACTIVE_TARGET = (245, 145, 25)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _reason_code(reason):
    return {"geometry": "G", "semantic": "S", "both": "B"}.get(
        str(reason or "").lower(), "?")


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
        patch = img[dy0:dy1, dx0:dx1]
        patch[local_geometry] = _COLOR_GEOMETRY_UNCERTAIN
        if semantic_enabled:
            patch[local_free & ~local_semantic] = _COLOR_FREE_UNINSPECTED
            patch[local_free & local_semantic] = _COLOR_FREE_INSPECTED
        else:
            patch[local_free] = _COLOR_FREE_INSPECTED
        patch[local_obstacle] = _COLOR_OBSTACLE

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
        entries = [
            (_COLOR_FREE_INSPECTED, "free + semantically inspected"),
            (_COLOR_FREE_UNINSPECTED, "free + semantic inspection needed"),
            (_COLOR_GEOMETRY_UNCERTAIN, "geometry seen, occupancy uncertain"),
            (_COLOR_GEOMETRY_UNKNOWN, "geometry unseen"),
            (_COLOR_OBSTACLE, "obstacle / inflated obstacle"),
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
                f"view={'local' if crop_center is not None else 'global'}"
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
