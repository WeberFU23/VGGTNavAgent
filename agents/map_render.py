"""俯视标注地图渲染（Phase 4a，agent 端，仅依赖 numpy + PIL）。

占据栅格 -> 俯视图 PNG：free 白 / obstacle 黑 / unknown 灰；叠加历史
轨迹（红折线）、当前位姿箭头（蓝）、confirmed 实例（绿编号实线圈）、
belief 锚点（橙编号虚线圈）、frontier（紫编号十字）。编号与决策状态
JSON 中的 id 严格一一对应（由调用方传入同一个 id）。
"""

import io
import math

import numpy as np
from PIL import Image, ImageDraw

_COLOR_TRAJECTORY = (220, 40, 40)
_COLOR_POSE = (40, 80, 220)
_COLOR_INSTANCE = (30, 160, 60)
_COLOR_ANCHOR = (230, 150, 30)
_COLOR_FRONTIER = (150, 60, 200)


def render_topdown(grid, trajectory=None, pose=None, instances=None,
                   anchors=None, frontiers=None, pixels_per_cell=4):
    """渲染俯视标注地图，返回 PNG bytes。

    grid: OccupancyGrid（free/obstacle (H,W) bool，[y, x] 索引）；
    trajectory: [(x, y), ...] 世界坐标折线；
    pose: (x, y, yaw) 当前位姿（yaw 为对齐地图系朝向，弧度）；
    instances: [{"id", "xy", "visited"}] confirmed/visited 实例；
    anchors: [{"id", "xy"}] belief 锚点；
    frontiers: [{"id", "xy"}] 探索前沿。
    """
    free = np.asarray(grid.free)
    obstacle = np.asarray(grid.obstacle)
    h, w = free.shape
    ppc = max(1, int(pixels_per_cell))
    img = np.full((h, w, 3), 128, dtype=np.uint8)   # unknown 灰
    img[free] = (255, 255, 255)
    img[obstacle] = (0, 0, 0)
    pil = Image.fromarray(img).resize(
        (w * ppc, h * ppc), Image.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(pil)

    def to_px(xy):
        cx, cy = grid.world_to_cell(xy)
        return cx * ppc + ppc // 2, cy * ppc + ppc // 2

    # 历史轨迹
    if trajectory:
        pts = [to_px(p) for p in trajectory]
        if len(pts) >= 2:
            draw.line(pts, fill=_COLOR_TRAJECTORY, width=1)

    # frontier：编号十字
    for fr in frontiers or []:
        x, y = to_px(fr["xy"])
        r = max(3, ppc)
        draw.line([(x - r, y), (x + r, y)], fill=_COLOR_FRONTIER, width=2)
        draw.line([(x, y - r), (x, y + r)], fill=_COLOR_FRONTIER, width=2)
        draw.text((x + r + 1, y - r), str(fr["id"]), fill=_COLOR_FRONTIER)

    # belief 锚点：编号虚线圈
    for a in anchors or []:
        x, y = to_px(a["xy"])
        r = max(5, ppc * 2)
        _dashed_circle(draw, (x, y), r, _COLOR_ANCHOR)
        draw.text((x + r + 1, y - r), str(a["id"]), fill=_COLOR_ANCHOR)

    # confirmed 实例：编号实线圈（visited 画叉）
    for inst in instances or []:
        x, y = to_px(inst["xy"])
        r = max(5, ppc * 2)
        draw.ellipse([x - r, y - r, x + r, y + r],
                     outline=_COLOR_INSTANCE, width=2)
        if inst.get("visited"):
            draw.line([(x - r, y - r), (x + r, y + r)],
                      fill=_COLOR_INSTANCE, width=2)
            draw.line([(x - r, y + r), (x + r, y - r)],
                      fill=_COLOR_INSTANCE, width=2)
        draw.text((x + r + 1, y - r), str(inst["id"]), fill=_COLOR_INSTANCE)

    # 当前位姿箭头
    if pose is not None:
        x, y = to_px(pose[:2])
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        length = max(8, ppc * 3)
        # 图像 y 轴向下，与栅格行索引方向一致，直接用 (cos, sin)
        tip = (x + length * math.cos(yaw), y + length * math.sin(yaw))
        draw.line([(x, y), tip], fill=_COLOR_POSE, width=2)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=_COLOR_POSE)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _dashed_circle(draw, center, r, color, segments=12, duty=0.55):
    x, y = center
    step = 360.0 / segments
    for i in range(segments):
        a0 = i * step
        a1 = a0 + step * duty
        draw.arc([x - r, y - r, x + r, y + r], a0, a1, fill=color, width=2)
