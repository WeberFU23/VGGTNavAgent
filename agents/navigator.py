"""导航执行模块（client 侧，纯 numpy，不依赖 habitat/torch）。

把 VGGT-SLAM 的相对尺度地图变成可执行的离散动作序列：

1. 重力对齐：相机旋转（经安装俯仰角补偿）的稳健中位数估计竖直轴，
   得到 z'=up 的 2D 规划平面。
2. 占据栅格：点云按高度分层——近地面层为可行走面，腰部高度层为障碍；
   障碍物按机器人半径膨胀。相机轨迹仅记录在独立 ``traversed`` 层，
   不参与自由空间和障碍物分类。
3. A* 寻路 + 视线捷径简化。
4. 路径跟随：最新关键帧位姿为锚点，动作序列做航位推算（turn=±30°，
   forward=0.25m/scale），输出 benchmark 离散动作。

坐标约定：对齐坐标系右手系、z' 向上；偏航 yaw 为绕 z' 的角度，
TURN_LEFT = +30°（俯视逆时针），MOVE_FORWARD 沿 (cos yaw, sin yaw)。
所有长度在规划内部用地图单位（米 = 单位 * scale，scale 来自
mapping.scale_calibration.ScaleCalibrator）。
"""

import heapq
import math

import numpy as np

MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

FORWARD_STEP_M = 0.25
TURN_STEP_RAD = math.radians(30.0)


# 相机安装下俯角。配置标称 30°（hm3d_config.yaml orientation [-pi/6,0,0]），
# 但实测（scripts/diagnostics/check_gravity.py，238 关键帧）散布最小值在 40°——
# 标称值与 VGGT 位姿链的合成有效角有 ~10° 偏差，以实测为准。
MOUNT_PITCH_DOWN_RAD = math.radians(40.0)


def mount_compensated_cam_up(pitch_down_rad=MOUNT_PITCH_DOWN_RAD):
    """相机相对机身下俯 pitch_down_rad 时，世界 up 在相机系中的方向。

    benchmark 的 RGB 传感器固定下俯 30°（hm3d_config.yaml 的
    orientation: [-pi/6, 0, 0]），机身只转 yaw、不滚转不俯仰，
    因此世界 up 在相机系里是解析已知的常量：
    下俯 theta 时 u_cam = Rx(theta) @ [0,-1,0] = [0, -cos, -sin]。
    这是 agent 自身的安装外参，不属于模拟器特权信息。
    """
    c = math.cos(pitch_down_rad)
    s = math.sin(pitch_down_rad)
    return np.array([0.0, -c, -s])


def gravity_alignment(poses, cam_up=None, pca_refine=True):
    """用 SLAM 相机旋转的稳健平均 up 估计重力方向。

    cam_up：世界 up 在相机系中的方向（常量）。默认 [0,-1,0]（相机
    水平安装，OpenCV y 轴朝下）；传感器带俯仰角时应传入
    mount_compensated_cam_up() 的结果，否则估计出的"重力"会系统性
    偏离真值一个安装角（benchmark 相机下俯 30°）。

    pca_refine：轨迹有足够 2D 覆盖时（单楼层、非直线），用相机中心
    协方差的最小特征向量修正 up——实测能消除安装角模型的几度残差
    （中位数法校准到 40° 后仍与 PCA 差 6.2°）。轨迹近直线或多楼层
    （平面性检查不通过）时自动回退到中位数法。

    R @ p 把点旋转到 z'=up 的对齐坐标系（行向量为新坐标轴）。
    """
    poses = np.asarray(poses, dtype=np.float64)
    if cam_up is None:
        cam_up = np.array([0.0, -1.0, 0.0])
    cam_up = np.asarray(cam_up, dtype=np.float64)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)
    ups = poses[:, :3, :3] @ cam_up
    reference = ups[0]
    ups = np.asarray([u if np.dot(u, reference) >= 0 else -u for u in ups])
    up = np.median(ups, axis=0)
    up /= np.linalg.norm(up) + 1e-9

    if pca_refine and len(poses) >= 8:
        centers = poses[:, :3, 3]
        cov = np.cov((centers - centers.mean(axis=0)).T)
        eigval, eigvec = np.linalg.eigh(cov)
        # 平面性：最小特征值远小于次小——相机近似共面（单楼层）
        if eigval[0] < 0.05 * max(eigval[1], 1e-12):
            up_pca = eigvec[:, 0]
            if np.dot(up_pca, up) < 0:
                up_pca = -up_pca
            up = up_pca / (np.linalg.norm(up_pca) + 1e-12)
    fwd = poses[:, :3, 2].mean(0)
    fwd = fwd - np.dot(fwd, up) * up
    if np.linalg.norm(fwd) < 1e-6:
        fwd = poses[0, :3, 0]
        fwd = fwd - np.dot(fwd, up) * up
    x = np.cross(up, fwd)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(up, x)
    return np.stack([x, y, up], axis=0)


def pose_to_yaw_2d(pose, R):
    """cam2world 位姿 -> 对齐坐标系下的 (x, y, yaw)。"""
    pos = R @ pose[:3, 3]
    fwd = R @ (pose[:3, :3] @ np.array([0.0, 0.0, 1.0]))
    yaw = math.atan2(fwd[1], fwd[0])
    return float(pos[0]), float(pos[1]), yaw


def _flood_component(mask, start):
    """4-连通 flood fill：返回 start 所在的 True 连通域（其余置 False）。"""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sx, sy = int(start[0]), int(start[1])
    if not (0 <= sx < w and 0 <= sy < h) or not mask[sy, sx]:
        return seen
    seen[sy, sx] = True
    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] \
                    and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((nx, ny))
    return seen


def _binary_dilate(mask, iterations=1):
    """用 3x3 邻域膨胀布尔栅格，边界外恒为 False。

    显式 padding 很重要：``np.roll`` 会让左边界的障碍出现在右边界，
    从而在地图两侧制造并不存在的障碍或自由空间。
    """
    result = np.asarray(mask, dtype=bool).copy()
    h, w = result.shape
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(result, 1, constant_values=False)
        grown = np.zeros_like(result)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy:1 + dy + h,
                                1 + dx:1 + dx + w]
        result = grown
    return result


def _trajectory_mask(shape, origin, res, cam_centers, radius_cells=0):
    """Rasterize a camera polyline without changing occupancy evidence."""
    mask = np.zeros(shape, dtype=bool)
    centers = np.asarray(cam_centers, dtype=np.float64)
    if centers.size == 0:
        return mask
    centers = centers.reshape(-1, centers.shape[-1])
    cells = np.floor((centers[:, :2] - origin) / res).astype(np.int64)
    line_cells = []
    for index, cell in enumerate(cells):
        x0, y0 = int(cell[0]), int(cell[1])
        if index:
            x1, y1 = (int(cells[index - 1, 0]),
                      int(cells[index - 1, 1]))
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            while (x0, y0) != (x1, y1):
                line_cells.append((x0, y0))
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
        line_cells.append((int(cell[0]), int(cell[1])))

    h, w = mask.shape
    radius = max(int(radius_cells), 0)
    for cx, cy in line_cells:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    mask[y, x] = True
    return mask


def _nearest_mask_cell(mask, start, max_radius):
    """Return the nearest True cell around ``start`` without mutating ``mask``."""
    h, w = mask.shape
    sx, sy = int(start[0]), int(start[1])
    for radius in range(max(int(max_radius), 0) + 1):
        candidates = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if radius and max(abs(dx), abs(dy)) != radius:
                    continue
                x, y = sx + dx, sy + dy
                if 0 <= x < w and 0 <= y < h and mask[y, x]:
                    candidates.append((dx * dx + dy * dy, x, y))
        if candidates:
            _, x, y = min(candidates)
            return x, y
    return None


class OccupancyGrid:
    """2D 占据栅格（对齐坐标系），单位与输入点云一致（地图单位）。"""

    def __init__(self, res, origin, free, obstacle, observed=None,
                 semantic_inspected=None, semantic_coverage_enabled=False,
                 traversed=None, ground_votes=None, obstacle_votes=None,
                 raw_obstacle=None):
        self.res = float(res)           # 地图单位/格
        if not math.isfinite(self.res) or self.res <= 0:
            raise ValueError(f"grid resolution must be positive: {res!r}")
        self.origin = np.asarray(origin, dtype=np.float64).reshape(-1)
        if self.origin.size != 2:
            raise ValueError("grid origin must contain exactly two coordinates")
        self.free = np.asarray(free, dtype=bool)
        self.obstacle = np.asarray(obstacle, dtype=bool)
        if self.free.ndim != 2 or self.free.shape != self.obstacle.shape:
            raise ValueError("free and obstacle must be same-shaped 2D arrays")
        # 几何覆盖只回答“该格是否进入过可靠的 3D 重建”，不表示目标已被
        # 语义模型看清。observed 保留为兼容别名，避免旧调用混淆 occupancy。
        self.geometry_observed = np.asarray(
            self.free | self.obstacle if observed is None else observed,
            dtype=bool)
        if self.geometry_observed.shape != self.free.shape:
            raise ValueError("observed must match the occupancy grid shape")
        self.observed = self.geometry_observed
        # 未显式提供语义层时沿用几何层，保持历史合成测试的行为；真实 agent
        # 会在拿到 caption 完成帧后覆盖此值并打开 semantic_coverage_enabled。
        self.semantic_inspected = np.asarray(
            self.geometry_observed if semantic_inspected is None
            else semantic_inspected, dtype=bool)
        if self.semantic_inspected.shape != self.free.shape:
            raise ValueError(
                "semantic_inspected must match the occupancy grid shape")
        self.semantic_coverage_enabled = bool(semantic_coverage_enabled)
        self.semantic_view_count = np.zeros(self.free.shape, dtype=np.uint16)
        # ``traversed`` is physical execution evidence, not geometric free-space
        # evidence. Keeping it separate prevents the robot path from painting a
        # corridor into the occupancy map.
        self.traversed = self._layer(
            traversed, bool, "traversed", default=False)
        self.ground_votes = self._layer(
            ground_votes, np.uint32, "ground_votes", default=0)
        self.obstacle_votes = self._layer(
            obstacle_votes, np.uint32, "obstacle_votes", default=0)
        self.raw_obstacle = self._layer(
            raw_obstacle, bool, "raw_obstacle", default=False)
        self.start_cell = None
        self.start_seed_cell = None
        self.start_seed_distance_cells = None
        self.connectivity_filtered = False

    def _layer(self, value, dtype, name, default):
        if value is None:
            return np.full(self.free.shape, default, dtype=dtype)
        layer = np.asarray(value, dtype=dtype)
        if layer.shape != self.free.shape:
            raise ValueError(f"{name} must match the occupancy grid shape")
        return layer

    def update_semantic_coverage(self, frames, captioned_frame_ids, align_R,
                                 max_range_m=4.0, close_range_m=2.0,
                                 min_views=2, camera_disk_m=0.4,
                                 min_view_angle_deg=25.0,
                                 min_view_baseline_m=0.5):
        """从已完成 caption 的关键帧建立语义检查层。

        每个有效 VGGT 点本身就是未被遮挡的可见表面。一个可通行格满足
        “一次近距离观察”，或至少 ``min_views`` 个 caption 帧且视点之间
        同时具有足够基线和方位角差，才算完成语义检查。连续同向关键帧
        不再被误当作多视角证据。
        """
        completed = {int(fid) for fid in captioned_frame_ids or []}
        counts = np.zeros(self.free.shape, dtype=np.uint16)
        close = np.zeros(self.free.shape, dtype=bool)
        diverse = np.zeros(self.free.shape, dtype=bool)
        first_angle = np.full(self.free.shape, np.nan, dtype=np.float32)
        first_camera_x = np.full(self.free.shape, np.nan, dtype=np.float32)
        first_camera_y = np.full(self.free.shape, np.nan, dtype=np.float32)
        if not completed:
            self.semantic_inspected = np.zeros_like(self.free)
            self.semantic_view_count = counts
            self.semantic_coverage_enabled = True
            return self.semantic_inspected

        # 子图重叠可能重复返回同一 frame；按 frame_id 去重，防止把一次
        # 观察误算成多视角证据。
        unique_frames = {}
        for frame in frames or []:
            fid = int(frame.get("frame_id", -1))
            if fid in completed:
                unique_frames[fid] = frame

        units_per_m = max(float(getattr(self, "unit_per_m", 1.0)), 1e-9)
        max_range = max(float(max_range_m), 0.0) * units_per_m
        close_range = max(float(close_range_m), 0.0) * units_per_m
        min_baseline = max(float(min_view_baseline_m), 0.0) * units_per_m
        min_angle = math.radians(max(float(min_view_angle_deg), 0.0))
        h, w = self.free.shape
        for frame in unique_frames.values():
            points = np.asarray(frame["points"], dtype=np.float64)
            finite = np.isfinite(points).all(axis=1)
            if not finite.any():
                continue
            aligned = points[finite] @ align_R.T
            camera = (np.asarray(frame["pose"], dtype=np.float64)[:3, 3]
                      @ align_R.T)
            distances = np.linalg.norm(aligned[:, :2] - camera[:2], axis=1)
            valid = distances <= max_range
            if valid.any():
                cells = np.floor(
                    (aligned[valid, :2] - self.origin) / self.res
                ).astype(np.int64)
                inside = ((cells[:, 0] >= 0) & (cells[:, 0] < w) &
                          (cells[:, 1] >= 0) & (cells[:, 1] < h))
                cells = cells[inside]
                if len(cells):
                    linear, first_indices = np.unique(
                        cells[:, 1] * w + cells[:, 0], return_index=True)
                    ys, xs = np.divmod(linear, w)
                    counts[ys, xs] += 1
                    # 从格子中心指向相机的方位角。以该格第一次有效观察为
                    # 基准，后续观察必须既有视角差又有空间基线。
                    sample_cells = cells[first_indices]
                    cell_xy = self.origin + (
                        sample_cells.astype(np.float64) + 0.5) * self.res
                    angles = np.arctan2(
                        camera[1] - cell_xy[:, 1],
                        camera[0] - cell_xy[:, 0])
                    previous = first_angle[ys, xs]
                    seen = np.isfinite(previous)
                    if seen.any():
                        delta = np.abs(np.arctan2(
                            np.sin(angles[seen] - previous[seen]),
                            np.cos(angles[seen] - previous[seen])))
                        baseline = np.hypot(
                            camera[0] - first_camera_x[ys[seen], xs[seen]],
                            camera[1] - first_camera_y[ys[seen], xs[seen]])
                        diverse[ys[seen], xs[seen]] |= (
                            (delta >= min_angle) &
                            (baseline >= min_baseline))
                    fresh = ~seen
                    if fresh.any():
                        first_angle[ys[fresh], xs[fresh]] = angles[fresh]
                        first_camera_x[ys[fresh], xs[fresh]] = camera[0]
                        first_camera_y[ys[fresh], xs[fresh]] = camera[1]

                near = valid & (distances <= close_range)
                near_cells = np.floor(
                    (aligned[near, :2] - self.origin) / self.res
                ).astype(np.int64)
                near_inside = ((near_cells[:, 0] >= 0) &
                               (near_cells[:, 0] < w) &
                               (near_cells[:, 1] >= 0) &
                               (near_cells[:, 1] < h))
                near_cells = near_cells[near_inside]
                if len(near_cells):
                    linear = np.unique(
                        near_cells[:, 1] * w + near_cells[:, 0])
                    ys, xs = np.divmod(linear, w)
                    close[ys, xs] = True

            # 相机脚下通常没有可回投影地面点，但显然属于近距离检查区。
            cx, cy = self.world_to_cell(camera[:2])
            radius = max(0, int(math.ceil(
                float(camera_disk_m) * units_per_m / self.res)))
            for yy in range(max(0, cy - radius), min(h, cy + radius + 1)):
                for xx in range(max(0, cx - radius), min(w, cx + radius + 1)):
                    if (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2:
                        close[yy, xx] = True

        required_views = max(int(min_views), 1)
        multi_view = counts >= required_views
        if required_views > 1:
            multi_view &= diverse
        inspected = multi_view | close
        # 语义层只解释已有几何表面；不能越过未重建区域或障碍膨胀边界。
        self.semantic_inspected = inspected & self.geometry_observed & self.free
        self.semantic_view_count = counts
        self.semantic_coverage_enabled = True
        return self.semantic_inspected

    @classmethod
    def from_trajectory(cls, cam_centers_aligned, stamp=3):
        """Build a diagnostic trajectory-only grid with no inferred free cells.

        This compatibility constructor intentionally produces only the
        ``traversed`` layer. A robot path proves that motion occurred, but it
        does not prove that nearby cells are currently geometrically free.
        Navigation must not use this grid as an occupancy fallback.
        """
        xy = np.asarray(cam_centers_aligned, dtype=np.float64)[:, :2]
        if len(xy) < 2:
            return None
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        d = d[d > 1e-9]
        spacing = float(np.median(d)) if len(d) else 1.0
        res = spacing / 6.0           # ≈0.1m（关键帧间距约 0.5~0.7m）
        margin = spacing * 5
        qlo = xy.min(axis=0) - margin
        qhi = xy.max(axis=0) + margin
        w = max(int(np.ceil((qhi[0] - qlo[0]) / res)), 8)
        h = max(int(np.ceil((qhi[1] - qlo[1]) / res)), 8)
        free = np.zeros((h, w), dtype=bool)
        traversed = _trajectory_mask(
            free.shape, qlo, res, cam_centers_aligned,
            radius_cells=max(int(stamp), 0))
        grid = cls(
            res, qlo, free, np.zeros_like(free),
            observed=np.zeros_like(free),
            semantic_inspected=np.zeros_like(free), traversed=traversed)
        grid.floor_z = None
        grid.unit_per_m = 1.0 / (spacing * 1.5) if spacing else 0.0  # 调试估算
        return grid

    @classmethod
    def build(cls, points_aligned, cam_centers_aligned,
              res_m=0.10, floor_band_m=0.12, obs_low_m=0.15, obs_high_m=1.8,
              robot_radius_m=0.25, margin_m=0.5, cam_height_m=1.5,
              unit_per_m=None, point_frame_ids=None,
              voxel_size_m=0.05, min_voxel_views=3,
              min_obstacle_votes=2, raycast_free=True,
              ray_clear_votes=5, cam_frame_ids=None):
        """从对齐点云构建栅格。

        有动作尺度时，地板峰只在“相机下方约 1.5m”的范围内搜索；没有
        动作尺度时，才从全局高度峰和水平覆盖反推地图单位。两种路径都
        只使用融合后的 3D 点与相机中心，不使用图像底部行。
        """
        points_aligned = np.asarray(points_aligned, dtype=np.float64)
        finite = np.isfinite(points_aligned).all(axis=1)
        if point_frame_ids is not None:
            point_frame_ids = np.asarray(point_frame_ids).reshape(-1)
            if len(point_frame_ids) != len(points_aligned):
                raise ValueError("point_frame_ids must match points_aligned")
            point_frame_ids = point_frame_ids[finite]
        points_aligned = points_aligned[finite]
        cam_centers_aligned = np.asarray(
            cam_centers_aligned, dtype=np.float64)
        finite_cameras = np.isfinite(cam_centers_aligned).all(axis=1)
        cam_centers_aligned = cam_centers_aligned[finite_cameras]
        if len(points_aligned) == 0:
            return None
        xy = points_aligned[:, :2]
        z = points_aligned[:, 2]

        # 地面高度：直方图里"最低的显著峰"。不能简单取计数最大的
        # bin——杂乱房间里桌/床面（0.7~0.9m 高）的点可能比地板还多
        # （实测地板被误判高 0.65m，尺规随之缩水近一半）。
        cam_h = float(np.median(cam_centers_aligned[:, 2])) \
            if len(cam_centers_aligned) else float(np.median(z))
        z_min = float(np.percentile(z, 1))
        supplied_units = unit_per_m is not None and \
            math.isfinite(float(unit_per_m)) and float(unit_per_m) > 0
        ruler_est = (cam_height_m * float(unit_per_m)) if supplied_units else \
            max(cam_h - z_min, 1e-6)

        # 融合点云仍包含每个关键帧各自预测的稠密点。先量化为 3D 体素，
        # 每帧对同一体素最多贡献一次支持，只保留至少两个独立视角重建的
        # 表面。输出体素中心后，后续高度统计不再受单帧点密度支配。
        voxel_size = max(float(voxel_size_m) * (
            float(unit_per_m) if supplied_units else ruler_est / cam_height_m),
            1e-6)
        source_point_count = len(points_aligned)
        retained_voxel_count = source_point_count
        # 射线法自由空间用原始（未体素融合）点：融合后丢失帧关联，
        # 而射线必须知道每个点属于哪个相机。
        raw_ray = None
        if raycast_free and point_frame_ids is not None:
            raw_ray = (xy.copy(), z.copy(), point_frame_ids.copy())
        if point_frame_ids is not None and min_voxel_views > 1 and \
                np.unique(point_frame_ids).size >= min_voxel_views:
            cells = np.floor(points_aligned / voxel_size).astype(np.int64)
            frame_cells = np.column_stack([point_frame_ids, cells])
            frame_cells = np.unique(frame_cells, axis=0)
            voxels, view_counts = np.unique(
                frame_cells[:, 1:], axis=0, return_counts=True)
            keep = view_counts >= int(min_voxel_views)
            if keep.any():
                points_aligned = (voxels[keep].astype(np.float64) + 0.5) \
                    * voxel_size
                xy = points_aligned[:, :2]
                z = points_aligned[:, 2]
                retained_voxel_count = int(keep.sum())
        lo, hi = np.percentile(z, [1, 99])
        nbins = max(int((hi - lo) / (0.033 * ruler_est)), 10)
        hist, edges = np.histogram(z, bins=nbins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        below = centers < cam_h - 0.15 * ruler_est
        if supplied_units:
            # 动作里程计已经给出地图尺度时，地板应位于相机下方约一个
            # cam_height。只在 ±0.35m 的窄带内找峰，避免桌面/床面凭更高
            # 点密度或水平覆盖被选成地板。
            expected_floor = cam_h - cam_height_m * float(unit_per_m)
            search_radius = 0.35 * float(unit_per_m)
            below &= np.abs(centers - expected_floor) <= search_radius
        if not below.any():
            return None
        # 平滑后找局部峰，取高度最低且计数 >= 最大峰 30% 的峰
        h_smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
        peaks = []
        for i in range(1, nbins - 1):
            if below[i] and h_smooth[i] >= h_smooth[i - 1] and \
                    h_smooth[i] >= h_smooth[i + 1]:
                peaks.append(i)
        if not peaks:
            return None
        strong = [i for i in peaks
                  if h_smooth[i] >= 0.3 * h_smooth[below].max()]
        if strong:
            candidates = strong
        else:
            valid_bins = np.flatnonzero(below)
            candidates = [int(valid_bins[np.argmax(h_smooth[valid_bins])])]
        # 候选峰中选水平覆盖最广的：地板铺满整个可行走区域，
        # 桌/床面计数可以超过地板，但水平 footprint 永远有限。
        # （尺度无关，替代之前"取最低强峰"——实测会锁到床面，
        # 尺规随之缩水数倍，整个分层失效。）
        cspan = np.maximum(np.percentile(xy, 99, axis=0)
                           - np.percentile(xy, 1, axis=0), 1e-6)
        cres = cspan / 64.0
        cxy = np.floor((xy - np.percentile(xy, 1, axis=0)) / cres) \
            .astype(np.int64)
        band = 0.06 * ruler_est
        best_i, best_cov = candidates[0], -1
        for i in candidates:
            inband = np.abs(z - centers[i]) < band
            cov_count = np.unique(
                cxy[inband, 0] * 4096 + cxy[inband, 1]).size
            if cov_count > best_cov:
                best_i, best_cov = i, cov_count
        floor_z = float(centers[best_i])
        # 三格平滑可能把尖锐地板峰的局部最大值推到相邻 bin。回到原始点
        # 上取窄带中位数，避免半个 bin 的高度偏差吞掉低矮障碍。
        refine_radius = max(0.10 * ruler_est,
                            float(edges[1] - edges[0]) * 1.5)
        refine = np.abs(z - floor_z) <= refine_radius
        if refine.any():
            floor_z = float(np.median(z[refine]))

        # 尺规：相机离地高度（地图单位）≈ cam_height_m 米
        ruler = max(cam_h - floor_z, 1e-6)
        # 有可靠动作尺度时保持米制分辨率固定；地板峰仅负责 floor_z，不能
        # 反过来用少量高度误差改写整张地图的尺度。
        u = float(unit_per_m) if supplied_units else ruler / cam_height_m
        res = res_m * u

        # 分层（分块地板高度）：回环修正/对齐误差会把地板点垂直涂抹
        # 开，全局单一 floor_z + 窄带会漏掉大部分地板（实测 free 层只剩
        # 零头）。按瓦片各自估计地板高度，证据不足或明显不可能时回落
        # 全局 floor_z。
        is_floor, is_obs = cls._classify_layers(
            xy, z, floor_z, cam_h, u, floor_band_m, obs_low_m, obs_high_m)

        # 栅格边界（裁剪离群点 + 边距）
        qlo = np.percentile(xy, 0.5, axis=0) - margin_m * u
        qhi = np.percentile(xy, 99.5, axis=0) + margin_m * u
        # 保证相机中心都在界内
        if len(cam_centers_aligned):
            qlo = np.minimum(qlo, cam_centers_aligned[:, :2].min(0) - res)
            qhi = np.maximum(qhi, cam_centers_aligned[:, :2].max(0) + res)
        w = np.maximum(int(np.ceil((qhi[0] - qlo[0]) / res)), 8)
        h = np.maximum(int(np.ceil((qhi[1] - qlo[1]) / res)), 8)

        def raster(mask, splat=0):
            """splat>0 时把每个点溅射到 (2*splat+1)^2 邻域——点间距大于
            栅格分辨率时（如地板点 0.2m 间距 vs 0.1m 格）避免棋盘格断连。"""
            grid = np.zeros((h, w), dtype=bool)
            p = xy[mask]
            cell = np.floor((p - qlo) / res).astype(np.int64)
            for dx in range(-splat, splat + 1):
                for dy in range(-splat, splat + 1):
                    c = cell + (dx, dy)
                    ok = (c[:, 0] >= 0) & (c[:, 0] < w) & \
                         (c[:, 1] >= 0) & (c[:, 1] < h)
                    grid[c[ok, 1], c[ok, 0]] = True
            return grid

        def raster_count(mask):
            """不溅射的逐格计数（障碍票数用）。"""
            cnt = np.zeros((h, w), dtype=np.int32)
            p = xy[mask]
            cell = np.floor((p - qlo) / res).astype(np.int64)
            ok = (cell[:, 0] >= 0) & (cell[:, 0] < w) & \
                 (cell[:, 1] >= 0) & (cell[:, 1] < h)
            np.add.at(cnt, (cell[ok, 1], cell[ok, 0]), 1)
            return cnt

        free = raster(is_floor, splat=1)
        # 孤立障碍点才滤除：单点噪声（无相邻障碍、票数不足）若直接
        # 参与 0.25m 半径膨胀会抹掉大片自由空间；薄但连续的表面
        # （桌面等，体素融合后每格仅 1 票）靠邻域支持保留。
        obs_votes = raster_count(is_obs)
        obs_raw = obs_votes >= 1
        obs_has_neighbor = _binary_dilate(obs_raw, iterations=1)
        obstacle = obs_raw & (
            (obs_votes >= int(min_obstacle_votes)) | obs_has_neighbor)
        if raw_ray is not None and len(cam_centers_aligned):
            # 射线法：相机到观测点的连线在躯干高度带内经过的格子为自由。
            # 光线既然能到达表面，路径就是空气——与地板高度分类无关，
            # 对鬼影/涂抹鲁棒。同时，被多条射线穿过的"障碍"格是鬼影
            # （真实墙面不会被看到它后面的射线穿过），予以清除。
            ray_free, ray_votes = cls._raycast_free_cells(
                raw_ray[0], raw_ray[1], raw_ray[2], cam_centers_aligned,
                cam_frame_ids, qlo, res, free.shape, floor_z, u)
            free |= ray_free
            obstacle &= ~(ray_votes >= int(ray_clear_votes))
        observed = raster(np.ones(len(xy), dtype=bool), splat=1)
        grid = cls._finalize(
            free, obstacle, res, qlo, cam_centers_aligned,
            floor_z=floor_z, unit_per_m=u,
            robot_radius_m=robot_radius_m, res_m=res_m,
            observed=observed, ground_votes=free.astype(np.uint32),
            obstacle_votes=obs_votes.astype(np.uint32))
        grid.source_point_count = int(source_point_count)
        grid.retained_voxel_count = int(retained_voxel_count)
        grid.voxel_size_m = float(voxel_size_m)
        grid.min_voxel_views = int(min_voxel_views)
        return grid

    @classmethod
    def _finalize(cls, free, obstacle, res, qlo, cam_centers_aligned,
                  floor_z, unit_per_m, robot_radius_m=0.25, res_m=0.10,
                  observed=None, ground_votes=None, obstacle_votes=None,
                  max_seed_distance_m=0.75):
        """Inflate obstacles and select a connected geometric free component.

        Camera motion is rasterized into a separate ``traversed`` layer. It is
        never allowed to add free cells, erase obstacles, or mark geometry as
        observed. Connectivity filtering is applied only when the latest pose
        has nearby ground-supported free space; otherwise the raw geometric
        classification is retained and the missing seed is exposed in metadata.
        """
        free = np.asarray(free, dtype=bool).copy()
        raw_obstacle = np.asarray(obstacle, dtype=bool).copy()
        obstacle = raw_obstacle.copy()
        # 障碍物按机器人半径膨胀（3x3 最大滤波迭代）
        iters = max(int(math.ceil(robot_radius_m / res_m)), 1)
        obstacle = _binary_dilate(obstacle, iterations=iters)

        # 被膨胀障碍覆盖的自由格不再可行走
        free &= ~obstacle

        traversed = _trajectory_mask(
            free.shape, qlo, res, cam_centers_aligned,
            radius_cells=max(int(math.ceil(robot_radius_m / res_m)), 0))
        start = None
        seed = None
        if len(cam_centers_aligned):
            latest = np.asarray(cam_centers_aligned, dtype=np.float64)[-1, :2]
            cell = np.floor((latest - qlo) / res).astype(np.int64)
            start = (int(cell[0]), int(cell[1]))
            max_radius = max(
                int(math.ceil(max_seed_distance_m / max(res_m, 1e-9))), 0)
            seed = _nearest_mask_cell(free, start, max_radius)
            if seed is not None:
                free = _flood_component(free, seed)
        # free/obstacle 之外，被点云覆盖但因高度分类不确定的格子仍是
        # "已观测"，不能作为 frontier。障碍膨胀和 traversed 都不是新的
        # 3D 观测，因此不得扩大 geometry_observed。
        if observed is None or observed.shape != free.shape:
            observed = np.asarray(free | raw_obstacle, dtype=bool)
        else:
            observed = np.asarray(observed, dtype=bool).copy()
        grid = cls(
            res, qlo, free, obstacle, observed=observed,
            traversed=traversed, ground_votes=ground_votes,
            obstacle_votes=obstacle_votes, raw_obstacle=raw_obstacle)
        grid.floor_z = floor_z
        grid.unit_per_m = unit_per_m  # 调试用：1 米对应的地图单位数
        grid.start_cell = start
        grid.start_seed_cell = seed
        grid.start_seed_distance_cells = (
            None if start is None or seed is None else
            float(math.hypot(seed[0] - start[0], seed[1] - start[1])))
        grid.connectivity_filtered = seed is not None
        return grid

    @classmethod
    def from_frame_points(cls, frames, align_R, cam_height_m=1.5,
                          res_m=0.10, floor_band_m=0.12,
                          obs_low_m=0.15, obs_high_m=1.8,
                          robot_radius_m=0.25, margin_m=0.5,
                          unit_per_m=None, voxel_size_m=0.05,
                          min_voxel_views=3):
        """Merge unique VGGT frames and build one global-floor occupancy map.

        ``rows`` are deliberately ignored: after SLAM has produced a common 3D
        map, floor classification should depend on global height rather than on
        where a point appeared in an individual image.  Overlapping submaps may
        return the same physical frame repeatedly, so ``frame_id`` is deduplicated
        before point clouds and camera poses are merged.
        """
        unique_frames = {}
        for index, frame in enumerate(frames or []):
            fid = int(frame.get("frame_id", index))
            unique_frames[fid] = frame

        point_parts = []
        point_frame_parts = []
        camera_parts = []
        for fid in sorted(unique_frames):
            frame = unique_frames[fid]
            points = np.asarray(frame["points"], dtype=np.float64)
            finite = np.isfinite(points).all(axis=1)
            if finite.any():
                point_parts.append(points[finite] @ align_R.T)
                point_frame_parts.append(np.full(
                    int(finite.sum()), fid, dtype=np.int64))
            pose = np.asarray(frame["pose"], dtype=np.float64)
            if pose.shape == (4, 4) and np.isfinite(pose).all():
                camera_parts.append(pose[:3, 3] @ align_R.T)
        if not point_parts or not camera_parts:
            return None

        grid = cls.build(
            np.concatenate(point_parts), np.stack(camera_parts),
            res_m=res_m, floor_band_m=floor_band_m,
            obs_low_m=obs_low_m, obs_high_m=obs_high_m,
            robot_radius_m=robot_radius_m, margin_m=margin_m,
            cam_height_m=cam_height_m, unit_per_m=unit_per_m,
            point_frame_ids=np.concatenate(point_frame_parts),
            voxel_size_m=voxel_size_m,
            min_voxel_views=min_voxel_views,
            cam_frame_ids=sorted(unique_frames))
        if grid is not None:
            grid.floor_model = "global_height_peak"
            grid.source_frame_count = len(unique_frames)
        return grid

    @staticmethod
    def _raycast_free_cells(xy, z, fids, cam_centers, cam_fids, qlo, res,
                            shape, floor_z, u, max_rays_per_frame=400,
                            torso_low_m=0.25, torso_high_m=1.6):
        """相机→观测点射线经过的格子（躯干高度带内）标为自由。

        返回 (ray_free bool 栅格, ray_votes 射线穿过计数)。表面所在端点格
        不计入自由。按帧子采样射线数量以控制计算量；所有帧的射线合并后
        逐步进（全体射线向量化），步长 0.75 格防止对角跳格。
        """
        h, w = shape
        ray_free = np.zeros((h, w), dtype=bool)
        ray_votes = np.zeros((h, w), dtype=np.int32)
        if fids is None or cam_fids is None:
            return ray_free, ray_votes
        fids = np.asarray(fids)
        cam_map = {int(f): np.asarray(c, dtype=np.float64)
                   for f, c in zip(cam_fids, cam_centers)}
        rays_o, rays_d, rays_n, rays_oz, rays_dz = [], [], [], [], []
        for fid, cam in cam_map.items():
            idx = np.flatnonzero(fids == fid)
            if len(idx) == 0:
                continue
            if len(idx) > max_rays_per_frame:
                idx = idx[np.linspace(0, len(idx) - 1,
                                      max_rays_per_frame).astype(np.int64)]
            d = xy[idx] - cam[:2]
            dist = np.hypot(d[:, 0], d[:, 1])
            ok = dist > res
            d, dist, idx = d[ok], dist[ok], idx[ok]
            if len(idx) == 0:
                continue
            rays_o.append(np.tile(cam[:2], (len(idx), 1)))
            rays_d.append(d)
            rays_n.append(
                np.maximum((dist / (res * 0.75)).astype(np.int64), 1))
            rays_oz.append(np.full(len(idx), cam[2]))
            rays_dz.append(z[idx] - cam[2])
        if not rays_n:
            return ray_free, ray_votes
        O = np.concatenate(rays_o)
        D = np.concatenate(rays_d)
        N = np.concatenate(rays_n)
        OZ = np.concatenate(rays_oz)
        DZ = np.concatenate(rays_dz)
        lo = floor_z + torso_low_m * u
        hi = floor_z + torso_high_m * u
        for t in range(1, int(N.max())):
            active = t < N - 1          # 端点（表面）格不算自由
            if not active.any():
                break
            frac = t / N[active]
            pos = O[active] + D[active] * frac[:, None]
            zt = OZ[active] + DZ[active] * frac
            cell = np.floor((pos - qlo) / res).astype(np.int64)
            inb = (cell[:, 0] >= 0) & (cell[:, 0] < w) & \
                  (cell[:, 1] >= 0) & (cell[:, 1] < h)
            m = inb & (zt >= lo) & (zt <= hi)
            if m.any():
                cc = cell[m]
                ray_free[cc[:, 1], cc[:, 0]] = True
                np.add.at(ray_votes, (cc[:, 1], cc[:, 0]), 1)
        return ray_free, ray_votes

    @staticmethod
    def _classify_layers(xy, z, floor_z, cam_h, u,
                         floor_band_m, obs_low_m, obs_high_m,
                         tile_m=0.5, min_tile_points=20):
        """分块估计地板高度并判定 floor/obstacle 层。

        全局单一 floor_z 在回环涂抹或轻微倾斜下会漏掉大量地板点。
        这里按 ~0.5m 瓦片取局部 5% 高度分位数并做邻域中位数精修；
        候选明显高于"相机下方 1m"（瓦片内其实没有地面，被家具占据）
        或点数不足的瓦片回落到全局 floor_z。
        """
        band = floor_band_m * u
        tile = max(tile_m * u, 1e-6)
        t0 = np.percentile(xy, 1, axis=0)
        tidx = np.floor((xy - t0) / tile).astype(np.int64)
        _uniq, inv = np.unique(tidx, axis=0, return_inverse=True)
        ref = np.full(len(z), floor_z)
        for t in range(len(_uniq)):
            sel = inv == t
            if int(sel.sum()) < min_tile_points:
                continue
            tz = z[sel]
            # 以全局地板峰为锚，在 ±0.35m 窗口内取局部中位数——吸收
            # 倾斜/回环涂抹，同时排除全局峰下方的鬼影地板点。
            near = tz[np.abs(tz - floor_z) <= 0.35 * u]
            if len(near) >= max(min_tile_points // 2, 1):
                ref[sel] = float(np.median(near))
        is_floor = np.abs(z - ref) < band
        is_obs = (z > ref + obs_low_m * u) & (z < ref + obs_high_m * u)
        return is_floor, is_obs

    # ------------------------------------------------------------------
    def world_to_cell(self, p):
        cell = np.floor((np.asarray(p, dtype=np.float64)[:2] - self.origin)
                        / self.res).astype(np.int64)
        return int(cell[0]), int(cell[1])

    def cell_to_world(self, cell):
        return self.origin + (np.asarray(cell, dtype=np.float64) + 0.5) * self.res

    def in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.free.shape[1] and 0 <= y < self.free.shape[0]

    def traversable(self, cell):
        x, y = cell
        return self.in_bounds(cell) and self.free[y, x] and not self.obstacle[y, x]

    def nearest_traversable(self, cell, max_radius=20):
        """螺旋搜索最近可行走格。"""
        if self.traversable(cell):
            return cell
        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    c = (cell[0] + dx, cell[1] + dy)
                    if self.traversable(c):
                        return c
        return None

    # ------------------------------------------------------------------
    def astar(self, start_xy, goal_xy, snap_radius=20):
        """A* 寻路。起终点先吸附到最近可行走格。返回世界坐标路径或 None。"""
        start = self.nearest_traversable(self.world_to_cell(start_xy), snap_radius)
        goal = self.nearest_traversable(self.world_to_cell(goal_xy), snap_radius)
        if start is None or goal is None:
            return None

        dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0),
                (-1, 1), (0, 1), (1, 1)]
        openq = [(0.0, start)]
        came = {start: None}
        g = {start: 0.0}
        H, W = self.free.shape
        while openq:
            _, cur = heapq.heappop(openq)
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(self.cell_to_world(cur))
                    cur = came[cur]
                return path[::-1]
            for dx, dy in dirs:
                nxt = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nxt[0] < W and 0 <= nxt[1] < H):
                    continue
                if not self.traversable(nxt):
                    continue
                if dx and dy:  # 禁止切角
                    if not (self.traversable((cur[0] + dx, cur[1])) and
                            self.traversable((cur[0], cur[1] + dy))):
                        continue
                ng = g[cur] + math.hypot(dx, dy)
                if ng < g.get(nxt, math.inf):
                    g[nxt] = ng
                    came[nxt] = cur
                    h = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                    heapq.heappush(openq, (ng + h, nxt))
        return None

    def line_of_sight(self, a, b):
        """两世界坐标点间是否无遮挡（Bresenham 走查可行走格）。"""
        ca, cb = self.world_to_cell(a), self.world_to_cell(b)
        x0, y0 = ca
        x1, y1 = cb
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if not self.traversable((x0, y0)):
                return False
            if (x0, y0) == (x1, y1):
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def shortcut(self, path):
        """视线捷径简化，减少跟随拐点。"""
        if not path or len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.line_of_sight(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out


class PathFollower:
    """路径跟随 + 航位推算。位姿锚定最新关键帧，动作间隙外推。"""

    def __init__(self, scale, reach_m=0.8, waypoint_m=0.3):
        self.scale = scale                # 米/地图单位（可随标定更新）
        self.reach_m = reach_m            # 判定到达的距离（米）
        self.waypoint_m = waypoint_m      # 中间航点丢弃半径（米）
        self.path = None                  # 世界坐标(地图单位)路径
        self.anchor_frame = -1            # 锚点关键帧 frame_id
        self.x = self.y = self.yaw = 0.0  # 当前估计（对齐坐标系）

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"path follower scale must be positive: {value!r}")
        self._scale = value

    @property
    def reach(self):
        return self.reach_m / self.scale

    @property
    def wp_dist(self):
        return self.waypoint_m / self.scale

    def set_path(self, path):
        self.path = list(path) if path else None

    def update_anchor(self, pose, R, frame_id):
        """用最新关键帧位姿重置锚点（清除推算累积误差）。"""
        self.x, self.y, self.yaw = pose_to_yaw_2d(pose, R)
        self.anchor_frame = int(frame_id)

    def dead_reckon(self, action):
        """锚点之后每执行一个动作调用一次。"""
        if action == TURN_LEFT:
            self.yaw += TURN_STEP_RAD
        elif action == TURN_RIGHT:
            self.yaw -= TURN_STEP_RAD
        elif action == MOVE_FORWARD:
            d = FORWARD_STEP_M / self.scale
            self.x += math.cos(self.yaw) * d
            self.y += math.sin(self.yaw) * d

    def undo_dead_reckon(self, action):
        """撤销被 RGB 运动检测判定为未执行成功的上一动作。"""
        if action == TURN_LEFT:
            self.yaw -= TURN_STEP_RAD
        elif action == TURN_RIGHT:
            self.yaw += TURN_STEP_RAD
        elif action == MOVE_FORWARD:
            d = FORWARD_STEP_M / self.scale
            self.x -= math.cos(self.yaw) * d
            self.y -= math.sin(self.yaw) * d

    def distance_to_goal(self):
        if not self.path:
            return math.inf
        gx, gy = self.path[-1][:2]
        return math.hypot(gx - self.x, gy - self.y)

    def next_action(self):
        """返回 (action_id, arrived)。路径为空返回 (None, False)。"""
        if not self.path:
            return None, False
        # 丢掉已经走到的中间航点
        while len(self.path) > 1 and \
                math.hypot(self.path[0][0] - self.x,
                           self.path[0][1] - self.y) < self.wp_dist:
            self.path.pop(0)
        gx, gy = self.path[-1][:2]
        if math.hypot(gx - self.x, gy - self.y) < self.reach:
            return None, True
        tx, ty = self.path[0][:2]
        target_yaw = math.atan2(ty - self.y, tx - self.x)
        err = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) > TURN_STEP_RAD / 2:
            return (TURN_LEFT if err > 0 else TURN_RIGHT), False
        return MOVE_FORWARD, False
