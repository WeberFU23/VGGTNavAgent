"""导航执行模块（client 侧，纯 numpy，不依赖 habitat/torch）。

把 VGGT-SLAM 的相对尺度地图变成可执行的离散动作序列：

1. 重力对齐：相机轨迹最小主成分估计竖直轴，得到 z'=up 的 2D 规划平面。
2. 占据栅格：点云按高度分层——近地面层为可行走面，腰部高度层为障碍；
   相机中心投影强制标记为自由（机器人确实走过），障碍物按机器人半径膨胀。
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


def gravity_alignment(poses):
    """用 SLAM 相机旋转的稳健平均 up 估计重力方向。

    R @ p 把点旋转到 z'=up 的对齐坐标系（行向量为新坐标轴）。
    相比轨迹位置 PCA，该方法在直走走廊、短轨迹和近退化轨迹下稳定。
    """
    poses = np.asarray(poses, dtype=np.float64)
    ups = -poses[:, :3, 1]  # OpenCV 相机 y 轴朝下
    reference = ups[0]
    ups = np.asarray([u if np.dot(u, reference) >= 0 else -u for u in ups])
    up = np.median(ups, axis=0)
    up /= np.linalg.norm(up) + 1e-9
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


class OccupancyGrid:
    """2D 占据栅格（对齐坐标系），单位与输入点云一致（地图单位）。"""

    def __init__(self, res, origin, free, obstacle):
        self.res = res                  # 地图单位/格
        self.origin = np.asarray(origin, dtype=np.float64)  # 格子(0,0)的xy
        self.free = free                # (H,W) bool 可行走
        self.obstacle = obstacle        # (H,W) bool 膨胀后障碍

    @classmethod
    def from_trajectory(cls, cam_centers_aligned, stamp=3):
        """只从相机轨迹构建可行走网络（"面包屑"导航）。

        语义层定位的目标必然在机器人曾经到达过的地方附近（它就是从
        历史关键帧里检出的），因此沿走过的路线规划既不需要地板/障碍
        高度分层（实测在稀疏点云+尺度误差下极不可靠），也天然避障
        ——这条路机器人真的走过。栅格分辨率用关键帧间距自适应，
        同样不依赖在线尺度标定。
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

        cc = np.floor((xy - qlo) / res).astype(np.int64)
        cells = []
        for i in range(len(cc)):
            x0, y0 = int(cc[i][0]), int(cc[i][1])
            if i > 0:  # 与上一相机 Bresenham 连线，覆盖大步长间隙
                x1, y1 = int(cc[i - 1][0]), int(cc[i - 1][1])
                dx, dy = abs(x1 - x0), abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy
                while (x0, y0) != (x1, y1):
                    cells.append((x0, y0))
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x0 += sx
                    if e2 < dx:
                        err += dx
                        y0 += sy
            cells.append((x0, y0))
        for cx, cy in cells:
            for dx in range(-stamp, stamp + 1):
                for dy in range(-stamp, stamp + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        free[ny, nx] = True
        grid = cls(res, qlo, free, np.zeros_like(free))
        grid.floor_z = None
        grid.unit_per_m = 1.0 / (spacing * 1.5) if spacing else 0.0  # 调试估算
        return grid

    @classmethod
    def build(cls, points_aligned, cam_centers_aligned,
              res_m=0.10, floor_band_m=0.12, obs_low_m=0.15, obs_high_m=1.8,
              robot_radius_m=0.25, margin_m=0.5, cam_height_m=1.5):
        """从对齐点云构建栅格。

        所有米制参数都通过"尺规"换算成地图单位：尺规 = 相机中位高度 -
        地面高度（habitat 中相机离地约 1.5m，是地图内生的参照，不受
        在线尺度标定误差影响——实测标定误差可达 40%+，直接用它会
        同时压窄地板带宽、戳破溅射连通性，导致自由空间消失）。
        """
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
        ruler_est = max(cam_h - z_min, 1e-6)
        lo, hi = np.percentile(z, [1, 99])
        nbins = max(int((hi - lo) / (0.033 * ruler_est)), 10)
        hist, edges = np.histogram(z, bins=nbins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        below = centers < cam_h - 0.15 * ruler_est
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
        floor_z = float(centers[strong[0] if strong else
                                int(np.argmax(h_smooth[below]))])

        # 尺规：相机离地高度（地图单位）≈ cam_height_m 米
        ruler = max(cam_h - floor_z, 1e-6)
        u = ruler / cam_height_m          # 1 米对应的地图单位数
        res = res_m * u

        # 分层
        is_floor = np.abs(z - floor_z) < floor_band_m * u
        is_obs = (z > floor_z + obs_low_m * u) & \
                 (z < floor_z + obs_high_m * u)

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

        free = raster(is_floor, splat=1)
        obstacle = raster(is_obs, splat=0)

        # 障碍物按机器人半径膨胀（3x3 最大滤波迭代）
        iters = max(int(math.ceil(robot_radius_m / res_m)), 1)
        inflated = obstacle.copy()
        for _ in range(iters):
            grown = inflated.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    grown |= np.roll(np.roll(inflated, dy, axis=0), dx, axis=1)
            # np.roll 会回绕，把回绕带清掉（边界外本来就视为障碍，无影响）
            inflated = grown
        obstacle = inflated

        # 被膨胀障碍覆盖的自由格不再可行走
        free &= ~obstacle

        # 相机轨迹走廊 = 机器人走过的路线，最后强制自由并清障
        # （必须在膨胀之后，否则膨胀会把走廊重新盖掉）。
        # 地板点稀疏/有洞时自由空间会碎裂，走廊保证起点沿走过的
        # 路线到目标附近的连通性（目标总是曾入镜的位置）。
        if len(cam_centers_aligned):
            cc = np.floor((cam_centers_aligned[:, :2] - qlo) / res).astype(np.int64)
            cells = []
            for i in range(len(cc)):
                x0, y0 = int(cc[i][0]), int(cc[i][1])
                if i > 0:  # 与上一相机连线（Bresenham），覆盖大步长间隙
                    x1, y1 = int(cc[i - 1][0]), int(cc[i - 1][1])
                    dx, dy = abs(x1 - x0), abs(y1 - y0)
                    sx = 1 if x0 < x1 else -1
                    sy = 1 if y0 < y1 else -1
                    err = dx - dy
                    while (x0, y0) != (x1, y1):
                        cells.append((x0, y0))
                        e2 = 2 * err
                        if e2 > -dy:
                            err -= dy
                            x0 += sx
                        if e2 < dx:
                            err += dx
                            y0 += sy
                cells.append((int(cc[i][0]), int(cc[i][1])))
            for cx, cy in cells:
                for dx in (-2, -1, 0, 1, 2):
                    for dy in (-2, -1, 0, 1, 2):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            free[ny, nx] = True
                            obstacle[ny, nx] = False
        grid = cls(res, qlo, free, obstacle)
        grid.floor_z = floor_z
        grid.unit_per_m = u  # 调试用：1 米对应的地图单位数（尺规导出）
        return grid

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
