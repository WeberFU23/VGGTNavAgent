"""导航执行模块（client 侧，纯 numpy，不依赖 habitat/torch）。

把 VGGT-SLAM 的相对尺度地图变成可执行的离散动作序列：

1. 重力对齐：相机旋转（经安装俯仰角补偿）的稳健中位数估计竖直轴，
   得到 z'=up 的 2D 规划平面。
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


# 相机安装下俯角。配置标称 30°（hm3d_config.yaml orientation [-pi/6,0,0]），
# 但实测（scripts/check_gravity.py，238 关键帧）散布最小值在 40°——
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


class OccupancyGrid:
    """2D 占据栅格（对齐坐标系），单位与输入点云一致（地图单位）。"""

    def __init__(self, res, origin, free, obstacle, observed=None):
        self.res = res                  # 地图单位/格
        self.origin = np.asarray(origin, dtype=np.float64)  # 格子(0,0)的xy
        self.free = np.asarray(free, dtype=bool)       # (H,W) bool 可行走
        self.obstacle = np.asarray(obstacle, dtype=bool)  # 膨胀后障碍
        # 显式观测覆盖层。未提供时保持旧构造器语义，便于合成测试和调用方。
        self.observed = np.asarray(
            self.free | self.obstacle if observed is None else observed,
            dtype=bool)

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
        candidates = strong if strong else [int(np.argmax(h_smooth[below]))]
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
        observed = raster(np.ones(len(xy), dtype=bool), splat=1)
        return cls._finalize(free, obstacle, res, qlo, cam_centers_aligned,
                             floor_z=floor_z, unit_per_m=u,
                             robot_radius_m=robot_radius_m, res_m=res_m,
                             observed=observed)

    @classmethod
    def _finalize(cls, free, obstacle, res, qlo, cam_centers_aligned,
                  floor_z, unit_per_m, robot_radius_m=0.25, res_m=0.10,
                  observed=None):
        """公共收尾：障碍膨胀 -> 走廊覆盖 -> 连通域约束 -> 建栅格。"""
        h, w = free.shape
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
            # 连通域约束：只有与机器人当前位置连通的自由空间才可行走。
            # 镜面/玻璃的假深度会在墙另一侧造出"鬼自由空间"，
            # 不连通的区域一律剔除（探索时也不会把它当目标）。
            start = (int(cc[-1][0]), int(cc[-1][1]))
            free = _flood_component(free, start)
        # free/obstacle 之外，被点云覆盖但因高度分类不确定的格子仍是
        # "已观测"，不能作为 frontier。轨迹走廊也明确属于已观测区域。
        if observed is None or observed.shape != free.shape:
            observed = free | obstacle
        observed = np.asarray(observed, dtype=bool) | free | obstacle
        grid = cls(res, qlo, free, obstacle, observed=observed)
        grid.floor_z = floor_z
        grid.unit_per_m = unit_per_m  # 调试用：1 米对应的地图单位数
        return grid

    @classmethod
    def from_frame_points(cls, frames, align_R, cam_height_m=1.5,
                          res_m=0.10, bottom_frac=0.25, min_floor_pts=50,
                          ground_band=0.2, obs_low=0.2, obs_high=1.2,
                          min_ground_votes=1, obs_votes=2,
                          robot_radius_m=0.25, margin_m=0.5):
        """逐帧局部地板锚定的投票式自由空间（对抗子图间漂移/分层污损）。

        全局点云在子图边界处有重影，全局高度直方图会被抹平（实测地板
        峰消失、桌床峰胜出）。改为逐帧处理：相机下俯 40°，图像底部
        bottom_frac 行的点即脚下地板，取其中位高度为该帧局部地板，
        尺规 = 相机高 - 局部地板（≈1.5m，构造上尺度无关）；再以
        尺规的相对带宽投地面票/障碍票到公共栅格。每帧几何内部一致，
        子图错位只模糊格子边界而不破坏分层。

        frames: iterable of dict，含 keys:
            points (N,3) 世界系（NaN 表示无效）, rows (N,) 原始图像行号,
            pose (4,4) cam2world。
        """
        per_frame = []
        cam_centers = []
        rulers = []
        for fr in frames:
            pts = np.asarray(fr["points"], dtype=np.float64)
            rows = np.asarray(fr["rows"]).ravel()
            finite = np.isfinite(pts).all(axis=1)
            pts, rows = pts[finite], rows[finite]
            if len(pts) < min_floor_pts:
                continue
            pa = pts @ align_R.T
            cam = np.asarray(fr["pose"], dtype=np.float64)[:3, 3] @ align_R.T
            img_h = rows.max() + 1
            bottom = rows >= img_h * (1.0 - bottom_frac)
            floor_sel = bottom & (pa[:, 2] < cam[2])
            if floor_sel.sum() < min_floor_pts:
                continue
            floor_z = float(np.median(pa[floor_sel, 2]))
            ruler = cam[2] - floor_z
            if ruler <= 1e-6:
                continue
            z = pa[:, 2]
            ground = np.abs(z - floor_z) < ground_band * ruler
            obs = (z > floor_z + obs_low * ruler) & \
                  (z < floor_z + obs_high * ruler)
            per_frame.append((pa[:, :2], ground, obs))
            cam_centers.append(cam)
            rulers.append(ruler)
        if not per_frame:
            return None
        cam_centers = np.stack(cam_centers)
        ruler_med = float(np.median(rulers))
        u = ruler_med / cam_height_m
        res = res_m * u

        all_xy = np.concatenate([pf[0] for pf in per_frame])
        qlo = np.percentile(all_xy, 0.5, axis=0) - margin_m * u
        qhi = np.percentile(all_xy, 99.5, axis=0) + margin_m * u
        qlo = np.minimum(qlo, cam_centers[:, :2].min(0) - res)
        qhi = np.maximum(qhi, cam_centers[:, :2].max(0) + res)
        w = max(int(np.ceil((qhi[0] - qlo[0]) / res)), 8)
        h = max(int(np.ceil((qhi[1] - qlo[1]) / res)), 8)

        gv = np.zeros((h, w), dtype=np.int32)
        ov = np.zeros((h, w), dtype=np.int32)
        observed = np.zeros((h, w), dtype=bool)
        for xy2, ground, obs in per_frame:
            # 任意有效 VGGT 3D 回投影均贡献观测覆盖；一格膨胀填补稀疏
            # 点采样的小孔，但不会把视野外区域误标为已观测。
            cell_all = np.floor((xy2 - qlo) / res).astype(np.int64)
            ok_all = (cell_all[:, 0] >= 0) & (cell_all[:, 0] < w) & \
                     (cell_all[:, 1] >= 0) & (cell_all[:, 1] < h)
            observed[cell_all[ok_all, 1], cell_all[ok_all, 0]] = True
            for mask, acc in ((ground, gv), (obs, ov)):
                sel = xy2[mask]
                if not len(sel):
                    continue
                cell = np.floor((sel - qlo) / res).astype(np.int64)
                ok = (cell[:, 0] >= 0) & (cell[:, 0] < w) & \
                     (cell[:, 1] >= 0) & (cell[:, 1] < h)
                np.add.at(acc, (cell[ok, 1], cell[ok, 0]), 1)

        free = (gv >= min_ground_votes) & (ov < obs_votes)
        # 地面点间距 > 格距时自由空间碎裂，膨胀一格桥接（随后障碍会盖回）
        grown = free.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= np.roll(np.roll(free, dy, axis=0), dx, axis=1)
        free = grown
        # 3x3 闭运算：填补稀疏采样小孔，同时基本保持真实观测外边界；
        # 不能只做膨胀，否则 free 与 unknown 会被隔开而丢失 frontier。
        padded = np.pad(observed, 1, constant_values=False)
        dilated = observed.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dilated |= padded[1 + dy:1 + dy + h,
                                   1 + dx:1 + dx + w]
        padded_d = np.pad(dilated, 1, constant_values=False)
        covered = np.ones_like(observed)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                covered &= padded_d[1 + dy:1 + dy + h,
                                    1 + dx:1 + dx + w]
        obstacle = ov >= obs_votes
        floor_z_med = float(np.median(
            [c[2] for c in cam_centers])) - ruler_med
        return cls._finalize(free, obstacle, res, qlo, cam_centers,
                             floor_z=floor_z_med, unit_per_m=u,
                             robot_radius_m=robot_radius_m, res_m=res_m,
                             observed=covered)

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
