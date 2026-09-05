"""单目地图的在线尺度标定（client 侧，纯 numpy）。

VGGT-SLAM 输出的位姿/点云只有 Sim(3) 意义下的相对尺度。
本模块利用 benchmark 已知的离散动作步长（MOVE_FORWARD = 0.25m）
在线回归"地图单位 -> 米"的尺度因子：

    样本: 相邻关键帧间不含转向的纯直行片段及地图位移 ‖Δp‖
    比率: r = 0.25 * n_fwd / ‖Δp‖   （即尺度的一个观测）
    估计: 滑动窗口内的中位数，MAD 剔除异常值

撞墙卡住时真位移≈0 而 n_fwd>0，比率会异常偏大：调用方（mapping_agent）
会把视觉确认卡住的步回溯标记为 -1（no-op），本模块再把前进数少于
min_fwd 的段剔除（转向漂移主导），剩余异常由中位数和 MAD 抑制。
回环优化会全局改写历史位姿、子图间尺度也会缓慢漂移，因此使用
滑动窗口跟踪尺度，而不是全局累积平均。该估计只作诊断；正式导航
使用多帧地面到相机高度的归一化尺规。

用法::

    cal = ScaleCalibrator()
    cal.record_action(action_id)          # 每步记录执行的动作
    scale = cal.update(poses, frame_ids)  # 定期用最新位姿更新
    if scale is not None:
        metric_pos = scale * slam_pos
"""

import numpy as np

MOVE_FORWARD = 1


class ScaleCalibrator:
    def __init__(self, forward_step=0.25, window=50, min_samples=5,
                 mad_k=3.0, min_fwd=2, plausible_min=0.3,
                 plausible_max=3.0):
        self.forward_step = forward_step
        self.window = window
        self.min_samples = min_samples
        self.mad_k = mad_k
        self.min_fwd = min_fwd
        self.plausible_min = plausible_min
        self.plausible_max = plausible_max
        self.actions = []        # 每步执行的动作 id
        self.scale_history = []  # (num_actions_seen, scale) 调试记录
        self._starve = 0         # 连续"门控后样本不足"次数（坏 ref 自愈）

    def reset(self):
        self.actions = []
        self.scale_history = []
        self._starve = 0

    def record_action(self, action_id):
        self.actions.append(int(action_id))

    def update(self, poses, frame_ids):
        """用最新关键帧位姿更新尺度估计。

        Args:
            poses: (N, 4, 4) cam2world（相对尺度），与 frame_ids 对齐
            frame_ids: 长度为 N 的关键帧全局帧号（1 起始，与 step+1 对应）

        Returns:
            当前尺度（米/地图单位），样本不足返回 None。
        """
        if poses is None or len(poses) < 2:
            return self.current_scale()

        order = np.argsort(frame_ids)
        frame_ids = np.asarray(frame_ids)[order]
        positions = np.asarray(poses)[order][:, 0:3, 3]

        ref = self.current_scale()  # 参考尺度，用于门控异常样本
        if self._starve >= 3:
            # 连续多轮门控后样本不足：ref 本身可能已坏（坏种子自我锁死），
            # 本轮放开门控重新播种
            print(f"[ScaleCalibrator] 门控饥饿，放开门控重新播种 "
                  f"(旧 ref={ref})")
            ref = None
            self._starve = 0
        ratios = []
        n_reject = 0
        for i in range(1, len(positions)):
            f_a, f_b = int(frame_ids[i - 1]), int(frame_ids[i])
            # 帧 f 的位姿对应 step f-1 的观测；f_a 到 f_b 的位移由
            # step f_a-1 .. f_b-2 的动作产生，即 actions[f_a-1 : f_b-1]
            lo, hi = f_a - 1, f_b - 1
            if lo < 0 or hi > len(self.actions) or hi <= lo:
                continue
            segment = self.actions[lo:hi]
            # 端点距离只能代表净位移。若片段中包含转向，累计前进路程
            # 可能远大于端点弦长（折返时甚至接近 0），用 n_fwd*0.25
            # 作分子会系统性放大尺度。动作尺度现在只保留纯直行片段；
            # -1 是调用方确认的碰撞/no-op，可以安全忽略。
            if any(a not in (MOVE_FORWARD, -1) for a in segment):
                continue
            n_fwd = sum(1 for a in segment if a == MOVE_FORWARD)
            if n_fwd < self.min_fwd:
                # 前进太少的段由转向时的 SLAM 漂移主导（漂移也产生位移），
                # 指令位移占不到主导，比率噪声极大，不作样本。
                continue
            dist = float(np.linalg.norm(positions[i] - positions[i - 1]))
            if dist < 1e-6:
                continue  # 卡住段，无信息（避免除零）
            ratio = self.forward_step * n_fwd / dist
            # 物理量程必须在自指 ref 门控之前执行。否则首个碰撞片段可能
            # 产生 10x 坏种子，后续门控会围绕坏种子自洽并永久锁死。
            if not self.plausible_min <= ratio <= self.plausible_max:
                n_reject += 1
                continue
            if ref is not None:
                # 门控：实际位移应与指令位移（0.25*n_fwd 换算成地图单位）
                # 大致一致。撞墙卡住时 dist 远小于指令位移，比率会爆炸；
                # 回环跳变/跟踪丢失时 dist 异常大。两者都会污染中位数。
                expected = self.forward_step * n_fwd / ref
                if dist < 0.4 * expected or dist > 2.5 * expected:
                    n_reject += 1
                    continue
            ratios.append(ratio)

        if len(ratios) < self.min_samples:
            if ref is not None:
                self._starve += 1
            return self.current_scale()
        self._starve = 0

        # 滑动窗口 + 中位数 + MAD 异常值剔除
        r = np.asarray(ratios[-self.window:])
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        if mad > 1e-9:
            keep = np.abs(r - med) <= self.mad_k * 1.4826 * mad
            if keep.sum() >= self.min_samples:
                med = float(np.median(r[keep]))

        self.scale_history.append((len(self.actions), med))
        if n_reject:
            ref_text = "None" if ref is None else f"{ref:.3f}"
            print(f"[ScaleCalibrator] 门控剔除 {n_reject} 个异常段 "
                  f"(ref={ref_text})")
        return med

    def current_scale(self):
        """返回最近一次有效估计，尚无估计返回 None。"""
        if self.scale_history:
            return self.scale_history[-1][1]
        return None

    def to_metric(self, points):
        """把地图坐标（点或位姿位置）换算成米。无估计时原样返回。"""
        s = self.current_scale()
        if s is None:
            return np.asarray(points)
        return s * np.asarray(points)
