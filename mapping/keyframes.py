"""关键帧选择策略。

VGGT-SLAM 自带的 FrameTracker 根据相对上一关键帧的光流视差取帧。
弱纹理、长走廊或近似沿光轴前进时，二维平均视差可能长期低于阈值；
本模块在保留该策略的同时增加最大观测间隔，避免相邻关键帧相隔过远。
"""


# Upstream Solver.add_edge() aligns current[0] with previous[-1].  Supporting
# a wider overlap requires changing the graph constraints, not only retaining
# more paths in the window.  Keep the production server on the one-frame mode
# that upstream explicitly supports until that multi-frame alignment exists.
SUPPORTED_SUBMAP_OVERLAP = 1


def validate_supported_overlap(overlap):
    """Reject graph layouts that stock ``Solver.add_edge`` cannot align."""
    overlap = int(overlap)
    if overlap != SUPPORTED_SUBMAP_OVERLAP:
        raise ValueError(
            "当前 VGGT-SLAM add_edge 仅支持 "
            f"overlapping-window-size={SUPPORTED_SUBMAP_OVERLAP}；"
            "更大的 overlap 需要先实现对应帧的多因子约束")
    return overlap


class AdaptiveKeyframeSelector:
    """组合光流触发与最大间隔约束，并维护可诊断的选择状态。"""

    def __init__(self, tracker, min_disparity=40.0, max_interval=3):
        if min_disparity <= 0:
            raise ValueError("min_disparity 必须为正数")
        if max_interval < 1:
            raise ValueError("max_interval 必须至少为 1")
        self.tracker = tracker
        self.min_disparity = float(min_disparity)
        self.max_interval = int(max_interval)
        self.frames_since_keyframe = 0
        self.num_forced = 0

    def select(self, image):
        """返回 ``(is_keyframe, reason)``。

        tracker 在光流触发时会自行更新参考帧；最大间隔触发时必须显式
        更新参考帧，否则后续光流仍会相对过旧图像计算。
        """
        selected = bool(self.tracker.compute_disparity(
            image, self.min_disparity))
        if selected:
            self.frames_since_keyframe = 0
            return True, "flow"

        self.frames_since_keyframe += 1
        if self.frames_since_keyframe >= self.max_interval:
            self.tracker.initialize_keyframe(image)
            self.frames_since_keyframe = 0
            self.num_forced += 1
            return True, "max_interval"
        return False, "below_threshold"


def pop_submap_window(paths, target_size, overlap):
    """顺序取出一个固定大小子图，并把桥接帧放回待处理队列。

    返回 ``(submap, pending)``。pending 以当前子图最后 ``overlap`` 帧
    开头，后接尚未处理的新帧，因此不会因后端繁忙而丢失时间连续性。
    """
    if not 1 <= overlap < target_size:
        raise ValueError("overlap 必须在 [1, target_size) 范围内")
    if len(paths) < target_size:
        raise ValueError("关键帧数量不足一个完整子图")
    submap = list(paths[:target_size])
    pending = submap[-overlap:] + list(paths[target_size:])
    return submap, pending
