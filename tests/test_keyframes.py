"""关键帧策略的纯 Python 回归测试，不依赖 CUDA 或 VGGT 权重。"""

import pytest

from mapping.keyframes import (
    SUPPORTED_SUBMAP_OVERLAP,
    AdaptiveKeyframeSelector,
    pop_submap_window,
    validate_supported_overlap,
)


class FakeTracker:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.initialized_with = []

    def compute_disparity(self, image, min_disparity):
        assert min_disparity == 40.0
        return next(self.decisions)

    def initialize_keyframe(self, image):
        self.initialized_with.append(image)


def test_flow_selection_resets_interval():
    tracker = FakeTracker([True, False, True, False])
    selector = AdaptiveKeyframeSelector(tracker, max_interval=3)

    assert selector.select("frame-1") == (True, "flow")
    assert selector.select("frame-2") == (False, "below_threshold")
    assert selector.select("frame-3") == (True, "flow")
    assert selector.select("frame-4") == (False, "below_threshold")
    assert selector.frames_since_keyframe == 1
    assert tracker.initialized_with == []


def test_max_interval_forces_and_refreshes_reference_frame():
    tracker = FakeTracker([False, False, False, False])
    selector = AdaptiveKeyframeSelector(tracker, max_interval=3)

    assert selector.select("frame-1")[0] is False
    assert selector.select("frame-2")[0] is False
    assert selector.select("frame-3") == (True, "max_interval")
    assert tracker.initialized_with == ["frame-3"]
    assert selector.num_forced == 1
    assert selector.select("frame-4")[0] is False


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_max_interval_is_rejected(value):
    with pytest.raises(ValueError):
        AdaptiveKeyframeSelector(FakeTracker([]), max_interval=value)


def test_submap_window_preserves_overlap_and_pending_frames():
    submap, pending = pop_submap_window(list(range(25)), 19, 3)

    assert submap == list(range(19))
    assert pending == [16, 17, 18, 19, 20, 21, 22, 23, 24]


def test_submap_window_never_accepts_zero_overlap():
    with pytest.raises(ValueError):
        pop_submap_window(list(range(19)), 19, 0)


def test_supported_overlap_reuses_the_previous_last_frame():
    """The stock add_edge bridge is valid only for this exact identity."""
    assert SUPPORTED_SUBMAP_OVERLAP == 1
    target_size = 16 + SUPPORTED_SUBMAP_OVERLAP
    first, pending = pop_submap_window(
        list(range(40)), target_size, SUPPORTED_SUBMAP_OVERLAP)
    second, _pending = pop_submap_window(
        pending, target_size, SUPPORTED_SUBMAP_OVERLAP)

    assert first[-1] == second[0]


def test_unsupported_multi_frame_overlap_is_rejected():
    with pytest.raises(ValueError, match="add_edge"):
        validate_supported_overlap(3)
