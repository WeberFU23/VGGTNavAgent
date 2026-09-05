"""ScaleCalibrator 样本过滤：min_fwd 剔除漂移主导段、-1 no-op 不计前进。

注意索引约定：帧 f 的位姿对应 step f-1 的观测，帧 f_a 到 f_b 的位移
由 actions[f_a-1 : f_b-1] 产生。
"""
import numpy as np

from mapping.scale_calibration import MOVE_FORWARD, ScaleCalibrator


def _poses_from_positions(positions):
    poses = []
    for x in positions:
        p = np.eye(4)
        p[0, 3] = x
        poses.append(p)
    return poses


def test_min_fwd_filters_turn_drift_segments():
    """每段只有 1 次前进时（转向漂移主导）不产生尺度样本。"""
    cal = ScaleCalibrator(min_samples=1, min_fwd=2)
    TURN_LEFT = 2
    # 帧 1->3、3->5 两段，各 1 前进 + 3 转向，段位移 0.25 地图单位
    for a in [MOVE_FORWARD, TURN_LEFT, TURN_LEFT, TURN_LEFT,
              MOVE_FORWARD, TURN_LEFT, TURN_LEFT, TURN_LEFT]:
        cal.record_action(a)
    poses = _poses_from_positions([0.0, 0.25, 0.5])
    assert cal.update(poses, [1, 3, 5]) is None


def test_min_fwd_accepts_multi_forward_segments():
    """每段 >=min_fwd 次前进时正常产出尺度。"""
    cal = ScaleCalibrator(min_samples=1, min_fwd=2)
    # 帧 1->3、3->5 两段各 2 次前进；真实尺度 2.0 m/unit
    # （0.25*2 米对应 0.25 地图单位）
    for _ in range(4):
        cal.record_action(MOVE_FORWARD)
    poses = _poses_from_positions([0.0, 0.25, 0.5])
    scale = cal.update(poses, [1, 3, 5])
    assert scale is not None
    assert abs(scale - 2.0) < 1e-6


def test_noop_marked_actions_do_not_count_as_forward():
    """碰撞被回溯标记为 -1 的步不计入 n_fwd。"""
    cal = ScaleCalibrator(min_samples=1, min_fwd=2)
    # 帧 1->4、4->7 两段，各 3 个动作但其中 2 个被标记为 -1（碰撞），
    # 只剩 1 次有效前进 → 不足 min_fwd，无样本
    for _ in range(2):
        cal.record_action(MOVE_FORWARD)
        cal.record_action(-1)
        cal.record_action(-1)
    poses = _poses_from_positions([0.0, 0.25, 0.5])
    assert cal.update(poses, [1, 4, 7]) is None


def test_turning_segment_is_not_used_as_endpoint_scale_sample():
    """累计路程不能除以含转向片段的端点弦长。"""
    cal = ScaleCalibrator(min_samples=1, min_fwd=2)
    TURN_LEFT = 2
    for action in [MOVE_FORWARD, TURN_LEFT, MOVE_FORWARD]:
        cal.record_action(action)
    poses = _poses_from_positions([0.0, 0.05])
    assert cal.update(poses, [1, 4]) is None


def test_out_of_range_ratio_cannot_seed_calibrator():
    """碰撞漏检造成的极小 SLAM 位移不能成为自锁参考值。"""
    cal = ScaleCalibrator(min_samples=1, min_fwd=2,
                          plausible_min=0.3, plausible_max=3.0)
    cal.record_action(MOVE_FORWARD)
    cal.record_action(MOVE_FORWARD)
    poses = _poses_from_positions([0.0, 0.04])  # 0.5 / 0.04 = 12.5
    assert cal.update(poses, [1, 3]) is None
    assert cal.current_scale() is None
