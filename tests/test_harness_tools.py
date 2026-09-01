"""harness 化新增能力测试：决策工具、动作流水、白名单放宽、get_captions。

只依赖 numpy + mock client，不需要建图服务端或真实 VLM。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_api import Action
from agents import navigator as nav
from agents.decision_state import build_world_state
from agents.nav_agent import NavAgent
from decision import DecisionLoop, DecisionResult
from mapping.caption_store import CaptionStore
from mapping.client import MappingClient


def _make_agent():
    agent = NavAgent()
    agent.target_text = "basket"
    agent._target_mode = "all"
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    poses = np.stack([np.eye(4)] * 3)
    agent.client = SimpleNamespace(get_all_poses=lambda: (poses, [0, 1, 2]))
    return agent


def _state(mode="all", found=1, expected=None):
    return {
        "task": {"goal": "Find all baskets", "mode": mode,
                 "found": found, "expected": expected},
        "step": 400, "max_steps": 500,
        "instances": [{"id": 1, "text": "basket near a shelf",
                       "reported": False}],
        "frontiers": [{"id": "f0", "path_cost_m": 3.0}],
        "navigation": {"active_target": {"type": "instance", "id": 1}},
    }


class _ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, images):
        self.calls.append((prompt, images))
        return self.replies.pop(0) if self.replies else None


def _hit():
    return {"found": True, "point": [1.0, 2.0, 0.0], "text": "a basket",
            "frame_id": 5, "candidate_id": "c5", "point_score": 0.9,
            "semantic_validation": {"valid": True, "confidence": 0.99,
                                    "reason": "test fixture"}}


# ---------------------------------------------------------------- 工具：notes
def test_set_notes_tool_truncates_and_roundtrips():
    agent = _make_agent()
    out = agent._tool_set_notes("plan: check frame 7")
    assert out == {"notes": "plan: check frame 7"}
    assert agent._notes == "plan: check frame 7"
    out = agent._tool_set_notes("x" * 600)
    assert len(agent._notes) == 500
    assert out["notes"] == agent._notes


# -------------------------------------------------------- 工具：action history
def test_action_history_excludes_pending_and_pages():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=10)
    agent._record_action("GOTO_INSTANCE", 3)
    agent._last_observation = SimpleNamespace(step_count=11)
    agent._record_action("EXPLORE", None)
    # 第一条已结算，第二条 outcome=None（进行中）不出现在历史里
    agent._action_log[0]["outcome"] = "ok"
    rows = agent._tool_get_action_history()
    assert rows == [{"step": 10, "action": "GOTO_INSTANCE",
                     "target_id": "3", "outcome": "ok"}]
    agent._action_log[1]["outcome"] = "collision"
    rows = agent._tool_get_action_history(before_step=11)
    assert [r["step"] for r in rows] == [10]
    rows = agent._tool_get_action_history(limit=1)
    assert [r["step"] for r in rows] == [11]


def test_action_history_limit_returns_latest():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=0)
    for i in range(5):
        agent._last_observation = SimpleNamespace(step_count=i)
        agent._record_action("EXPLORE", None)
        agent._action_log[-1]["outcome"] = "ok"
    rows = agent._tool_get_action_history(limit=2)
    assert [r["step"] for r in rows] == [3, 4]


def test_record_action_caps_log_at_500():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=1)
    for _ in range(600):
        agent._record_action("EXPLORE", None)
    assert len(agent._action_log) == 500


def test_settle_action_outcomes_marks_collision_and_ok():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=5)
    agent._record_action("GOTO_FRONTIER", "f0")
    agent._last_motion_failed = True
    agent._settle_action_outcomes()
    assert agent._action_log[0]["outcome"] == "collision"
    agent._record_action("TURN_LEFT", None)
    agent._last_motion_failed = False
    agent._settle_action_outcomes()
    assert agent._action_log[1]["outcome"] == "ok"


def test_mark_goto_arrived_settles_latest_goto():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=5)
    agent._record_action("EXPLORE", None)
    agent._record_action("GOTO_INSTANCE", 3)
    agent._mark_goto_arrived()
    assert agent._action_log[1]["outcome"] == "arrived"
    assert agent._action_log[0]["outcome"] is None


# --------------------------------------------------------- 工具：map status
def test_get_agent_status_aggregates_server_and_agent_state():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=100, max_steps=500)
    agent.memory.add([1, 0, 0], "basket")
    reported = agent.memory.add([2, 0, 0], "reported basket")
    agent.memory.mark_reported(reported)
    agent.client = SimpleNamespace(
        get_state=lambda: {"num_frames": 42, "num_submaps": 3,
                           "num_loop_closures": 1, "caption_pending": 2,
                           "semantic": {"caption_enabled": True}},
        get_captioned_frame_ids=lambda: (True, [3, 5, 7, 9, 11, 13, 15]))
    status = agent._tool_get_agent_status()
    assert status["num_frames"] == 42
    assert status["caption_pending"] == 2
    assert status["latest_captioned_frame_ids"] == [7, 9, 11, 13, 15]
    assert status["instances_total"] == 2
    assert status["unreported_instances"] == 1
    assert status["steps_remaining"] == 400


def test_get_agent_status_swallows_server_errors():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_state=lambda: (_ for _ in ()).throw(RuntimeError("rpc down")))
    out = agent._tool_get_agent_status()
    assert "error" in out


# --------------------------------------------------------- 工具：view_frame
def test_view_frame_returns_jpeg_payload():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_frame_image=lambda fid: (({"found": True}, b"jpeg-7")
                                     if fid == 7 else ({"found": False}, b"")))
    assert agent._tool_view_frame(7) == b"jpeg-7"
    assert agent._tool_view_frame(8) is None
    agent.client = SimpleNamespace()      # 旧服务端无此方法
    assert agent._tool_view_frame(7) is None


# ------------------------------- 工具：use_molmo_point / instantiate_points
def _candidate():
    """两段式 instantiate_points 第一段的候选行（prepare_pixels 输出）。"""
    return {"frame_id": 5, "pixel": [330, 186], "pixel_norm": [637.1, 359.1],
            "candidate_id": "c5", "bbox": [300, 160, 380, 220],
            "point_score": 0.9}


def _two_stage_client():
    """带 prepare_pixels/get_candidate_evidence/resolve_candidates 的 mock。

    prepare_pixels 按请求的像素生成候选（与真实服务端一致），使不同坐标
    的重发得到不同 pixel_norm，从而能测试 ±2 容差确认与未确认分支。
    """

    def prepare_pixels(fid, pixels, normalized=True):
        cand = _candidate()
        cand["pixel_norm"] = [float(pixels[0][0]), float(pixels[0][1])]
        cand["pixel"] = [round(p * 518.0 / 1000.0, 1)
                         for p in cand["pixel_norm"]]
        return {"candidates": [cand]}

    return SimpleNamespace(
        prepare_pixels=prepare_pixels,
        get_candidate_evidence=lambda cid:
            ({"found": True}, b"confirm-jpeg") if cid == "c5"
            else ({"found": False}, b""),
        resolve_candidates=lambda ids:
            {"c5": {"found": True, "point": [1.0, 2.0, 0.0]}})


def test_instantiate_points_unconfirmed_returns_pending_and_no_instances():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = _two_stage_client()
    out = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    # 第一段：只渲染十字证据图返回，不注册任何实例
    assert out["instances"] == []
    assert out["semantic_rejections"] == []
    assert out["geometry_rejections"] == []
    assert out["pending_confirmation"] == [
        {"candidate_id": "c5", "frame_id": 5, "pixel": [637.1, 359.1]}]
    assert out["_tool_images"] == [("confirm_c5", b"confirm-jpeg")]
    assert agent.memory.nodes == []


def test_instantiate_points_accept_review_then_ingests():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = _two_stage_client()
    first = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    assert first["instances"] == []
    review = agent._tool_review_crosshair(
        5, [637.1, 359.1], "ACCEPT", "crosshair center is on the basket")
    assert review["instantiation_allowed"] is True
    second = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    assert "pending_confirmation" not in second
    assert "_tool_images" not in second      # 十字图已渲染过，不重复渲染
    assert len(agent.memory.nodes) == 1
    row = second["instances"][0]
    assert row["instance_id"] == agent.memory.nodes[0].iid
    assert row["observation_id"] == 1
    assert row["frame_id"] == 5
    assert row["association"] == "new_without_visual_relation"
    assert agent.memory.nodes[0].text == "basket"


def test_candidate_transaction_only_commits_explicit_accepts():
    """批量 proposal 不可导航；仅 ACCEPT 才进入 canonical memory。"""
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    client = _two_stage_client()
    client.point_pixels = lambda fid, query: {
        "width": 518, "height": 518,
        "points": [{"pixel": [330, 186], "confidence": 0.9}]}
    agent.client = client
    proposed = agent._tool_propose_candidates(5, "basket")
    assert proposed["proposals"] == [
        {"candidate_id": "c5", "frame_id": 5,
         "pixel": [637.1, 359.1]}]
    assert agent.memory.nodes == []
    committed = agent._tool_commit_candidates([
        {"candidate_id": "c5", "verdict": "ACCEPT",
         "reason": "crosshair is on basket"}], "basket")
    assert len(committed["instances"]) == 1
    assert len(agent.memory.nodes) == 1


def test_candidate_transaction_keeps_uncertain_out_of_memory():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    client = _two_stage_client()
    client.point_pixels = lambda fid, query: {
        "width": 518, "height": 518,
        "points": [{"pixel": [330, 186], "confidence": 0.9}]}
    agent.client = client
    agent._tool_propose_candidates(5, "basket")
    committed = agent._tool_commit_candidates([
        {"candidate_id": "c5", "verdict": "UNCERTAIN"}], "basket")
    assert committed["uncertain"] == ["c5"]
    assert agent._proposals["c5"]["status"] == "uncertain"
    assert agent.memory.nodes == []


def test_reject_or_uncertain_crosshair_never_instantiates():
    for verdict in ("REJECT", "UNCERTAIN"):
        agent = _make_agent()
        agent._last_observation = SimpleNamespace(step_count=50)
        agent.client = _two_stage_client()
        agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
        review = agent._tool_review_crosshair(
            5, [637.1, 359.1], verdict, "crosshair is not a clear basket")
        assert review["instantiation_allowed"] is False
        out = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
        assert out["instances"] == []
        assert out["semantic_rejections"][0]["verdict"] == verdict
        assert agent.memory.nodes == []


def test_crosshair_review_requires_shown_evidence():
    agent = _make_agent()
    out = agent._tool_review_crosshair(5, [637.1, 359.1], "ACCEPT")
    assert "error" in out


def test_instantiate_points_rounding_within_tolerance_accepts_review():
    """VLM 审核/重发同坐标但四舍五入（±2）仍匹配同一证据。"""
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = _two_stage_client()
    agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    agent._tool_review_crosshair(5, [637.1, 359.1], "ACCEPT")
    out = agent._tool_instantiate_points(5, [[637, 359]], "basket")
    assert "pending_confirmation" not in out
    assert len(agent.memory.nodes) == 1
    # 明显不同的坐标不算确认，重新进入 pending
    out2 = agent._tool_instantiate_points(5, [[300.0, 300.0]], "basket")
    assert out2["instances"] == []
    assert out2["pending_confirmation"][0]["candidate_id"] == "c5"


def test_instantiate_points_geometry_rejection_when_depth_invalid():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    client = _two_stage_client()
    client.resolve_candidates = lambda ids: {
        "c5": {"found": False, "error": "no valid depth"}}
    agent.client = client
    agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    agent._tool_review_crosshair(5, [637.1, 359.1], "ACCEPT")
    out = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    assert out["instances"] == []
    assert "pending_confirmation" not in out
    assert out["geometry_rejections"] == [{
        "candidate_id": "c5", "frame_id": 5,
        "pixel": [637.1, 359.1], "reason": "no valid depth"}]
    assert agent.memory.nodes == []


def test_molmo_pixels_require_explicit_accept_before_instantiation():
    """Molmo 十字图只提供审核证据，不能自动授予实例化权限。"""
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = _two_stage_client()
    agent._record_crosshair_evidence(5, [637.1, 359.1])
    pending = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    assert pending["instances"] == []
    assert pending["pending_confirmation"]
    agent._tool_review_crosshair(5, [637.1, 359.1], "ACCEPT")
    out = agent._tool_instantiate_points(5, [[637.1, 359.1]], "basket")
    assert len(agent.memory.nodes) == 1


def test_instantiate_points_requires_observation_and_handles_errors():
    agent = _make_agent()
    assert "error" in agent._tool_instantiate_points(5, [[500, 500]], "basket")
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace(
        prepare_pixels=lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("rpc down")))
    assert "error" in agent._tool_instantiate_points(5, [[500, 500]], "basket")
    agent.client = SimpleNamespace(
        prepare_pixels=lambda *_a, **_k: {"candidates": []})
    assert agent._tool_instantiate_points(5, [[500, 500]], "basket") == {
        "instances": [], "semantic_rejections": [], "geometry_rejections": []}


def test_instantiate_points_rejects_legacy_server_without_audit_path():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace(instantiate_pixels=lambda *_a, **_k: {
        "results": [_hit()]})
    rows = agent._tool_instantiate_points(5, [[500, 500]], "basket")
    assert "error" in rows
    assert agent.memory.nodes == []


def test_instantiate_points_label_becomes_initial_text():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = _two_stage_client()
    agent._tool_instantiate_points(5, [[500, 500]], "wooden chair by the table")
    agent._tool_review_crosshair(5, [500, 500], "ACCEPT")
    agent._tool_instantiate_points(5, [[500, 500]], "wooden chair by the table")
    assert agent.memory.nodes[0].text == "wooden chair by the table"


def test_instantiate_points_requires_pixels():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace()
    out = agent._tool_instantiate_points(5, [], "basket")
    assert "error" in out


def test_instantiate_points_propagates_server_error():
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = SimpleNamespace()
    out = agent._tool_instantiate_points(99, [[500, 500]], "basket")
    assert "error" in out


def test_confirm_images_reach_decision_prompt():
    """agent_loop 将 _tool_images 弹出 JSON 并附加到下一轮提示的图像通道。"""
    import decision.agent_loop as loop_mod

    class Chat:
        def decide(self, event, state, state_fn, **kwargs):
            return None  # 不实际调用

    chat = Chat()
    agent = _make_agent()
    agent._last_observation = SimpleNamespace(step_count=50)
    agent.client = _two_stage_client()

    loop = loop_mod.DecisionLoop(
        chat_fn=chat, tools={
            "instantiate_points": agent._tool_instantiate_points})
    payload, images, ok = loop._run_tool(
        {"name": "instantiate_points", "frame_id": 5,
         "pixels_1000": [[637.1, 359.1]], "label": "basket"})
    assert ok
    # 证据图不进 JSON
    assert "_tool_images" not in payload
    assert images == [("confirm_c5", b"confirm-jpeg")]
    # 附加到图像通道：重复调用同 label 不累积
    images2 = loop._with_tool_image(
        list(images), "confirm_c5", b"confirm-jpeg-2")
    assert len(images2) == 1
    assert images2[0] == ("confirm_c5", b"confirm-jpeg-2")


def test_pointing_backend_error_code_reaches_decision_tool():
    agent = _make_agent()
    agent.client = SimpleNamespace(point_pixels=lambda *_args, **_kwargs: {
        "points": [], "error_code": "POINTING_BACKEND_UNAVAILABLE",
        "error": "connection refused"})
    out = agent._tool_use_molmo_point(7, "basket")
    assert out == {"error": {
        "code": "POINTING_BACKEND_UNAVAILABLE",
        "message": "connection refused"}}


def test_use_molmo_point_returns_normalized_pixels_and_records_evidence():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        point_pixels=lambda fid, text: {
            "width": 1280, "height": 720,
            "points": [{"pixel": [640.0, 360.0], "confidence": 1.0,
                        "bbox": None}]},
        evidence_for_point=lambda fid, pixel, bbox:
            ({"found": True}, b"molmo-jpeg"))
    out = agent._tool_use_molmo_point(7, "wooden chair")
    assert out["points"] == [{"pixel": [500.0, 500.0], "confidence": 1.0}]
    # 十字证据图随结果返回，供 VLM 看图审核
    assert out["_tool_images"] == [("molmo_point_7_0", b"molmo-jpeg")]
    # 渲染只登记“已展示证据”，仍需要显式 ACCEPT 审核。
    assert agent._find_crosshair_key(
        7, [500.0, 500.0], agent._crosshair_evidence) is not None
    assert agent._find_crosshair_key(
        7, [500.0, 500.0], agent._crosshair_reviews) is None


def test_use_molmo_point_skips_unrenderable_points():
    """证据图渲染失败的标点返回坐标但不登记确认资格。"""
    agent = _make_agent()
    agent.client = SimpleNamespace(
        point_pixels=lambda fid, text: {
            "width": 1280, "height": 720,
            "points": [{"pixel": [640.0, 360.0], "confidence": 1.0,
                        "bbox": None}]},
        evidence_for_point=lambda fid, pixel, bbox:
            ({"found": False}, b""))
    out = agent._tool_use_molmo_point(7, "wooden chair")
    assert out["points"] == [{"pixel": [500.0, 500.0], "confidence": 1.0}]
    assert "_tool_images" not in out
    assert agent._find_crosshair_key(
        7, [500.0, 500.0], agent._crosshair_evidence) is None


def test_use_molmo_point_propagates_server_error():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        point_pixels=lambda fid, text: {"points": [],
                                        "error": "unknown frame_id 99"})
    out = agent._tool_use_molmo_point(99, "wooden chair")
    assert "error" in out


# -------------------------------------------- world-state：notes / recent_actions
def test_world_state_includes_notes_and_recent_actions():
    agent = _make_agent()
    agent._notes = "working on frame 7"
    agent._last_observation = SimpleNamespace(step_count=9)
    agent._record_action("GOTO_INSTANCE", 1)
    agent._action_log[0]["outcome"] = "arrived"
    agent._record_action("EXPLORE", None)      # 进行中，不注入
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state = build_world_state(agent, obs)
    assert state["notes"] == "working on frame 7"
    assert state["recent_actions"] == [
        {"step": 9, "action": "GOTO_INSTANCE", "target_id": "1",
         "outcome": "arrived"}]


def test_world_state_recent_actions_caps_at_three():
    agent = _make_agent()
    for i in range(5):
        agent._last_observation = SimpleNamespace(step_count=i)
        agent._record_action("EXPLORE", None)
        agent._action_log[-1]["outcome"] = "ok"
    obs = SimpleNamespace(step_count=50, max_steps=500, goal_text="x")
    state = build_world_state(agent, obs)
    assert [r["step"] for r in state["recent_actions"]] == [2, 3, 4]


# -------------------------------------------- _build_decider_input：新关键帧通知
def test_build_decider_input_attaches_new_keyframes_once():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (np.stack([np.eye(4)] * 3), [0, 1, 2]),
        get_state=lambda: {"caption_pending": 0},
        get_captioned_frame_ids=lambda: (True, [3, 5]),
        get_captions=lambda ids: {"captions": [
            {"frame_id": 3, "caption": "a kitchen"},
            {"frame_id": 5, "caption": "c" * 300}]})
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state, _map = agent._build_decider_input(obs)
    assert state["new_keyframes"] == [
        {"frame_id": 3, "caption": "a kitchen"},
        {"frame_id": 5, "caption": "c" * 200}]
    assert agent._last_notified_frame_id == 5
    # 第二次决策无新帧：不再携带该字段
    state, _map = agent._build_decider_input(obs)
    assert "new_keyframes" not in state


def test_build_decider_input_skips_keyframes_when_server_is_old():
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (np.stack([np.eye(4)] * 3), [0, 1, 2]),
        get_state=lambda: {"caption_pending": 0})
    obs = SimpleNamespace(step_count=50, max_steps=500, goal_text="x")
    state, _map = agent._build_decider_input(obs)
    assert "new_keyframes" not in state


def test_build_decider_input_exposes_current_frame_id():
    """最新已喂帧号进入 world state：逼近后可直接 view/instantiate。"""
    agent = _make_agent()
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (np.stack([np.eye(4)] * 3), [0, 1, 2]),
        get_state=lambda: {"caption_pending": 0})
    agent._last_feed_info = {"frame_id": 42, "busy": False}
    obs = SimpleNamespace(step_count=50, max_steps=500, goal_text="x")
    state, _map = agent._build_decider_input(obs)
    assert state["navigation"]["current_frame_id"] == 42
    # 无喂帧记录时安全置 None
    agent._last_feed_info = {}
    state, _map = agent._build_decider_input(obs)
    assert state["navigation"]["current_frame_id"] is None
    # 回归：最新 feed 帧可能尚未入 submap（异步处理），current_frame_id
    # 必须用已可检索的 last_available_frame_id 而不是 feed 帧号，
    # 否则 VLM 按提示词直接 instantiate_points 会 unknown frame_id。
    agent._last_feed_info = {"frame_id": 131, "busy": False,
                             "last_available_frame_id": 128}
    state, _map = agent._build_decider_input(obs)
    assert state["navigation"]["current_frame_id"] == 128
    # 尚未处理完任何 submap（0）时回退 feed 帧号（旧 server 无此字段同理）
    agent._last_feed_info = {"frame_id": 5, "busy": False,
                             "last_available_frame_id": 0}
    state, _map = agent._build_decider_input(obs)
    assert state["navigation"]["current_frame_id"] == 5


# ------------------------------------------------------ agent_loop：view_frame
def test_view_frame_tool_attaches_frame_image():
    chat = _ScriptedChat([
        {"tool_call": {"name": "view_frame", "frame_id": 7}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"view_frame": lambda frame_id: b"frame-jpeg"}).decide(
            "world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    assert ("tool_frame_7_rgb", b"frame-jpeg") in chat.calls[1][1]
    assert '"image_ref": "tool_frame_7_rgb"' in chat.calls[1][0]


def test_view_frame_missing_image_returns_error():
    chat = _ScriptedChat([
        {"tool_call": {"name": "view_frame", "frame_id": 7}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    result = DecisionLoop(
        chat, tools={"view_frame": lambda frame_id: None}).decide(
            "world_state_updated", _state())
    assert result.action == "GOTO_FRONTIER"
    assert "frame image not found" in chat.calls[1][0]


# ------------------------------------------- agent_loop：adjustment 禁工具
def test_adjustment_disables_tool_calls():
    called = []
    chat = _ScriptedChat([
        {"tool_call": {"name": "search_frames", "query": "x"}},
        {"action": "END_ADJUST"},
    ])
    result = DecisionLoop(
        chat,
        tools={"search_frames": lambda query: called.append(query) or []}
    ).decide("adjustment", _state())
    assert result.action == "END_ADJUST"
    assert called == []
    assert result.tool_calls == 0
    assert "tools are disabled during adjustment" in chat.calls[1][0]


def test_new_write_tools_refresh_world_state():
    fresh = _state()

    def set_notes(text):
        return {"notes": text}

    chat = _ScriptedChat([
        {"tool_call": {"name": "set_notes", "text": "go to frame 7"}},
        {"action": "GOTO_FRONTIER", "target_id": "f0"},
    ])
    calls = []

    def state_fn():
        calls.append(1)
        return fresh

    result = DecisionLoop(
        chat, tools={"set_notes": set_notes}).decide(
            "world_state_updated", _state(), state_fn=state_fn)
    assert result.action == "GOTO_FRONTIER"
    assert calls == [1]          # set_notes 触发 world-state 刷新
    assert "World state after your write" in chat.calls[1][0]


# ------------------------------------------------------- 放宽后的事件白名单
def test_relaxed_event_whitelists():
    # arrival 允许 GOTO_FRONTIER
    result = DecisionLoop(_ScriptedChat([
        {"action": "GOTO_FRONTIER", "target_id": "f0"}])).decide(
            "arrival", _state())
    assert result.action == "GOTO_FRONTIER" and result.target_id == "f0"
    # world_state_updated 允许 FINISH / REPORT_FOUND / SCAN
    for action in ("FINISH", "REPORT_FOUND", "SCAN"):
        target_id = "1" if action == "REPORT_FOUND" else None
        result = DecisionLoop(_ScriptedChat([{
            "action": action, "target_id": target_id}])).decide(
                "world_state_updated", _state())
        assert result.action == action
    # finish_check 仍不允许 SCAN / REPORT_FOUND
    for rejected in ("SCAN", "REPORT_FOUND"):
        chat = _ScriptedChat([
            {"action": rejected}, {"action": "FINISH"}])
        assert DecisionLoop(chat).decide(
            "finish_check", _state()).action == "FINISH"
    # EXPLORE 在所有事件中都被拒绝（finish_check 也不例外）
    chat = _ScriptedChat([
        {"action": "EXPLORE"}, {"action": "FINISH"}])
    assert DecisionLoop(chat).decide(
        "finish_check", _state()).action == "FINISH"
    # adjustment 白名单不变
    chat = _ScriptedChat([
        {"action": "EXPLORE"}, {"action": "END_ADJUST"}])
    assert DecisionLoop(chat).decide(
        "adjustment", _state()).action == "END_ADJUST"


# ------------------------------------------------------------ get_captions
def test_search_frames_is_pure_caption_retrieval_and_never_grounds():
    """回归：_tool_search_frames 曾引用未定义 frame_id 抛 NameError，
    导致决策层检索全断（远端 ep3 只选 frontier 零上报）。只读契约：
    仅走 retrieve_captions，绝不触发 pointing/ground 路径。"""
    agent = _make_agent()
    retrieved = []
    grounded = []
    agent.client = SimpleNamespace(
        retrieve_captions=lambda query, top_k=5: retrieved.append((query, top_k))
        or [{"frame_id": 7, "score": 0.8, "caption": "bicycle in hallway"}],
        # 若误走遗留 ground 分支，旧代码会因未定义 frame_id 抛 NameError，
        # 或经 ground_object_pixels 触发这些调用。
        point_pixels=lambda *a, **k: grounded.append("point_pixels") or {"points": []},
        prepare_pixels=lambda *a, **k: grounded.append("prepare_pixels") or {},
        ground_object_pixels=lambda *a, **k: grounded.append("ground_object_pixels")
        or {"results": []},
    )
    out = agent._tool_search_frames("bicycle bike cycle wheel", top_k=5)
    assert retrieved == [("bicycle bike cycle wheel", 5)]
    assert grounded == []
    assert out == [{"frame_id": 7, "score": 0.8, "caption": "bicycle in hallway"}]


def test_caption_store_get_captions_skips_missing():
    store = CaptionStore()
    store.add(3, None, "a kitchen", np.ones(8, dtype=np.float32))
    store.add(5, None, "a hallway", np.ones(8, dtype=np.float32))
    assert store.get_captions([3, 99, 5]) == [
        {"frame_id": 3, "caption": "a kitchen"},
        {"frame_id": 5, "caption": "a hallway"}]
    assert store.get_captions([]) == []


def test_client_get_captions_rpc_shape():
    client = MappingClient.__new__(MappingClient)
    seen = []

    def fake_request(header, payload=b"", retries=1):
        seen.append(header)
        return {"ok": True,
                "captions": [{"frame_id": 3, "caption": "x"}]}, b""

    client._request = fake_request
    out = client.get_captions([3, "5"])
    assert seen[0]["cmd"] == "get_captions"
    assert seen[0]["frame_ids"] == [3, 5]
    assert out["captions"] == [{"frame_id": 3, "caption": "x"}]


# ---------------------------------------------------------- nav 卡死恢复（方案 A）
def _nav_action_agent():
    """mode=nav + follower 挂 2m 直线路径；client 最小依赖，规划可替换。"""
    agent = _make_agent()
    fl = nav.PathFollower(scale=1.0, reach_m=0.8)
    fl.x, fl.y, fl.yaw = 0.0, 0.0, 0.0
    fl.set_path([(0.0, 0.0), (2.0, 0.0)])
    agent.follower = fl
    agent.mode = "nav"
    agent.target_point = np.array([2.0, 0.0, 0.0])
    agent.target_instance_id = 3
    agent._last_plan_step = 0
    agent._last_motion_failed = False
    agent._refresh_anchor = lambda poses, frame_ids: None
    return agent


def test_nav_collision_escapes_then_replans_then_stuck():
    """撞墙先转向脱困，再临时封路重规划；连续撞限次才 stuck。"""
    agent = _nav_action_agent()
    planned = []
    agent._plan_to_target = lambda obs: planned.append(1) or True
    obs = SimpleNamespace(step_count=1, previous_action=None)
    # 无碰撞：计数保持 0
    action, arrived, stuck = agent._nav_action(obs)
    assert not stuck and not arrived and action is not None
    assert agent._nav_collision_streak == 0
    # 碰撞 1：先执行交替转向，下一轮才在临时封路约束下重规划。
    obs.previous_action = int(Action.MOVE_FORWARD)
    agent._last_motion_failed = True
    action, arrived, stuck = agent._nav_action(obs)
    assert not stuck and not arrived
    assert action == int(Action.TURN_LEFT)
    assert len(agent._nav_blocked_points) == 1
    assert len(planned) == 0
    obs.previous_action = int(Action.TURN_LEFT)
    agent._last_motion_failed = False
    action, arrived, stuck = agent._nav_action(obs)
    assert not stuck and not arrived and action == int(Action.MOVE_FORWARD)
    assert len(planned) == 1
    # 再次前进碰撞后交替向右转；仍不会立刻判定不可达。
    obs.previous_action = int(Action.MOVE_FORWARD)
    agent._last_motion_failed = True
    action, arrived, stuck = agent._nav_action(obs)
    assert not stuck and action == int(Action.TURN_RIGHT)
    assert len(planned) == 1
    # 碰撞 3：达到 nav_collision_limit → stuck
    obs.previous_action = int(Action.MOVE_FORWARD)
    agent._last_motion_failed = True
    action, arrived, stuck = agent._nav_action(obs)
    assert stuck and action is None and not arrived
    assert agent._nav_collision_streak == 3
    # 恢复运动后计数清零，可再次进入卡死检测
    obs.previous_action = int(Action.MOVE_FORWARD)
    agent._last_motion_failed = False
    action, arrived, stuck = agent._nav_action(obs)
    assert agent._nav_collision_streak == 0 and not stuck


def test_nav_stuck_recovery_marks_unreachable_and_asks_decider():
    """stuck → 实例标记不可达 + nav_failed 决策事件 + 映射动作执行。"""
    agent = _nav_action_agent()
    agent.client = SimpleNamespace(
        get_all_poses=lambda: (np.stack([np.eye(4)] * 3), [0, 1, 2]),
        get_state=lambda: {"caption_pending": 0})
    events = []

    class _Loop:
        def decide(self, event, state, map_png=None, images=None,
                   state_fn=None):
            events.append(event)
            state, _map = state_fn()
            assert state["navigation"]["blocked_target"]["instance_id"] == 3
            return DecisionResult("FINISH", None)

    agent.decision_loop = _Loop()
    agent._plan_to_target = lambda obs: True
    obs = SimpleNamespace(step_count=1, max_steps=500,
                          rgb=np.zeros((16, 16, 3), dtype=np.uint8),
                          previous_action=int(Action.MOVE_FORWARD))
    agent._last_motion_failed = True
    agent._nav_action(obs)                 # streak 1
    agent._last_motion_failed = True
    agent._nav_action(obs)                 # streak 2
    agent._last_motion_failed = True
    action, arrived, stuck = agent._nav_action(obs)   # streak 3 → stuck
    assert stuck
    result = agent._nav_failed_recovery(obs)
    assert 3 in agent._unreachable_instance_ids
    assert events == ["nav_failed"]
    assert result == int(Action.FINISH)


def test_nav_stuck_recovery_fallback_to_explore_when_decider_unavailable():
    agent = _nav_action_agent()
    agent._unreachable_instance_ids.clear()
    agent._plan_to_target = lambda obs: True
    obs = SimpleNamespace(step_count=1,
                          previous_action=int(Action.MOVE_FORWARD))
    for _ in range(3):
        agent._last_motion_failed = True
        agent._nav_action(obs)
    result = agent._nav_failed_recovery(obs)   # decision_loop None
    assert 3 in agent._unreachable_instance_ids
    assert agent.mode == "explore"
    assert isinstance(result, int)


def test_metric_snapshot_defers_one_off_scale_jump():
    agent = _make_agent()
    assert agent._update_metric_snapshot(2.0, "grid") == 2.0
    assert agent._update_metric_snapshot(3.0, "grid") == 2.0
    assert agent._update_metric_snapshot(3.02, "grid") == 2.0
    assert agent._update_metric_snapshot(2.98, "grid") == 2.98
    assert agent._metric_snapshot["revision"] == 2


def test_unreachable_instance_excluded_from_world_state():
    agent = _make_agent()
    agent._unreachable_instance_ids = {2}
    agent.memory = SimpleNamespace(
        nodes=[
            SimpleNamespace(iid=1, reported=False, text="basket",
                            point=np.array([1.0, 1.0, 0.0]), step=1,
                            observation_ids=[], report_claim_id=None,
                            candidate_id=None),
            SimpleNamespace(iid=2, reported=False, text="basket too",
                            point=np.array([2.0, 2.0, 0.0]), step=2,
                            observation_ids=[], report_claim_id=None,
                            candidate_id=None)])
    obs = SimpleNamespace(step_count=50, max_steps=500,
                          goal_text="Find all baskets")
    state = build_world_state(agent, obs)
    assert [r["id"] for r in state["instances"]] == [1]
    assert state["instances_total"] == 2
    assert state["instances_unreachable_ids"] == ["2"]


def test_activate_memory_target_skips_unreachable():
    agent = _make_agent()
    agent._unreachable_instance_ids = {1}
    agent.memory = SimpleNamespace(
        nodes=[
            SimpleNamespace(iid=1, reported=False, text="basket",
                            point=np.array([1.0, 1.0, 0.0]), step=1,
                            observation_ids=[], report_claim_id=None,
                            candidate_id=None),
            SimpleNamespace(iid=2, reported=False, text="basket too",
                            point=np.array([2.0, 2.0, 0.0]), step=2,
                            observation_ids=[], report_claim_id=None,
                            candidate_id=None)])
    agent._ordered_memory_nodes = lambda: agent.memory.nodes
    agent._plan_to_target = lambda obs: True
    obs = SimpleNamespace(step_count=1)
    assert agent._activate_memory_target(obs)
    assert agent.target_instance_id == 2
    assert agent.mode == "nav"
