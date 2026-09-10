"""图像目标（image-goal）支持回归测试。

覆盖：模式检测与收图、冷启动描述与失败标记（不可用而非任务句占位）、
决策附件只含未找到目标、goal_index 记账链（实例化 -> 实例 -> report 撤下）、
description 模式不受影响。
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.nav_agent import NavAgent
from decision import prompts


class _FakeVLM:
    def __init__(self, description_reply=None):
        self.description_reply = description_reply
        self.chat_calls = []

    def encode_rgb(self, rgb):
        return b"jpeg-bytes"

    def chat_text(self, prompt, images=None, max_tokens=None):
        self.chat_calls.append((prompt, images))
        return self.description_reply


def _obs(step=0, goal_type="description", goal_images=None):
    return SimpleNamespace(
        step_count=step, goal_text="Find the target",
        target_mode="all", target_count=None,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8), max_steps=500,
        episode_id="ep_test", previous_action=None,
        goal_type=goal_type, goal_images=goal_images)


def _make_agent(vlm=None):
    agent = NavAgent()
    agent.vlm = vlm or _FakeVLM()
    agent.align_R = np.eye(3)
    agent.calibrator = SimpleNamespace(current_scale=lambda: 1.0, actions=[])
    return agent


def _goal_rgb(seed):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = seed
    return img


def test_image_mode_detection_and_capture():
    vlm = _FakeVLM("goal_image_0: a round wall clock\n"
                   "goal_image_1: a purple armchair")
    agent = _make_agent(vlm)
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1), _goal_rgb(2)]))
    assert agent._image_goal_mode is True
    assert len(agent._goal_images) == 2
    assert agent._goal_descriptions == ["a round wall clock",
                                        "a purple armchair"]
    assert len(vlm.chat_calls) == 1
    # 幂等：第二次 capture 不再收图也不再调 VLM
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(3)]))
    assert len(agent._goal_images) == 2
    assert len(vlm.chat_calls) == 1


def test_description_mode_untouched():
    vlm = _FakeVLM("should not be called")
    agent = _make_agent(vlm)
    agent._capture_goal_images(_obs(goal_type="description",
                                    goal_images=[_goal_rgb(1)]))
    assert agent._image_goal_mode is False
    assert agent._goal_images == []
    assert vlm.chat_calls == []
    assert agent._goal_images_payload() == []


def test_cold_start_description_unavailable_not_faked():
    """冷启动描述生成失败：标记为空（不可用），不再用任务句伪装有效
    描述；检索查询跳过它，但照片仍随决策附件发给 VLM。"""
    vlm = _FakeVLM(None)  # API 失败
    agent = _make_agent(vlm)
    obs = _obs(goal_type="image", goal_images=[_goal_rgb(1)])
    agent._last_observation = obs
    agent._capture_goal_images(obs)
    assert agent._goal_descriptions == [""]
    assert agent._goal_retrieval_queries() == []
    assert [label for label, _ in agent._goal_images_payload()] == [
        "goal_image_0"]


def test_cold_start_description_rejects_task_sentence_echo():
    """VLM 原样复述任务句也不算有效描述，必须留空等待重试。"""
    vlm = _FakeVLM("goal_image_0: Find the target")
    agent = _make_agent(vlm)
    obs = _obs(goal_type="image", goal_images=[_goal_rgb(1)])
    agent._last_observation = obs
    agent._capture_goal_images(obs)
    assert agent._goal_descriptions == [""]


def test_payload_excludes_found_goals():
    agent = _make_agent(_FakeVLM("goal_image_0: clock\ngoal_image_1: chair"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1), _goal_rgb(2)]))
    payload = agent._goal_images_payload()
    assert [label for label, _ in payload] == ["goal_image_0",
                                               "goal_image_1"]
    agent._goal_found.add(0)
    payload = agent._goal_images_payload()
    assert [label for label, _ in payload] == ["goal_image_1"]


def _hit(point, candidate_id, goal_index=None, frame_id=10,
         pixel=(100.0, 100.0)):
    return {"point": list(point), "found": True, "frame_id": frame_id,
            "candidate_id": candidate_id, "pixel": list(pixel),
            "bbox": None, "point_score": 1.0, "text": "target",
            "goal_index": goal_index}


def test_goal_index_bookkeeping_and_report():
    agent = _make_agent(_FakeVLM("goal_image_0: clock"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1)]))
    obs = _obs(step=100)
    agent._last_observation = obs
    changed = agent._ingest_semantic_hits(
        obs, [_hit([1.0, 2.0, 0.0], "c1", goal_index=0)], select=False)
    iid = changed[0]["instance_id"]
    assert agent._instance_goal_index[iid] == 0
    # report 后目标撤下
    agent.target_instance_id = iid
    action = agent._report_found(iid)
    from benchmark_api import Action
    assert action == int(Action.TARGET_FOUND)
    assert agent._goal_found == {0}
    assert agent._goal_images_payload() == []


def test_goal_index_ignored_in_description_mode():
    agent = _make_agent()
    agent._last_observation = _obs(step=100)
    changed = agent._ingest_semantic_hits(
        _obs(step=100), [_hit([1.0, 2.0, 0.0], "c1", goal_index=0)],
        select=False)
    assert changed[0]["instance_id"] not in agent._instance_goal_index


def test_merge_migrates_goal_index():
    agent = _make_agent(_FakeVLM("goal_image_0: clock"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1)]))
    obs = _obs(step=100)
    agent._last_observation = obs
    keep = agent.instance_store.add([0.0, 0.0, 0.0], "old clock")
    changed = agent._ingest_semantic_hits(
        obs, [_hit([5.0, 5.0, 0.0], "c1", goal_index=0)], select=False)
    drop_id = changed[0]["instance_id"]
    out = agent._tool_merge_instances(keep.iid, drop_id)
    assert out["into"] == keep.iid
    assert agent._instance_goal_index[keep.iid] == 0
    assert drop_id not in agent._instance_goal_index


def test_prompt_image_section_only_in_image_mode():
    state = {"task": {"goal": "g", "mode": "all", "found": 0,
                      "expected": None}}
    text = prompts.build_decision_prompt("world_state_updated", state, 15)
    assert "Image-goal mode" not in text
    state_img = {"task": {"goal": "g", "mode": "all", "found": 0,
                          "expected": None, "goal_type": "image",
                          "goal_descriptions": ["a clock"],
                          "goals_unfound": [0], "goals_total": 1}}
    text = prompts.build_decision_prompt("world_state_updated", state_img, 15)
    assert "Image-goal mode" in text
    assert "goal_index=N" in text


def _two_goal_agent():
    agent = _make_agent(_FakeVLM("goal_image_0: clock\ngoal_image_1: chair"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1), _goal_rgb(2)]))
    agent._last_observation = _obs(step=100)
    return agent


def test_goal_retrieval_queries_skip_found():
    agent = _two_goal_agent()
    assert agent._goal_retrieval_queries() == [(0, "clock"), (1, "chair")]
    agent._goal_found.add(0)
    assert agent._goal_retrieval_queries() == [(1, "chair")]
    agent._goal_found.add(1)
    assert agent._goal_retrieval_queries() == []


def test_goal_relevant_frames_merge_per_frame():
    agent = _two_goal_agent()
    replies = {
        "clock": [{"frame_id": 5, "score": 0.9, "caption": "a wall clock"},
                  {"frame_id": 7, "score": 0.5, "caption": "c7"}],
        "chair": [{"frame_id": 5, "score": 0.7, "caption": "clock again"},
                  {"frame_id": 9, "score": 0.8, "caption": "a chair"}],
    }
    agent.client = SimpleNamespace(
        retrieve_captions=lambda query, top_k=5: replies[query])
    rows = agent._goal_relevant_frames()
    by_id = {row["frame_id"]: row for row in rows}
    # 同一帧被两个目标命中：合并成一行并带 matched_goals；分数取最高
    assert by_id[5]["matched_goals"] == [0, 1]
    assert by_id[5]["score"] == 0.9
    assert by_id[9]["matched_goals"] == [1]
    assert rows[0]["frame_id"] == 5              # 按分数降序
    # 找到 goal 0 后只剩一路查询
    agent._goal_found.add(0)
    rows = agent._goal_relevant_frames()
    assert all(row["matched_goals"] == [1] for row in rows)


def test_caption_hit_interrupt_queries_unfound_goals():
    agent = _two_goal_agent()
    agent._goal_found.add(0)                     # 只为未找到的 goal 1 检索
    seen = []
    agent.client = SimpleNamespace(
        get_captioned_frame_ids=lambda: (True, [42]),
        retrieve_captions=lambda query, top_k=3: (
            seen.append(query),
            [{"frame_id": 42, "score": 0.9, "caption": "chair"}])[1])
    chosen = []
    agent._choose_high_level_target = (
        lambda observation, event: chosen.append(event) or "DECIDE")
    out = agent._caption_hit_decision(_obs(step=10))
    assert out == "DECIDE"
    assert seen == ["chair"]
    assert any("goal_image_1" in e for e in agent._events)


def test_goal_index_conflict_withheld_then_rebound():
    agent = _two_goal_agent()
    obs = _obs(step=100)
    first = agent._ingest_semantic_hits(
        obs, [_hit([0.0, 0.0, 0.0], "c1", goal_index=0)], select=False)
    second = agent._ingest_semantic_hits(
        obs, [_hit([8.0, 8.0, 0.0], "c2", goal_index=0, frame_id=20,
                   pixel=(500.0, 500.0))], select=False)
    id_first = first[0]["instance_id"]
    id_second = second[0]["instance_id"]
    # 一图一实例：第二次绑定被扣下，冲突挂起且实例照常创建
    assert agent._instance_goal_index == {id_first: 0}
    assert agent.active_goal_conflicts() == [
        {"goal_index": 0, "holder_instance_id": id_first,
         "challenger_instance_id": id_second}]
    # 裁决：绑定转移到真身，另一方解绑，冲突清除
    out = agent._tool_resolve_goal_conflict(0, id_second)
    assert out["bound_instance"] == id_second
    assert out["unbound_instance"] == id_first
    assert agent._instance_goal_index == {id_second: 0}
    assert agent.active_goal_conflicts() == []


def test_resolve_goal_conflict_validation():
    agent = _two_goal_agent()
    obs = _obs(step=100)
    agent._ingest_semantic_hits(
        obs, [_hit([0.0, 0.0, 0.0], "c1", goal_index=0)], select=False)
    agent._ingest_semantic_hits(
        obs, [_hit([8.0, 8.0, 0.0], "c2", goal_index=0, frame_id=20,
                   pixel=(500.0, 500.0))], select=False)
    third = agent._ingest_semantic_hits(
        obs, [_hit([4.0, 4.0, 0.0], "c3", goal_index=1, frame_id=30,
                   pixel=(300.0, 300.0))], select=False)
    # 局外实例不能接收绑定
    out = agent._tool_resolve_goal_conflict(0, third[0]["instance_id"])
    assert "error" in out
    # 无冲突的 goal_index 报错
    assert "error" in agent._tool_resolve_goal_conflict(1, third[0][
        "instance_id"])
    # description 模式禁用
    plain = _make_agent()
    assert "error" in plain._tool_resolve_goal_conflict(0, 1)


def test_goal_conflict_pruned_after_merge_and_report():
    agent = _two_goal_agent()
    obs = _obs(step=100)
    first = agent._ingest_semantic_hits(
        obs, [_hit([0.0, 0.0, 0.0], "c1", goal_index=0)], select=False)
    second = agent._ingest_semantic_hits(
        obs, [_hit([8.0, 8.0, 0.0], "c2", goal_index=0, frame_id=20,
                   pixel=(500.0, 500.0))], select=False)
    id_first = first[0]["instance_id"]
    id_second = second[0]["instance_id"]
    # merge 吸收一方后冲突自动失效
    agent._tool_merge_instances(id_first, id_second)
    assert agent.active_goal_conflicts() == []
    # 重建冲突后报告目标：冲突同样失效
    agent._bind_goal_index(id_first, 0)
    other = agent._ingest_semantic_hits(
        obs, [_hit([20.0, 20.0, 0.0], "c4", goal_index=0, frame_id=40,
                   pixel=(700.0, 700.0))], select=False)
    assert agent.active_goal_conflicts()[0]["challenger_instance_id"] == \
        other[0]["instance_id"]
    agent.target_instance_id = id_first
    agent._report_found(id_first)
    assert agent.active_goal_conflicts() == []


def test_decision_prompt_shows_binding_conflicts():
    state_img = {"task": {"goal": "g", "mode": "all", "found": 0,
                          "expected": None, "goal_type": "image",
                          "goal_descriptions": ["a clock"],
                          "goals_unfound": [0], "goals_total": 1,
                          "goal_conflicts": [
                              {"goal_index": 0, "holder_instance_id": 3,
                               "challenger_instance_id": 5}]}}
    text = prompts.build_decision_prompt("world_state_updated", state_img, 15)
    assert "BINDING CONFLICTS" in text
    assert "resolve_goal_conflict" in text


def test_image_system_prompt_split():
    desc = prompts.build_decider_system(15)
    image = prompts.build_decider_system(15, image_mode=True)
    assert desc != image
    assert "resolve_goal_conflict" not in desc   # description 契约不变
    assert "goals_unfound" in image
    assert "matched_goals" in image
    assert "resolve_goal_conflict(goal_index, keep_instance_id)" in image
    # 工具 JSON 契约仍在 image 契约里
    assert '"tool_call":' in image


def test_image_mode_force_finish_when_report_quota_reached():
    agent = _make_agent(_FakeVLM("goal_image_0: clock\ngoal_image_1: chair"))
    agent._capture_goal_images(_obs(
        goal_type="image", goal_images=[_goal_rgb(1), _goal_rgb(2)]))
    agent._target_mode = "all"
    agent._decider_should_finish = lambda obs: (_ for _ in ()).throw(
        AssertionError("image finish must not consult VLM"))
    obs = _obs(step=100)
    # 未找齐：不结束（step=100 也达不到规则兜底的 late 条件）
    agent._reported_count = 1
    agent._goal_found = {0, 1}  # Photo checklist cannot override report count.
    assert agent._should_finish(obs) is False
    # Unbound reports still satisfy the quota.
    agent._reported_count = 2
    agent._goal_found = set()
    assert agent._should_finish(obs) is True
    agent._reported_count = 3
    assert agent._should_finish(obs) is True


def test_image_report_quota_finishes_on_next_act():
    from benchmark_api import Action
    from agents.decision_state import build_world_state

    agent = _two_goal_agent()
    obs = _obs(step=101, goal_type="image")
    for point in ([0, 0, 0], [4, 0, 0]):
        node = agent.instance_store.add(point, "target")
        agent.target_instance_id = node.iid
        assert agent._report_found(node.iid) == int(Action.TARGET_FOUND)
    assert agent._goal_found == set()
    state = build_world_state(agent, obs, start_xy=[0, 0], scale=1.0)
    assert state["task"]["found"] == 2
    assert state["task"]["goals_unfound"] == [0, 1]
    assert state["task"]["goals_matched_count"] == 0

    agent._feed_frame = lambda obs: None
    agent._capture_pool_world_anchor = lambda obs: None
    agent._record_and_update = lambda obs, action: None
    assert agent.act(obs) == int(Action.FINISH)


def test_empty_image_list_does_not_finish_immediately():
    agent = _make_agent()
    agent._image_goal_mode = True
    agent._goal_images = []
    assert agent._should_finish(_obs()) is False


def test_force_finish_not_applied_in_description_mode():
    agent = _make_agent()
    agent._target_mode = "all"
    agent.decision_loop = None
    agent._goal_found = {0}  # description 模式下此集合不应生效
    assert agent._should_finish(_obs(step=100)) is False


def test_missing_descriptions_retried_with_backoff():
    """论断 6 补全：描述缺失的 goal 每隔 query_interval 步只补缺失项
    重试，间隔内节流不再调用。"""
    vlm = _FakeVLM(None)  # 冷启动失败
    agent = _make_agent(vlm)
    obs = _obs(goal_type="image", goal_images=[_goal_rgb(1)])
    agent._last_observation = obs
    agent._capture_goal_images(obs)
    assert agent._goal_descriptions == [""]
    calls_before = len(vlm.chat_calls)
    # 间隔不足：节流，不重试
    agent._retry_goal_descriptions(_obs(step=5, goal_type="image"))
    assert len(vlm.chat_calls) == calls_before
    # 到达间隔：重试成功补齐
    vlm.description_reply = "goal_image_0: a brass wall clock"
    agent._retry_goal_descriptions(_obs(step=25, goal_type="image"))
    assert agent._goal_descriptions == ["a brass wall clock"]
