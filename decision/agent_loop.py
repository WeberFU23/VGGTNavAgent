"""事件驱动的具身 VLM harness。

VLM 通过工具读取和编辑 3D instance memory，再选择实例、frontier、扫描、
报告或结束。程序只校验 ID 和工具参数；底层跟随、避障与路径规划保持
确定性。VLM 不直接输出坐标或电机动作。

chat_fn(user_text, images) -> dict|None 可注入（生产接
VLMDecisionClient.agentic_chat，单测用 mock）。
"""

import json
import os
import threading
import time

ACTIONS = ("GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND", "SCAN",
           "EXPLORE", "FINISH")

# 写工具：成功执行后实例记忆已变化，动作校验前必须刷新 world-state。
WRITE_TOOLS = ("update_instance", "merge_instances", "undo_merge")

EVENT_ACTIONS = {
    "world_state_updated": {"GOTO_INSTANCE", "GOTO_FRONTIER", "EXPLORE"},
    "arrival": {"REPORT_FOUND", "SCAN", "EXPLORE"},
    "scan_complete": {"GOTO_INSTANCE", "GOTO_FRONTIER", "EXPLORE"},
    "finish_check": {"GOTO_INSTANCE", "GOTO_FRONTIER", "EXPLORE", "FINISH"},
}

DECIDER_PROMPT = """You are the reasoning core of an embodied multi-object
navigation harness. You receive a JSON world state (all distances and path
costs are precomputed — never estimate geometry yourself) and an annotated
top-down map image (white=free, black=obstacle, gray=unknown; green circles=
3D instances, crossed green circles=reported instances, purple crosses=
frontiers; numbers match ids in the state JSON).

Every instance is created from a VLM-pointed image pixel and its VGGT 3D point.
Its text is your editable working memory, not a fixed class label. Read evidence,
revise instance text when useful, and reason about uncertainty yourself.

The instance table is a bounded summary of unreported instances (nearest,
newest, task-related; table text may be truncated). instances_omitted_ids are
valid GOTO_INSTANCE targets not shown in the table; reported_instance_ids are
already reported. Use search_instances to find instances beyond the table and
inspect_instance for full text and evidence.

Available actions:
- GOTO_INSTANCE (target_id = an unreported instance id): the system resolves
  that instance's stored 3D point, plans an A* path, follows it with collision
  recovery, and triggers a new arrival decision near the point. This action
  does not assert that the instance matches the task.
- GOTO_FRONTIER (target_id = a frontier id): the system follows the precomputed
  path to that geometric exploration frontier, feeds new RGB frames into SLAM,
  and later refreshes instances and frontiers.
- REPORT_FOUND (target_id = null): valid only at arrival. The system starts
  target-centering/approach visual servo and emits the benchmark TARGET_FOUND
  signal only when the visible object is close and centered. Servo failure
  falls back to SCAN; it never reports from the stored coordinate alone.
- SCAN (target_id = null): valid only at arrival. The system performs a general
  360-degree panorama (12 left turns, four sampled views), keeps feeding SLAM,
  refreshes task-relevant instances with caption retrieval + pointing, then
  invokes scan_complete for a new global choice. It is not target verification.
- EXPLORE (target_id = null): leave any active target and continue geometric
  exploration. The instance remains in memory and can be selected again.
- FINISH (target_id = null): irreversibly end the episode. For an explicit
  many-count task, the system rejects FINISH until the required report count.

You may call one tool per reply, at most {max_rounds} times:
  {{"tool_call": {{"name": "search_captions", "text": "<search text>"}}}}
    Use when current instances are insufficient and historical image captions
    may reveal task-relevant places or objects. It searches image-caption memory,
    not 3D instances. Returns a JSON array of {{frame_id, score, caption}} rows.
  {{"tool_call": {{"name": "search_instances", "keywords": ["red", "cup"],
                     "reported": false, "top_k": 5}}}}
    Use to find existing 3D instances by concrete keywords you choose from the
    task or your reasoning. Matching is case-insensitive substring search over
    VLM-authored instance text; any keyword may match and more matches rank
    first. reported may be true, false, or null. Returns compact rows
    {{id, text, reported, matched_keywords, evidence_count, frame_ids}}.
  {{"tool_call": {{"name": "look_instance", "instance_id": <id>}}}}
    Use when an instance's text or metadata is insufficient for visual judgment.
    Returns no JSON data; on success the best available image for that instance
    is attached to your next input. The system prefers its pointing overlay and
    falls back to an associated keyframe. A missing instance/image returns
    {{"error": "instance image not found"}}. This is read-only.
  {{"tool_call": {{"name": "inspect_instance", "instance_id": <id>}}}}
    Use to examine the stored metadata and all evidence references before
    navigation, editing, or merging. Returns the full object {{id, point, text,
    reported, frame_id, candidate_id, evidence}}, or
    {{"error": "instance ... not found"}}. This is read-only and returns no image.
  {{"tool_call": {{"name": "update_instance", "instance_id": <id>,
                     "text": "<your revised memory text>"}}}}
    Use after interpreting new evidence to preserve your best current semantic
    understanding and uncertainty. Replaces only the instance text and returns
    the updated full instance. Geometry, evidence and reported state are fixed.
  {{"tool_call": {{"name": "merge_instances", "instance_ids": [<id>, ...],
                     "text": "<summary for the merged instance>"}}}}
    Use only after judging from text, metadata, and preferably images that
    two or more records are the same physical object. Merges them and
    returns the surviving full instance (smallest id); its point is the
    median, evidence is unioned, and reported is true if any input was
    reported. Other merged ids are removed. Invalid input returns an error.
    A merge can be reverted with undo_merge.
  {{"tool_call": {{"name": "undo_merge"}}}}
    Use when new evidence shows your most recent merge_instances was wrong.
    Restores the pre-merge records exactly as they were (a report that
    happened after the merge is never revoked). Returns {{"kept": ...,
    "restored": [...]}} or {{"error": "no merge to undo"}}.
All non-image tool failures return {{"error": "message"}}. Tool results are
included in your next prompt; after a write tool (update_instance or
merge_instances) the world state is regenerated and the refreshed version is
included in your next prompt — rely on it, not on the pre-write state.

Standard tool workflows:
- Existing-instance reasoning: search_instances -> inspect_instance and/or
  look_instance -> optionally update_instance or merge_instances -> final action.
- Historical-context reasoning: search_captions -> use caption clues to choose
  an instance, frontier, SCAN, or EXPLORE. search_captions does not create or
  modify an instance by itself.
- Do not call tools mechanically: stop as soon as supplied state and evidence
  are sufficient for a final action.

After tools (or immediately), reply with exactly one JSON object:
  {{"action": "GOTO_INSTANCE|GOTO_FRONTIER|REPORT_FOUND|SCAN|EXPLORE|FINISH",
    "target_id": "<id from the state tables, or null>",
    "reason": "short reason (log only)"}}
No markdown, no extra text."""


class DecisionResult:
    __slots__ = ("action", "target_id", "reason", "validation", "tool_calls")

    def __init__(self, action, target_id=None, reason="", validation="ok",
                 tool_calls=0):
        self.action = action
        self.target_id = target_id
        self.reason = reason
        self.validation = validation
        self.tool_calls = tool_calls

    def as_dict(self):
        return {"action": self.action, "target_id": self.target_id,
                "reason": self.reason, "validation": self.validation,
                "tool_calls": self.tool_calls}

    def __repr__(self):
        return (f"DecisionResult({self.action} target={self.target_id} "
                f"{self.validation})")


class DecisionTraceLogger:
    """决策 trace JSONL（时间步、输入摘要、输出、校验结果）。"""

    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record):
        record = dict(record)
        record.setdefault("t", time.strftime("%H:%M:%S"))
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass


class DecisionLoop:
    def __init__(self, chat_fn, tools=None, logger=None, max_tool_rounds=3):
        self.chat_fn = chat_fn
        self.tools = dict(tools or {})
        self.logger = logger
        self.max_tool_rounds = int(max_tool_rounds)

    # ------------------------------------------------------------------
    def decide(self, event, world_state, map_png=None, images=None,
               state_fn=None):
        """一次事件驱动决策。返回 DecisionResult；最终非法/模型不可用
        返回 None（调用方回退确定性规则）。

        state_fn: 可选无参回调，在写工具（update_instance/merge_instances）
        成功执行后调用，返回重新生成的 world-state dict；之后的动作校验、
        FINISH 强制与日志都基于刷新后的状态，避免 VLM 选择已被合并删除
        或状态改变的实例。"""
        state = world_state
        prompt = self._build_prompt(event, state)
        images = list(images or [])
        if map_png:
            images.append(("topdown_map", map_png))
        tool_calls = 0
        for _round in range(self.max_tool_rounds + 2):
            data = self._chat(prompt, images)
            if data is None:
                self._log(event, state, None, "model_unavailable",
                          tool_calls)
                return None
            tool_call = data.get("tool_call")
            if tool_call and tool_calls < self.max_tool_rounds:
                tool_calls += 1
                feedback, tool_img, ok = self._run_tool(tool_call)
                prompt += ("\n\nTool result:\n" + feedback
                           + "\nContinue with another tool call or reply with "
                             "the final decision JSON.")
                if tool_img:
                    images.append(("tool_instance_evidence", tool_img))
                if ok and state_fn is not None and \
                        str(tool_call.get("name") or "") in WRITE_TOOLS:
                    state = self._refresh_state(state_fn, state)
                    prompt += ("\n\nWorld state after your write:\n"
                               + json.dumps(state, ensure_ascii=False))
                continue
            result, err = self._validate(data, state, tool_calls, event)
            if result is not None:
                result = self._enforce_finish(result, state)
                self._log(event, state, result.as_dict(),
                          result.validation, tool_calls)
                return result
            # 校验失败重试一次
            prompt += ("\n\nYour previous output was rejected: " + err
                       + "\nReturn exactly one valid decision JSON object.")
            data2 = self._chat(prompt, images)
            result2, err2 = self._validate(data2, state, tool_calls, event) \
                if data2 is not None else (None, "model_unavailable")
            if result2 is not None:
                result2 = self._enforce_finish(result2, state)
                self._log(event, state, result2.as_dict(),
                          result2.validation, tool_calls)
                return result2
            self._log(event, state, None,
                      f"fallback: {err2}", tool_calls)
            return None
        self._log(event, state, None, "fallback: tool rounds exhausted",
                  tool_calls)
        return None

    @staticmethod
    def _refresh_state(state_fn, fallback):
        """写工具后重新生成 world-state；失败时保留调用前状态。"""
        try:
            refreshed = state_fn()
        except Exception:
            return fallback
        return refreshed if isinstance(refreshed, dict) else fallback

    # ------------------------------------------------------------------
    def _build_prompt(self, event, world_state):
        parts = [DECIDER_PROMPT.format(max_rounds=self.max_tool_rounds),
                 "\nEvent: " + str(event),
                 "\nWorld state:\n"
                 + json.dumps(world_state, ensure_ascii=False)]
        event_guidance = {
            "world_state_updated": (
                "\nInstances and reachable frontiers were refreshed together. "
                "Read and, when useful, update instance texts. Choose globally "
                "among GOTO_INSTANCE, GOTO_FRONTIER and EXPLORE."),
            "arrival": (
                "\nThe first extra image is current RGB; later images are "
                "historical evidence. Update the current instance text if your "
                "understanding changed. Use REPORT_FOUND when it satisfies the "
                "task, SCAN to gather broader environmental information, or "
                "EXPLORE to leave it unresolved."),
            "scan_complete": (
                "\nA general panoramic scan is complete. The images show the "
                "surrounding environment rather than a target verification "
                "sequence. Reconsider all refreshed instances and frontiers; "
                "choose GOTO_INSTANCE, GOTO_FRONTIER, or EXPLORE."),
            "finish_check": (
                "\nFINISH is irreversible. Inspect instance memory and task "
                "progress before deciding."),
        }
        if str(event) in event_guidance:
            parts.append(event_guidance[str(event)])
        task = world_state.get("task", {})
        # many 的明确数量是任务事实，提示 VLM 自行规划剩余工作。
        if task.get("mode") == "many" and task.get("expected") is not None \
                and task.get("found", 0) < task["expected"]:
            parts.append(
                "\nCounting hint: only "
                f"{task.get('found', 0)}/{task['expected']} required "
                "instances found. Inspect available instances and evidence, "
                "then choose where to navigate or explore.")
        return "".join(parts)

    def _chat(self, prompt, images):
        try:
            data = self.chat_fn(prompt, images)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _run_tool(self, tool_call):
        """执行工具，返回 (feedback_text, tool_image|None, ok)。"""
        name = str(tool_call.get("name") or "")
        fn = self.tools.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool: {name}"}), None, False
        try:
            if name == "search_captions":
                out = fn(str(tool_call.get("text") or ""))
                return json.dumps(out, ensure_ascii=False)[:4000], None, True
            if name == "look_instance":
                out = fn(tool_call.get("instance_id"))
                if not out:
                    return json.dumps(
                        {"error": "instance image not found"}), None, False
                return "Instance evidence image attached.", out, True
            out = fn(**{k: v for k, v in tool_call.items() if k != "name"})
            if isinstance(out, dict) and "error" in out:
                return json.dumps(out, ensure_ascii=False), None, False
            return json.dumps(out, ensure_ascii=False)[:4000], None, True
        except Exception as exc:
            return json.dumps({"error": str(exc)[:200]}), None, False

    # ------------------------------------------------------------------
    def _validate(self, data, world_state, tool_calls, event=None):
        """schema + id 存在性校验。返回 (DecisionResult|None, error)。"""
        if not isinstance(data, dict):
            return None, "not a JSON object"
        action = str(data.get("action") or "").strip().upper()
        if action not in ACTIONS:
            return None, f"unknown action: {action!r}"
        allowed = EVENT_ACTIONS.get(str(event))
        if allowed is not None and action not in allowed:
            return None, f"action {action!r} is invalid for event {event!r}"
        target_id = data.get("target_id")
        if target_id is not None:
            target_id = str(target_id)
        if action == "GOTO_INSTANCE":
            # 摘要表之外但未被报告的实例（omitted）同样是合法导航目标。
            valid = {str(i["id"]) for i in world_state.get("instances", [])
                     if not i.get("reported", False)}
            valid |= {str(i) for i in
                      world_state.get("instances_omitted_ids", [])}
            if target_id not in valid:
                return None, f"target_id {target_id!r} not an unreported instance"
        elif action == "GOTO_FRONTIER":
            valid = {str(f["id"]) for f in world_state.get("frontiers", [])}
            if target_id not in valid:
                return None, f"target_id {target_id!r} not a frontier"
        else:
            target_id = None
        return DecisionResult(action, target_id,
                              str(data.get("reason") or "")[:300],
                              tool_calls=tool_calls), None

    def _enforce_finish(self, result, world_state):
        """只强制 benchmark 明确给出的 many 数量；其他判断交给 VLM。"""
        if result.action != "FINISH":
            return result
        task = world_state.get("task", {})
        needs_count = task.get("mode") == "many" \
            and task.get("expected") is not None \
            and task.get("found", 0) < task["expected"]
        if not needs_count:
            return result
        instances = [item for item in world_state.get("instances", [])
                     if not item.get("reported", False)]
        if instances:
            return DecisionResult(
                "GOTO_INSTANCE", str(instances[0]["id"]),
                "Required count not reached; continue with instance memory. "
                + result.reason, validation="finish_downgraded",
                tool_calls=result.tool_calls)
        frontiers = world_state.get("frontiers", [])
        if frontiers:
            return DecisionResult(
                "GOTO_FRONTIER", str(frontiers[0]["id"]),
                "Required count not reached; continue exploring. "
                + result.reason,
                validation="finish_downgraded",
                tool_calls=result.tool_calls)
        return DecisionResult(
            "EXPLORE", None,
            "Required count not reached; continue exploring. " + result.reason,
            validation="finish_downgraded_no_target",
            tool_calls=result.tool_calls)

    def _log(self, event, world_state, output, validation, tool_calls):
        if self.logger is None:
            return
        self.logger.log({
            "step": world_state.get("step"),
            "event": str(event),
            "input_summary": {
                "instances": len(world_state.get("instances", [])),
                "frontiers": len(world_state.get("frontiers", [])),
                "task": world_state.get("task"),
                "termination": world_state.get("termination"),
            },
            "output": output,
            "validation": validation,
            "tool_calls": tool_calls,
        })
