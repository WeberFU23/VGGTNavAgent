"""Agentic 决策循环 + 受约束输出（Phase 4c/4d/4e，agent 端）。

VLM 决策层接管高层规划，替代/增强纯规则的 _should_finish 与无先验的
frontier 打分。事件驱动触发（到达/新实例确认/扫描完成/frontier 耗尽/
VERIFY 返回），不是每步调；最多 max_tool_rounds 轮只读工具调用后必须
下决策；输出走 JSON schema 校验，FINISH 硬条件由状态机强制，非法输出
回退确定性规则。决策层不进控制回路：底层跟随/避障/重规划永远确定性。

chat_fn(user_text, images) -> dict|None 可注入（生产接
VLMDecisionClient.agentic_chat，单测用 mock）。
"""

import json
import os
import threading
import time

ACTIONS = ("GOTO_INSTANCE", "GOTO_FRONTIER", "VERIFY", "FINISH")

DECIDER_PROMPT = """You are the high-level planner of an embodied multi-object
navigation agent. You receive a JSON world state (all distances and path
costs are precomputed — never estimate geometry yourself) and an annotated
top-down map image (white=free, black=obstacle, gray=unknown; green solid
circles=confirmed instances, orange dashed=unverified beliefs, purple
crosses=frontiers; numbers match the ids in the state JSON).

Decide the next high-level action:
- GOTO_INSTANCE: navigate to a confirmed, unvisited instance (target_id =
  instance id from the instances table).
- GOTO_FRONTIER: explore a frontier (target_id = frontier id).
- VERIFY: re-verify the current arrival target visually.
- FINISH: end the episode (only when every required target is found and
  exploration is sufficiently complete).

You may first call read-only tools, one per reply, at most {max_rounds} times:
  {{"tool_call": {{"name": "query_memory", "text": "<search text>"}}}}
  -> returns top-K semantic memory captions.
  {{"tool_call": {{"name": "look_at", "frame_id": <id>}}}}
  -> returns the keyframe image.
After tools (or immediately), reply with exactly one JSON object:
  {{"action": "GOTO_INSTANCE|GOTO_FRONTIER|VERIFY|FINISH",
    "target_id": "<id from the state tables, or null>",
    "reason": "short reason (log only)",
    "confidence": 0.0}}
No markdown, no extra text."""


class DecisionResult:
    __slots__ = ("action", "target_id", "reason", "confidence",
                 "validation", "tool_calls")

    def __init__(self, action, target_id=None, reason="", confidence=0.0,
                 validation="ok", tool_calls=0):
        self.action = action
        self.target_id = target_id
        self.reason = reason
        self.confidence = confidence
        self.validation = validation
        self.tool_calls = tool_calls

    def as_dict(self):
        return {"action": self.action, "target_id": self.target_id,
                "reason": self.reason, "confidence": self.confidence,
                "validation": self.validation,
                "tool_calls": self.tool_calls}

    def __repr__(self):
        return (f"DecisionResult({self.action} target={self.target_id} "
                f"conf={self.confidence:.2f} {self.validation})")


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
    def __init__(self, chat_fn, tools=None, logger=None, max_tool_rounds=3,
                 finish_unexplored_max=0.15):
        self.chat_fn = chat_fn
        self.tools = dict(tools or {})
        self.logger = logger
        self.max_tool_rounds = int(max_tool_rounds)
        self.finish_unexplored_max = float(finish_unexplored_max)

    # ------------------------------------------------------------------
    def decide(self, event, world_state, map_png=None):
        """一次事件驱动决策。返回 DecisionResult；最终非法/模型不可用
        返回 None（调用方回退确定性规则）。"""
        prompt = self._build_prompt(event, world_state)
        images = [map_png] if map_png else []
        tool_calls = 0
        for _round in range(self.max_tool_rounds + 2):
            data = self._chat(prompt, images)
            if data is None:
                self._log(event, world_state, None, "model_unavailable",
                          tool_calls)
                return None
            tool_call = data.get("tool_call")
            if tool_call and tool_calls < self.max_tool_rounds:
                tool_calls += 1
                feedback, tool_img = self._run_tool(tool_call)
                prompt += ("\n\nTool result:\n" + feedback
                           + "\nNow decide. Reply with the decision JSON "
                             "or another tool_call.")
                if tool_img:
                    images = [map_png] if map_png else []
                    images.append(tool_img)
                continue
            result, err = self._validate(data, world_state, tool_calls)
            if result is not None:
                result = self._enforce_finish(result, world_state)
                self._log(event, world_state, result.as_dict(),
                          result.validation, tool_calls)
                return result
            # 校验失败重试一次
            prompt += ("\n\nYour previous output was rejected: " + err
                       + "\nReturn exactly one valid decision JSON object.")
            data2 = self._chat(prompt, images)
            result2, err2 = self._validate(data2, world_state, tool_calls) \
                if data2 is not None else (None, "model_unavailable")
            if result2 is not None:
                result2 = self._enforce_finish(result2, world_state)
                self._log(event, world_state, result2.as_dict(),
                          result2.validation, tool_calls)
                return result2
            self._log(event, world_state, None,
                      f"fallback: {err2}", tool_calls)
            return None
        self._log(event, world_state, None, "fallback: tool rounds exhausted",
                  tool_calls)
        return None

    # ------------------------------------------------------------------
    def _build_prompt(self, event, world_state):
        parts = [DECIDER_PROMPT.format(max_rounds=self.max_tool_rounds),
                 "\nEvent: " + str(event),
                 "\nWorld state:\n"
                 + json.dumps(world_state, ensure_ascii=False)]
        task = world_state.get("task", {})
        # 4e: many 模式计数校验——簇数不足时提示在已确认实例周边补探索
        if task.get("mode") == "many" and task.get("expected") is not None \
                and task.get("found", 0) < task["expected"]:
            parts.append(
                "\nCounting hint: only "
                f"{task.get('found', 0)}/{task['expected']} required "
                "instances found. Prefer exploring AROUND confirmed "
                "instances of the same category (same-object clusters) "
                "before jumping to distant new frontiers.")
        return "".join(parts)

    def _chat(self, prompt, images):
        try:
            data = self.chat_fn(prompt, images)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _run_tool(self, tool_call):
        name = str(tool_call.get("name") or "")
        fn = self.tools.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool: {name}"}), None
        try:
            if name == "query_memory":
                out = fn(str(tool_call.get("text") or ""))
                return json.dumps(out, ensure_ascii=False)[:4000], None
            if name == "look_at":
                out = fn(tool_call.get("frame_id"))
                if not out:
                    return json.dumps({"error": "frame not found"}), None
                return "Keyframe image attached.", out
            out = fn(**{k: v for k, v in tool_call.items() if k != "name"})
            return json.dumps(out, ensure_ascii=False)[:4000], None
        except Exception as exc:
            return json.dumps({"error": str(exc)[:200]}), None

    # ------------------------------------------------------------------
    def _validate(self, data, world_state, tool_calls):
        """schema + id 存在性校验。返回 (DecisionResult|None, error)。"""
        if not isinstance(data, dict):
            return None, "not a JSON object"
        action = str(data.get("action") or "").strip().upper()
        if action not in ACTIONS:
            return None, f"unknown action: {action!r}"
        target_id = data.get("target_id")
        if target_id is not None:
            target_id = str(target_id)
        if action == "GOTO_INSTANCE":
            valid = {str(i["id"]) for i in world_state.get("instances", [])
                     if i.get("status") == "confirmed"}
            if target_id not in valid:
                return None, f"target_id {target_id!r} not a confirmed instance"
        elif action == "GOTO_FRONTIER":
            valid = {str(f["id"]) for f in world_state.get("frontiers", [])}
            if target_id not in valid:
                return None, f"target_id {target_id!r} not a frontier"
        else:
            target_id = None
        try:
            conf = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        return DecisionResult(action, target_id,
                              str(data.get("reason") or "")[:300], conf,
                              tool_calls=tool_calls), None

    def _enforce_finish(self, result, world_state):
        """FINISH 硬条件（状态机侧强制）：未探索占比低于阈值 且 无未复核
        高置信锚点；many 模式另需簇数 >= N。不满足自动降级为继续探索。"""
        if result.action != "FINISH":
            return result
        term = world_state.get("termination", {})
        task = world_state.get("task", {})
        unexplored = term.get("unexplored_ratio")
        unresolved = term.get("unresolved_anchor_count", 1)
        ok = unexplored is not None \
            and unexplored < self.finish_unexplored_max \
            and unresolved == 0
        if task.get("mode") == "many" and task.get("expected") is not None:
            ok = ok and task.get("found", 0) >= task["expected"]
        if ok:
            return result
        frontiers = world_state.get("frontiers", [])
        if frontiers:
            return DecisionResult(
                "GOTO_FRONTIER", str(frontiers[0]["id"]),
                "FINISH 硬条件不满足，降级继续探索: " + result.reason,
                result.confidence, validation="finish_downgraded",
                tool_calls=result.tool_calls)
        return DecisionResult(
            "GOTO_FRONTIER", None,
            "FINISH 硬条件不满足且无 frontier: " + result.reason,
            result.confidence, validation="finish_downgraded_no_frontier",
            tool_calls=result.tool_calls)

    def _log(self, event, world_state, output, validation, tool_calls):
        if self.logger is None:
            return
        self.logger.log({
            "step": world_state.get("step"),
            "event": str(event),
            "input_summary": {
                "instances": len(world_state.get("instances", [])),
                "anchors": len(world_state.get("belief_anchors", [])),
                "frontiers": len(world_state.get("frontiers", [])),
                "task": world_state.get("task"),
                "termination": world_state.get("termination"),
            },
            "output": output,
            "validation": validation,
            "tool_calls": tool_calls,
        })
