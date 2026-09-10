"""事件驱动的具身 VLM harness。

VLM 通过工具读取和编辑 3D instance memory，再选择实例、frontier、扫描、
报告或结束。底层跟随、避障与路径规划保持确定性；VLM 只有在自己
显式进入 adjustment 状态后，才能每轮输出一个白名单原子动作。

chat_fn(user_text, images, system_prompt=None) -> dict|None 可注入（生产接
VLMDecisionClient.agentic_chat，单测用 mock）。
VLMDecisionClient.agentic_chat，单测用 mock）。
"""

import json
import os
import threading
import time
import uuid

from decision.prompts import (build_decider_system, build_decision_prompt,
                              build_final_decision_prompt)
from decision.trace import snapshot

ACTIONS = ("GOTO_INSTANCE", "GOTO_FRONTIER",
           "REPORT_FOUND", "SCAN", "EXPLORE", "FINISH", "START_ADJUST",
           "END_ADJUST", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
           "LOOK_UP", "LOOK_DOWN")

DEFAULT_MAX_TOOL_ROUNDS = 15
FINAL_ACTION_ATTEMPTS = 2
# 工具结果统一截断口径：prompt transcript 与 decision_window exchange 共用
TOOL_RESULT_MAX_CHARS = int(os.environ.get("NAV_TOOL_RESULT_MAX_CHARS",
                                           "2000"))


# 走到目标附近即可上报：成功按距离判定（评估器 0.25m 测地线阈值），
# 不要求目标在视野内——太近时物体落在相机视野外是正常现象。非 active
# 实例只要 dist_m ≤ 该值也放行 REPORT_FOUND。
REPORT_NEAR_DIST_M = float(os.environ.get("NAV_REPORT_NEAR_DIST_M", "1.0"))

# 写工具：成功执行后世界状态已变化，动作校验前必须刷新 world-state。
WRITE_TOOLS = ("update_instance", "update_notes", "instantiate_points",
               "commit_candidates", "resolve_duplicate",
               "resolve_goal_conflict")

EVENT_ACTIONS = {
    # 除 finish_check / adjustment 外放行高层动作（EXPLORE 除外——VLM 滥用
    # 一键探索；探索应显式选 frontier 或用 START_ADJUST 局部观察）。
    # EXPLORE 保留在 ACTIONS 中供 harness 内部降级（_enforce_finish 兜底）。
    "world_state_updated": {"GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND",
                            "SCAN", "FINISH", "START_ADJUST"},
    "arrival": {"GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND",
                "SCAN", "FINISH", "START_ADJUST"},
    "scan_complete": {"GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND",
                      "SCAN", "FINISH", "START_ADJUST"},
    # 导航卡死（连续碰撞无法到达目标）：与 arrival 同集——当前 RGB 已附加，
    # VLM 换目标/探索/局部调整；REPORT_FOUND 仅在能直接确认时使用。
    "nav_failed": {"GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND",
                   "SCAN", "FINISH", "START_ADJUST"},
    "finish_check": {"GOTO_INSTANCE", "GOTO_FRONTIER", "FINISH"},
    "adjustment": {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP",
                   "LOOK_DOWN", "END_ADJUST"},
}

class DecisionResult:
    __slots__ = ("action", "target_id", "reason", "validation", "tool_calls",
                 "steps")

    def __init__(self, action, target_id=None, reason="", validation="ok",
                 tool_calls=0, steps=None):
        self.action = action
        self.target_id = target_id
        self.reason = reason
        self.validation = validation
        self.tool_calls = tool_calls
        self.steps = steps

    def as_dict(self):
        return {"action": self.action, "target_id": self.target_id,
                "reason": self.reason, "validation": self.validation,
                "tool_calls": self.tool_calls, "steps": self.steps}

    def __repr__(self):
        steps = f" steps={self.steps}" if self.steps is not None else ""
        return (f"DecisionResult({self.action} target={self.target_id}"
                f"{steps} {self.validation})")


class DecisionTraceLogger:
    """决策 trace JSONL（时间步、输入摘要、输出、校验结果）。"""

    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._warned = False

    def log(self, record):
        record = dict(record)
        record.setdefault("t", time.strftime("%H:%M:%S"))
        with self._lock:
            try:
                with open(self.path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(
                        record, ensure_ascii=False, default=str) + "\n")
            except OSError as exc:
                if not self._warned:
                    print(f"[DecisionTraceLogger] 无法写入 {self.path}: {exc}",
                          flush=True)
                    self._warned = True


class DecisionLoop:
    def __init__(self, chat_fn, tools=None, logger=None,
                 max_tool_rounds=DEFAULT_MAX_TOOL_ROUNDS):
        self.chat_fn = chat_fn
        self.tools = dict(tools or {})
        self.logger = logger
        self.max_tool_rounds = min(
            DEFAULT_MAX_TOOL_ROUNDS, max(0, int(max_tool_rounds)))
        # 决策级 API 重试次数（chat_fn 返回 None/异常时），默认 1 次
        self.api_retries = max(
            0, int(os.environ.get("NAV_DECIDER_API_RETRIES", "1")))
        self._trace_session = uuid.uuid4().hex
        self._trace_sequence = 0
        self._trace_record = None
        self._trace_snapshot_fn = None
        self._trace_sink = None
        self._trace_call_context = None
        self._trace_warning = False
        # 最近一次被接受决策的原始交换记录（tool_calls/tool_results/reply/
        # image_labels），供 nav_agent 追加到 decision_window。
        self.last_exchange = None

    def decide(self, event, world_state, map_png=None, images=None,
               state_fn=None, trace_context=None, trace_snapshot_fn=None,
               trace_sink=None, trace_call_context=None,
               decision_window=None):
        """一次事件驱动决策。返回 DecisionResult；最终非法/模型不可用
        返回 None（调用方回退确定性规则）。

        state_fn: 可选无参回调，在写工具成功后调用。可返回重新生成的
        world-state dict，或 (world-state, map_png)；后者会同时替换旧地图。"""
        self._trace_sequence += 1
        self._trace_record = {
            "schema_version": 2,
            "decision_id": f"{self._trace_session}:{self._trace_sequence}",
            "context": snapshot(trace_context or {}),
            "model_outputs": [], "tools": [], "capture_errors": [],
            "window_size": len(decision_window or []),
        }
        self.last_exchange = None
        self._trace_snapshot_fn = trace_snapshot_fn
        self._trace_sink = trace_sink
        self._trace_call_context = trace_call_context
        state = world_state
        # image-goal 模式使用独立重写的系统提示词（清单式任务契约）；
        # description 模式的提示词逐字节不变。
        task = world_state.get("task", {}) if isinstance(world_state, dict) \
            else {}
        self._system_prompt = build_decider_system(
            self.max_tool_rounds,
            image_mode=str(task.get("goal_type", "")).lower() == "image")
        images = list(images or [])
        if map_png:
            images = self._with_topdown_map(images, map_png)
        tool_calls = 0
        tool_results = []
        # decision_window 条目素材：本轮工具调用/结果（截断）、最终被接受
        # 的 reply 与当时附加图片的标签；校验重试的中间废品不进。
        exchange = {"tool_calls": [], "tool_results": [],
                    "image_labels": [label for label, _p in images]}
        while True:
            # 每轮重渲染 user 文本：world_state 只留最新一份，工具往来与
            # 校验拒绝经 transcript 滚动保留，不再字符串累加。
            prompt = self._render_user_prompt(
                event, state, decision_window, tool_results, tool_calls)
            data = self._chat(prompt, images)
            if data is None:
                self._log(event, state, None, "model_unavailable",
                          tool_calls)
                return None
            tool_call = self._extract_tool_call(data)
            if tool_call and str(event) == "adjustment":
                # takeover 期间禁止工具调用：按非法输出走校验失败重试路径。
                result, err = None, "tools are disabled during adjustment"
            elif tool_call:
                if tool_calls >= self.max_tool_rounds:
                    return self._finalize_after_tool_limit(
                        event, state, images, tool_calls, tool_results,
                        exchange)
                tool_calls += 1
                feedback, tool_img, ok = self._run_tool(tool_call)
                tool_name = str(tool_call.get("name") or "")
                tool_results.append(
                    f"Tool {tool_calls}/{self.max_tool_rounds} "
                    f"({tool_name}) result:\n{feedback}")
                exchange["tool_calls"].append({
                    "name": tool_name,
                    "args": {k: v for k, v in tool_call.items()
                             if k != "name"}})
                exchange["tool_results"].append(
                    str(feedback)[:TOOL_RESULT_MAX_CHARS])
                if tool_img:
                    for label, payload in tool_img:
                        images = self._with_tool_image(images, label, payload)
                        exchange["image_labels"].append(label)
                if ok and state_fn is not None and \
                        str(tool_call.get("name") or "") in WRITE_TOOLS:
                    state, refreshed_map, has_map = self._refresh_context(
                        state_fn, state)
                    if has_map:
                        images = self._with_topdown_map(
                            images, refreshed_map)
                if tool_calls >= self.max_tool_rounds:
                    return self._finalize_after_tool_limit(
                        event, state, images, tool_calls, tool_results,
                        exchange)
                continue
            else:
                result, err = self._validate(data, state, tool_calls, event)
            if result is not None:
                result = self._enforce_finish(result, state)
                self.last_exchange = {
                    **exchange,
                    "reply": json.dumps(data, ensure_ascii=False,
                                        default=str)}
                self._log(event, state, result.as_dict(),
                          result.validation, tool_calls)
                return result
            # 校验失败重试一次：拒绝原因进 transcript，重渲染后重试
            tool_results.append(
                "Your previous output was rejected: " + str(err)
                + "\nReturn exactly one valid decision JSON object.")
            prompt = self._render_user_prompt(
                event, state, decision_window, tool_results, tool_calls)
            data2 = self._chat(prompt, images)
            if data2 is None:
                result2, err2 = None, "model_unavailable"
            elif self._extract_tool_call(data2):
                result2, err2 = None, (
                    "a final action JSON was required; tool_call is not "
                    "allowed on validation retry")
            else:
                result2, err2 = self._validate(
                    data2, state, tool_calls, event)
            if result2 is not None:
                result2 = self._enforce_finish(result2, state)
                self.last_exchange = {
                    **exchange,
                    "reply": json.dumps(data2, ensure_ascii=False,
                                        default=str)}
                self._log(event, state, result2.as_dict(),
                          result2.validation, tool_calls)
                return result2
            self._log(event, state, None,
                      f"fallback: {err2}", tool_calls)
            return None

    def _finalize_after_tool_limit(self, event, state, images, tool_calls,
                                   tool_results, exchange=None):
        """Require an action after the hard limit; never validate tool_call.

        If the model still refuses the final-action-only contract, return a
        deterministic valid navigation decision rather than leaking a residual
        tool_call into action validation or returning None.
        """
        allowed = EVENT_ACTIONS.get(
            str(event), set(ACTIONS) - {"EXPLORE"})
        prompt = build_final_decision_prompt(
            event, state, tool_results, self.max_tool_rounds, allowed)
        last_error = "final action not produced"
        for _attempt in range(FINAL_ACTION_ATTEMPTS):
            data = self._chat(prompt, images)
            if data is None:
                last_error = "model_unavailable"
            elif self._extract_tool_call(data):
                last_error = (
                    "tool_call is disabled after the hard limit; output one "
                    "final action JSON")
            else:
                result, last_error = self._validate(
                    data, state, tool_calls, event)
                if result is not None:
                    result = self._enforce_finish(result, state)
                    self.last_exchange = {
                        **(exchange or {}),
                        "reply": json.dumps(data, ensure_ascii=False,
                                            default=str)}
                    self._log(event, state, result.as_dict(),
                              result.validation, tool_calls)
                    return result
            prompt += (
                "\n\nRejected final response: " + str(last_error)
                + "\nTools remain disabled. Return exactly one executable "
                  "final action JSON now.")
        result = self._forced_navigation_result(
            event, state, tool_calls, last_error)
        result = self._enforce_finish(result, state)
        self.last_exchange = {**(exchange or {}), "reply": None}
        self._log(event, state, result.as_dict(),
                  result.validation, tool_calls)
        return result

    @staticmethod
    def _forced_navigation_result(event, world_state, tool_calls, error):
        """Choose a valid progress action if final-only VLM replies stay invalid."""
        allowed = EVENT_ACTIONS.get(str(event), set(ACTIONS) - {"EXPLORE"})
        if "GOTO_INSTANCE" in allowed:
            unreachable = {str(i) for i in
                           world_state.get("instances_unreachable_ids", [])}
            instances = [item for item in world_state.get("instances", [])
                         if not item.get("reported", False)]
            if instances:
                return DecisionResult(
                    "GOTO_INSTANCE", str(instances[0]["id"]),
                    "Forced final action after tool limit: " + str(error),
                    validation="forced_after_tool_limit",
                    tool_calls=tool_calls)
            omitted = [i for i in world_state.get("instances_omitted_ids", [])
                       if str(i) not in unreachable]
            if omitted:
                return DecisionResult(
                    "GOTO_INSTANCE", str(omitted[0]),
                    "Forced final action after tool limit: " + str(error),
                    validation="forced_after_tool_limit",
                    tool_calls=tool_calls)
        if "GOTO_FRONTIER" in allowed:
            frontiers = world_state.get("frontiers", [])
            if frontiers:
                return DecisionResult(
                    "GOTO_FRONTIER", str(frontiers[0]["id"]),
                    "Forced final action after tool limit: " + str(error),
                    validation="forced_after_tool_limit",
                    tool_calls=tool_calls)
        for action in ("SCAN", "START_ADJUST", "FINISH", "END_ADJUST",
                       "TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "LOOK_UP",
                       "LOOK_DOWN"):
            if action in allowed:
                return DecisionResult(
                    action, None,
                    "Forced final action after tool limit: " + str(error),
                    validation="forced_after_tool_limit",
                    tool_calls=tool_calls)
        return DecisionResult(
            "FINISH", None,
            "Forced final action after tool limit: " + str(error),
            validation="forced_after_tool_limit", tool_calls=tool_calls)

    @staticmethod
    def _with_topdown_map(images, map_png):
        """Keep the map inside the VLM image budget.

        Current RGB remains first; the map is inserted immediately after it so
        panorama or tool evidence cannot push spatial context past
        NAV_VLM_MAX_IMAGES. Passing an empty map removes the previous map.
        """
        images = [(name, value) for name, value in images
                  if name != "topdown_map"]
        if not map_png:
            return images
        insert_at = 1 if images and images[0][0] == "current_observation" else 0
        images.insert(insert_at, ("topdown_map", map_png))
        return images

    @staticmethod
    def _with_tool_image(images, label, payload):
        """Replace an identical tool attachment instead of accumulating it."""
        images = [(name, value) for name, value in images if name != label]
        images.append((label, payload))
        return images

    @staticmethod
    def _refresh_context(state_fn, fallback):
        """写工具后刷新状态及可选地图；失败时保留调用前上下文。"""
        try:
            refreshed = state_fn()
        except Exception:
            return fallback, None, False
        if isinstance(refreshed, tuple) and len(refreshed) == 2:
            state, map_png = refreshed
            if isinstance(state, dict):
                return state, map_png, True
            return fallback, None, False
        if isinstance(refreshed, dict):
            return refreshed, None, False
        return fallback, None, False

    @staticmethod
    def _extract_tool_call(data):
        """归一化工具调用：约定 {"tool_call": {"name", ...}}；兼容部分模型
        输出的 {"tool": name, "arguments": {...}} 以及 tool_call 内嵌套
        arguments/args/parameters 的格式。"""
        tool_call = data.get("tool_call")
        if tool_call is None and isinstance(data.get("tool"), str):
            tool_call = {"name": data["tool"]}
            for key in ("arguments", "args", "parameters"):
                if isinstance(data.get(key), dict):
                    tool_call.update(data[key])
                    break
        if isinstance(tool_call, dict):
            for key in ("arguments", "args", "parameters"):
                nested = tool_call.get(key)
                if isinstance(nested, dict):
                    tool_call = {k: v for k, v in tool_call.items()
                                 if k != key}
                    tool_call.update(nested)
                    break
        return tool_call

    def _build_prompt(self, event, world_state, decision_window=None):
        return build_decision_prompt(
            event, world_state, max_tool_rounds=self.max_tool_rounds,
            decision_window=decision_window)

    def _render_user_prompt(self, event, state, decision_window,
                            tool_results, tool_calls):
        """每轮重渲染 user 文本：world_state 只留最新一份，工具往来与校验
        拒绝作为 transcript 一节滚动保留——user 文本不再只增不减。"""
        prompt = self._build_prompt(event, state, decision_window)
        if tool_results:
            remaining = max(0, self.max_tool_rounds - tool_calls)
            prompt += (
                "\n\nTool transcript this decision (oldest first):\n"
                + "\n\n".join(str(item) for item in tool_results)
                + f"\nTool usage: {tool_calls}/{self.max_tool_rounds}; "
                  f"{remaining} calls remain."
                + ("\nContinue with another tool call only if needed, "
                   "otherwise reply with the final decision JSON."
                   if remaining else
                   "\nThe hard tool-call limit has been reached."))
        return prompt

    def _chat(self, prompt, images):
        """单次决策内 API 调用；返回 None/异常时按 api_retries 决策级重试。"""
        record = self._trace_record
        data = None
        for _retry in range(1 + getattr(self, "api_retries", 0)):
            call_index = len(record["model_outputs"]) + 1 if record else None
            if record and self._trace_call_context is not None:
                try:
                    self._trace_call_context(record["decision_id"], call_index)
                except Exception as exc:
                    self._trace_warn(exc)
            try:
                data = self.chat_fn(
                    prompt, images,
                    system_prompt=getattr(self, "_system_prompt", None))
            except Exception:
                data = None
            if record is not None:
                try:
                    record["model_outputs"].append({
                        "call_index": call_index,
                        "output": snapshot(data)})
                except Exception as exc:
                    self._trace_warn(exc)
            if isinstance(data, dict):
                break
        return data if isinstance(data, dict) else None

    @staticmethod
    def _serialize_tool_feedback(payload, max_chars=TOOL_RESULT_MAX_CHARS):
        """Keep feedback valid JSON even when a tool returns a large record."""
        raw = json.dumps(payload, ensure_ascii=False, default=str)
        if len(raw) <= max_chars:
            return raw
        compact = {
            "ok": payload.get("ok", False),
            "tool": payload.get("tool"),
            "state_changed": payload.get("state_changed", False),
            "truncated": True,
            "result_preview": raw[:max_chars - 300],
        }
        return json.dumps(compact, ensure_ascii=False, default=str)

    @classmethod
    def _tool_error(cls, name, code, message):
        payload = {
            "ok": False,
            "tool": name,
            "state_changed": False,
            "error": {"code": code, "message": str(message)[:300]},
        }
        return cls._serialize_tool_feedback(payload), None, False

    @staticmethod
    def _tool_image_label(kind, value):
        safe = "".join(
            char if char.isalnum() or char in "-_." else "-"
            for char in str(value))
        return f"tool_{kind}_{safe or 'unknown'}"

    def _run_tool(self, tool_call):
        # Preserve the complete structured result before prompt truncation.
        self._trace_tool_payload = None
        result = self._execute_tool(tool_call)
        if self._trace_record is not None:
            try:
                self._trace_record["tools"].append({
                    "call_index": len(self._trace_record["model_outputs"]),
                    "request": snapshot(tool_call),
                    "response": (self._trace_tool_payload
                                 if self._trace_tool_payload is not None
                                 else snapshot(json.loads(result[0]))),
                    "image_labels": [item[0] for item in (result[1] or [])],
                })
            except Exception as exc:
                self._trace_warn(exc)
        return result

    def _trace_feedback(self, payload):
        try:
            self._trace_tool_payload = snapshot(payload)
        except Exception as exc:
            self._trace_warn(exc)
        return self._serialize_tool_feedback(payload)

    def _execute_tool(self, tool_call):
        """执行工具，返回 (统一 JSON, [(label, bytes)]|None, ok)。

        工具返回 dict 中的 "_tool_images"（[[label, bytes], ...] 拒绝证据图）
        会被弹出转交图像通道，不进入 JSON 反馈。"""
        name = str(tool_call.get("name") or "")
        fn = self.tools.get(name)
        if fn is None:
            return self._tool_error(
                name, "UNKNOWN_TOOL", f"unknown tool: {name}")
        try:
            if name == "view_instance":
                out = fn(tool_call.get("instance_id"))
                if not out:
                    return self._tool_error(
                        name, "IMAGE_NOT_FOUND", "instance image not found")
                iid = tool_call.get("instance_id")
                label = self._tool_image_label(
                    "instance", f"{iid}_evidence")
                payload = {
                    "ok": True, "tool": name, "state_changed": False,
                    "result": {"instance_id": iid, "image_ref": label},
                }
                return (self._trace_feedback(payload),
                        [(label, out)], True)
            if name == "view_frame":
                out = fn(tool_call.get("frame_id"))
                if not out:
                    return self._tool_error(
                        name, "IMAGE_NOT_FOUND", "frame image not found")
                fid = tool_call.get("frame_id")
                label = self._tool_image_label("frame", f"{fid}_rgb")
                payload = {
                    "ok": True, "tool": name, "state_changed": False,
                    "result": {"frame_id": fid, "image_ref": label},
                }
                return (self._trace_feedback(payload),
                        [(label, out)], True)
            out = fn(**{k: v for k, v in tool_call.items() if k != "name"})
            if isinstance(out, dict) and "error" in out:
                error = out["error"]
                if isinstance(error, dict):
                    return self._tool_error(
                        name, error.get("code", "TOOL_ERROR"),
                        error.get("message", error))
                return self._tool_error(name, "TOOL_ERROR", error)
            tool_images = None
            if isinstance(out, dict):
                tool_images = out.pop("_tool_images", None) or None
            payload = {
                "ok": True,
                "tool": name,
                "state_changed": name in WRITE_TOOLS,
                "result": out,
            }
            return (self._trace_feedback(payload),
                    tool_images, True)
        except Exception as exc:
            return self._tool_error(name, "TOOL_EXCEPTION", exc)

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
        if str(event) == "adjustment" and action in {"LOOK_UP", "LOOK_DOWN"}:
            adjustment = world_state.get("adjustment", {})
            offset = int(adjustment.get("pitch_offset_steps", 0) or 0)
            max_offset = max(
                0, int(adjustment.get("max_pitch_offset_steps", 1) or 0))
            next_offset = offset + (1 if action == "LOOK_UP" else -1)
            if abs(next_offset) > max_offset:
                return None, (
                    f"{action} would exceed the camera pitch limit "
                    f"(+/-{max_offset} steps); reverse pitch or choose another "
                    "action")
        steps = None
        if str(event) == "adjustment" and action == "MOVE_FORWARD":
            adjustment = world_state.get("adjustment", {})
            max_forward = max(1, int(
                adjustment.get("max_forward_steps", 8) or 8))
            raw_steps = data.get("steps")
            if raw_steps is None:
                return None, (
                    "MOVE_FORWARD requires \"steps\": how many forward steps "
                    f"to execute (1..{max_forward}, one step = 0.25m)")
            try:
                steps = int(raw_steps)
            except (TypeError, ValueError):
                return None, (
                    f"MOVE_FORWARD steps must be an integer, got {raw_steps!r}")
            if not 1 <= steps <= max_forward:
                return None, (
                    f"MOVE_FORWARD steps must be within 1..{max_forward}, "
                    f"got {steps}")
        target_id = data.get("target_id")
        if target_id is not None:
            target_id = str(target_id)
        if action == "GOTO_INSTANCE":
            # 摘要表之外但未被报告的实例（omitted）同样是合法导航目标；
            # 导航确认不可达的实例不再作为导航目标（但 REPORT_FOUND 可用）。
            valid = {str(i["id"]) for i in world_state.get("instances", [])
                     if not i.get("reported", False)}
            valid |= {str(i) for i in
                      world_state.get("instances_omitted_ids", [])}
            valid -= {str(i) for i in
                      world_state.get("instances_unreachable_ids", [])}
            if target_id not in valid:
                return None, f"target_id {target_id!r} not an unreported instance"
        elif action == "GOTO_FRONTIER":
            valid = {str(f["id"]) for f in world_state.get("frontiers", [])}
            if target_id not in valid:
                return None, f"target_id {target_id!r} not a frontier"
        elif action == "REPORT_FOUND":
            valid = {str(i["id"]) for i in world_state.get("instances", [])}
            valid |= {str(i) for i in
                      world_state.get("instances_omitted_ids", [])}
            if target_id not in valid:
                return None, (f"REPORT_FOUND target_id {target_id!r} is not "
                              "an available canonical instance")
            active = (world_state.get("navigation", {})
                      .get("active_target") or {})
            is_active = (active.get("type") == "instance" and
                         str(active.get("id")) == target_id)
            if not is_active:
                # 走到目标附近即可上报：距离判定，与视野无关。
                near = {str(i["id"]): i.get("dist_m")
                        for i in world_state.get("instances", [])}
                d = near.get(target_id)
                if d is None or d > REPORT_NEAR_DIST_M:
                    return None, ("REPORT_FOUND must target the active "
                                  "canonical instance (or one you are "
                                  f"within {REPORT_NEAR_DIST_M:g}m of)")
        else:
            target_id = None
        return DecisionResult(action, target_id,
                              str(data.get("reason") or "")[:300],
                              tool_calls=tool_calls, steps=steps), None

    def _enforce_finish(self, result, world_state):
        """强制终止的硬条件：many 数量未达、image-goal 仍有未找到的
        目标时拒绝 FINISH；其他判断交给 VLM。"""
        if result.action != "FINISH":
            return result
        task = world_state.get("task", {})
        needs_count = task.get("mode") == "many" \
            and task.get("expected") is not None \
            and task.get("found", 0) < task["expected"]
        # image-goal: 照片数即目标数，还有未找到的 goal 时不允许
        # VLM 提前 FINISH（找齐后由 NavAgent._should_finish 强制结束）。
        image_unfinished = task.get("goal_type") == "image" and bool(
            task.get("goals_unfound"))
        if not (needs_count or image_unfinished):
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
        record = {
            **(self._trace_record or {}),
            "step": world_state.get("step"),
            "event": str(event),
            "input_summary": {
                "instances": len(world_state.get("instances", [])),
                # U_t 统计用这个：上面的 instances 是有界表（top-K）行数，
                # 池子大时会低估。
                "unreported_instances": int(
                    world_state.get("instances_total") or 0) - len(
                    world_state.get("reported_instance_ids") or []),
                "frontiers": len(world_state.get("frontiers", [])),
                "task": world_state.get("task"),
            },
            "output": output,
            "validation": validation,
            "tool_calls": tool_calls,
        }
        try:
            record["world_state"] = snapshot(world_state)
            record["output"] = snapshot(output)
            record["decision_origin"] = (
                "fallback" if output is None else
                "harness" if validation != "ok" else "vlm")
            if self._trace_snapshot_fn is not None:
                try:
                    record["agent_snapshot"] = snapshot(self._trace_snapshot_fn())
                except Exception as exc:
                    record["snapshot_error"] = str(exc)
            record = snapshot(record)
            if self._trace_sink is not None:
                self._trace_sink(record)
            if self.logger is not None:
                self.logger.log(record)
        except Exception as exc:
            # An unavailable/full log disk must never turn a valid action
            # into the caller's navigation fallback.
            self._trace_warn(exc)

    def _trace_warn(self, exc):
        if self._trace_record is not None:
            self._trace_record.setdefault("capture_errors", []).append(str(exc))
        if not self._trace_warning:
            self._trace_warning = True
            print(f"[DecisionLoop] diagnostic capture failed: {exc}", flush=True)
