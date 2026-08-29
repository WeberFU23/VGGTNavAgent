"""事件驱动的具身 VLM harness。

VLM 通过工具读取和编辑 3D instance memory，再选择实例、frontier、扫描、
报告或结束。底层跟随、避障与路径规划保持确定性；VLM 只有在自己
显式进入 adjustment 状态后，才能每轮输出一个白名单原子动作。

chat_fn(user_text, images) -> dict|None 可注入（生产接
VLMDecisionClient.agentic_chat，单测用 mock）。
"""

import json
import os
import threading
import time

from decision.prompts import build_decision_prompt, build_final_decision_prompt

ACTIONS = ("GOTO_INSTANCE", "GOTO_FRONTIER", "REPORT_FOUND", "SCAN",
           "EXPLORE", "FINISH", "START_ADJUST", "END_ADJUST",
           "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP",
           "LOOK_DOWN")

DEFAULT_MAX_TOOL_ROUNDS = 15
FINAL_ACTION_ATTEMPTS = 2

# 写工具：成功执行后世界状态已变化，动作校验前必须刷新 world-state。
WRITE_TOOLS = ("update_instance", "set_notes", "instantiate_points",
               "ground_target")

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
    "finish_check": {"GOTO_INSTANCE", "GOTO_FRONTIER", "FINISH"},
    "adjustment": {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP",
                   "LOOK_DOWN", "END_ADJUST"},
}

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

    def decide(self, event, world_state, map_png=None, images=None,
               state_fn=None):
        """一次事件驱动决策。返回 DecisionResult；最终非法/模型不可用
        返回 None（调用方回退确定性规则）。

        state_fn: 可选无参回调，在写工具成功后调用。可返回重新生成的
        world-state dict，或 (world-state, map_png)；后者会同时替换旧地图。"""
        state = world_state
        prompt = self._build_prompt(event, state)
        images = list(images or [])
        if map_png:
            images = self._with_topdown_map(images, map_png)
        tool_calls = 0
        tool_results = []
        while True:
            data = self._chat(prompt, images)
            if data is None:
                self._log(event, state, None, "model_unavailable",
                          tool_calls)
                return None
            tool_call = data.get("tool_call")
            if tool_call and str(event) == "adjustment":
                # takeover 期间禁止工具调用：按非法输出走校验失败重试路径。
                result, err = None, "tools are disabled during adjustment"
            elif tool_call:
                if tool_calls >= self.max_tool_rounds:
                    return self._finalize_after_tool_limit(
                        event, state, images, tool_calls, tool_results)
                tool_calls += 1
                feedback, tool_img, ok = self._run_tool(tool_call)
                tool_name = str(tool_call.get("name") or "")
                tool_results.append(
                    f"Tool {tool_calls}/{self.max_tool_rounds} "
                    f"({tool_name}) result:\n{feedback}")
                remaining = self.max_tool_rounds - tool_calls
                prompt += (
                    "\n\nTool result:\n" + feedback
                    + f"\nTool usage: {tool_calls}/{self.max_tool_rounds}; "
                      f"{remaining} calls remain."
                    + ("\nContinue with another tool call only if needed, "
                       "otherwise reply with the final decision JSON."
                       if remaining else
                       "\nThe hard tool-call limit has been reached."))
                if tool_img:
                    for label, payload in tool_img:
                        images = self._with_tool_image(images, label, payload)
                if ok and state_fn is not None and \
                        str(tool_call.get("name") or "") in WRITE_TOOLS:
                    state, refreshed_map, has_map = self._refresh_context(
                        state_fn, state)
                    if has_map:
                        images = self._with_topdown_map(
                            images, refreshed_map)
                    prompt += ("\n\nWorld state after your write:\n"
                               + json.dumps(state, ensure_ascii=False))
                if tool_calls >= self.max_tool_rounds:
                    return self._finalize_after_tool_limit(
                        event, state, images, tool_calls, tool_results)
                continue
            else:
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
            if data2 is None:
                result2, err2 = None, "model_unavailable"
            elif data2.get("tool_call"):
                result2, err2 = None, (
                    "a final action JSON was required; tool_call is not "
                    "allowed on validation retry")
            else:
                result2, err2 = self._validate(
                    data2, state, tool_calls, event)
            if result2 is not None:
                result2 = self._enforce_finish(result2, state)
                self._log(event, state, result2.as_dict(),
                          result2.validation, tool_calls)
                return result2
            self._log(event, state, None,
                      f"fallback: {err2}", tool_calls)
            return None

    def _finalize_after_tool_limit(self, event, state, images, tool_calls,
                                   tool_results):
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
            elif data.get("tool_call"):
                last_error = (
                    "tool_call is disabled after the hard limit; output one "
                    "final action JSON")
            else:
                result, last_error = self._validate(
                    data, state, tool_calls, event)
                if result is not None:
                    result = self._enforce_finish(result, state)
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
        self._log(event, state, result.as_dict(),
                  result.validation, tool_calls)
        return result

    @staticmethod
    def _forced_navigation_result(event, world_state, tool_calls, error):
        """Choose a valid progress action if final-only VLM replies stay invalid."""
        allowed = EVENT_ACTIONS.get(str(event), set(ACTIONS) - {"EXPLORE"})
        if "GOTO_INSTANCE" in allowed:
            instances = [item for item in world_state.get("instances", [])
                         if not item.get("reported", False)]
            if instances:
                return DecisionResult(
                    "GOTO_INSTANCE", str(instances[0]["id"]),
                    "Forced final action after tool limit: " + str(error),
                    validation="forced_after_tool_limit",
                    tool_calls=tool_calls)
            omitted = world_state.get("instances_omitted_ids", [])
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

    def _build_prompt(self, event, world_state):
        return build_decision_prompt(
            event, world_state, max_tool_rounds=self.max_tool_rounds)

    def _chat(self, prompt, images):
        try:
            data = self.chat_fn(prompt, images)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _serialize_tool_feedback(payload, max_chars=4000):
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
                return (self._serialize_tool_feedback(payload),
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
                return (self._serialize_tool_feedback(payload),
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
            return (self._serialize_tool_feedback(payload),
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
        elif action == "REPORT_FOUND":
            valid = {str(i["id"]) for i in world_state.get("instances", [])}
            valid |= {str(i) for i in
                      world_state.get("instances_omitted_ids", [])}
            if target_id not in valid:
                return None, (f"REPORT_FOUND target_id {target_id!r} is not "
                              "an available canonical instance")
            active = (world_state.get("navigation", {})
                      .get("active_target") or {})
            if active.get("type") != "instance" or \
                    str(active.get("id")) != target_id:
                return None, ("REPORT_FOUND must target the active canonical "
                              "instance")
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
            },
            "output": output,
            "validation": validation,
            "tool_calls": tool_calls,
        })
