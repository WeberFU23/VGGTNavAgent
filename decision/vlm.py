"""OpenAI-compatible, event-driven VLM client with safe fallback."""

import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

from decision import prompts
from decision.types import StrategicDecision, TargetSpec


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VLMDecisionClient:
    """Small strategic client; it never chooses benchmark motor actions."""

    def __init__(self, api_url=None, api_key=None, model=None, timeout=45.0,
                 enabled=None, post_fn=None):
        self.api_url = str(api_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        configured = bool(self.api_url and self.model)
        self.enabled = configured if enabled is None else bool(enabled and configured)
        self.timeout = float(timeout)
        self.max_tokens = int(os.environ.get("NAV_VLM_MAX_TOKENS", "300"))
        self.json_mode = _env_bool("NAV_VLM_JSON_MODE", True)
        self._post_fn = post_fn
        self._warned = False

    @classmethod
    def from_env(cls):
        url = os.environ.get("NAV_VLM_API_URL") or os.environ.get("EVAL_MODEL_API_URL")
        key = os.environ.get("NAV_VLM_API_KEY") or os.environ.get("EVAL_MODEL_API_KEY")
        model = os.environ.get("NAV_VLM_MODEL") or os.environ.get("EVAL_MODEL_NAME")
        configured = bool(url and model)
        return cls(
            api_url=url, api_key=key, model=model,
            timeout=float(os.environ.get("NAV_VLM_TIMEOUT", "45")),
            enabled=_env_bool("NAV_VLM_ENABLED", configured))

    def parse_instruction(self, instruction, target_mode, target_count):
        data = self._request(
            prompts.parse_instruction_prompt(instruction, target_mode, target_count), [])
        if not data:
            return None
        query = str(data.get("grounding_query") or "").strip()
        description = str(data.get("target_description") or "").strip()
        if not query or not description:
            return None
        return TargetSpec(query, description, self._confidence(data))

    def choose_candidate(self, instruction, target_spec, state, current_rgb,
                         candidates, evidence_by_id):
        target_dict = self._target_dict(target_spec)
        images = [("current_observation", self._rgb_jpeg(current_rgb), "image/jpeg")]
        for i, candidate in enumerate(candidates):
            evidence = evidence_by_id.get(candidate.get("candidate_id"))
            if evidence:
                images.append((f"candidate_{i}", evidence, "image/jpeg"))
        data = self._request(
            prompts.candidate_prompt(instruction, target_dict, state, candidates), images)
        return self._strategic(data, {"navigate", "explore"})

    def verify_arrival(self, instruction, target_spec, state, current_rgb,
                       selected_evidence=None):
        images = [("current_observation", self._rgb_jpeg(current_rgb), "image/jpeg")]
        if selected_evidence:
            images.append(("selected_candidate", selected_evidence, "image/jpeg"))
        data = self._request(
            prompts.arrival_prompt(instruction, self._target_dict(target_spec), state),
            images)
        return self._strategic(data, {"report_found", "scan", "reject"})

    def decide_finish(self, instruction, target_spec, state, current_rgb):
        images = [("current_observation", self._rgb_jpeg(current_rgb), "image/jpeg")]
        data = self._request(
            prompts.finish_prompt(instruction, self._target_dict(target_spec), state),
            images)
        return self._strategic(data, {"finish", "explore"})

    def agentic_chat(self, user_prompt, images=None):
        """决策层 agentic 循环低层接口：自由 prompt + 原始 JPEG 字节图像
        列表，返回解析后的 JSON dict；API 不可达自动回退 None（调用方
        走确定性规则），并打 warning。复用同一 HTTP/JSON 解析通路。"""
        parts = []
        for i, raw in enumerate(images or []):
            if raw:
                parts.append((f"attached_image_{i}", raw, "image/jpeg"))
        return self._request(str(user_prompt), parts)

    @staticmethod
    def _target_dict(target_spec):
        return {
            "grounding_query": target_spec.grounding_query,
            "target_description": target_spec.target_description,
        }

    def _strategic(self, data, allowed):
        if not data:
            return None
        decision = str(data.get("decision") or "").strip().lower()
        if decision not in allowed:
            return None
        rejected = data.get("rejected_candidate_ids") or []
        if not isinstance(rejected, list):
            rejected = []
        hint = str(data.get("exploration_hint") or "none").lower()
        if hint not in {"none", "forward", "turn_left", "turn_right", "scan"}:
            hint = "none"
        candidate_id = data.get("candidate_id")
        return StrategicDecision(
            decision=decision,
            candidate_id=str(candidate_id) if candidate_id is not None else None,
            rejected_candidate_ids=[str(x) for x in rejected],
            exploration_hint=hint,
            confidence=self._confidence(data),
            reason=str(data.get("reason") or "")[:300])

    @staticmethod
    def _confidence(data):
        try:
            return min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def _request(self, user_prompt, images):
        if not self.enabled:
            return None
        content = [{"type": "text", "text": user_prompt}]
        for name, raw, mime in images:
            if not raw:
                continue
            encoded = base64.b64encode(raw).decode("ascii")
            content.append({"type": "text", "text": f"Image label: {name}"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            if self._post_fn is None:
                import requests
                post_fn = requests.post
            else:
                post_fn = self._post_fn
            response = post_fn(
                self._chat_url(), headers=headers, json=payload,
                timeout=self.timeout)
            response.raise_for_status()
            return self._extract_json(response.json())
        except Exception as exc:
            if not self._warned:
                print(f"[VLMDecision] API 不可用，回退确定性决策: {exc}")
                self._warned = True
            return None

    def _chat_url(self):
        url = self.api_url.rstrip("/")
        return url if url.endswith("/chat/completions") else f"{url}/chat/completions"

    @staticmethod
    def _extract_json(response):
        if not isinstance(response, dict):
            return None
        if "choices" not in response:
            return response
        try:
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) for item in content
                    if isinstance(item, dict))
            content = str(content).strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content,
                             flags=re.IGNORECASE)
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _rgb_jpeg(rgb):
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[-1] < 3:
            return b""
        array = np.clip(array[..., :3], 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image.thumbnail((768, 768))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        return buffer.getvalue()
