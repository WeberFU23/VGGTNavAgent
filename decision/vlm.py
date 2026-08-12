"""OpenAI-compatible transport for the unified event-driven decision loop."""

import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VLMDecisionClient:
    """Transport only; prompts, schemas and validation live in DecisionLoop."""

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

    def agentic_chat(self, user_prompt, images=None):
        """决策层 agentic 循环低层接口：自由 prompt + 原始 JPEG 字节图像
        列表，返回解析后的 JSON dict；API 不可达自动回退 None（调用方
        走确定性规则），并打 warning。复用同一 HTTP/JSON 解析通路。"""
        parts = []
        for i, item in enumerate(images or []):
            if isinstance(item, tuple) and len(item) == 2:
                name, raw = item
            else:
                name, raw = f"attached_image_{i}", item
            if raw:
                parts.append((str(name), raw, "image/jpeg"))
        return self._request(str(user_prompt), parts)

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
                {"role": "system", "content": (
                    "You are the strategic visual reasoning module of an "
                    "embodied navigation agent. Use only supplied state and "
                    "images. Never invent geometry or simulator state. Return "
                    "one strict JSON object with no markdown or extra text.")},
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
    def encode_rgb(rgb):
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[-1] < 3:
            return b""
        array = np.clip(array[..., :3], 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image.thumbnail((768, 768))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
        return buffer.getvalue()
