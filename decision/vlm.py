"""OpenAI-compatible transport for the unified event-driven decision loop."""

import base64
import hashlib
import io
import json
import os
import re
import threading
import time

import numpy as np
from PIL import Image


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VLMDecisionClient:
    """Transport only; prompts, schemas and validation live in DecisionLoop."""

    JSON_SYSTEM = (
        "You are the strategic visual reasoning module of an embodied "
        "navigation agent. Use only supplied state and images. Never invent "
        "geometry or simulator state. Return one strict JSON object with no "
        "markdown or extra text.")
    TEXT_SYSTEM = (
        "You are the visual perception module of an embodied navigation "
        "agent. Describe only what is visible in the supplied images. Be "
        "concise and factual.")

    def __init__(self, api_url=None, api_key=None, model=None, timeout=45.0,
                 enabled=None, post_fn=None, trace_path=None, image_dir=None):
        self.api_url = str(api_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip()
        configured = bool(self.api_url and self.model)
        self.enabled = (configured if enabled is None
                        else bool(enabled and configured))
        self.timeout = float(timeout)
        self.max_tokens = int(os.environ.get("NAV_VLM_MAX_TOKENS", "300"))
        self.json_mode = _env_bool("NAV_VLM_JSON_MODE", True)
        self._post_fn = post_fn
        self._warned = False
        self.trace_path = str(trace_path or os.environ.get(
            "NAV_VLM_TRACE", "")).strip()
        self.image_dir = str(image_dir or os.environ.get(
            "NAV_VLM_IMAGE_DIR", "")).strip()
        self._trace_lock = threading.Lock()
        self._trace_context = {}
        self._trace_warned = False
        self._trace_seq = 0

    def set_trace_path(self, path):
        self.trace_path = str(path or "").strip()
        if self.trace_path:
            os.makedirs(os.path.dirname(self.trace_path) or ".", exist_ok=True)

    def set_trace_context(self, **context):
        self._trace_context = dict(context)

    @classmethod
    def from_env(cls):
        url = (os.environ.get("NAV_VLM_API_URL")
               or os.environ.get("EVAL_MODEL_API_URL"))
        key = (os.environ.get("NAV_VLM_API_KEY")
               or os.environ.get("EVAL_MODEL_API_KEY"))
        model = (os.environ.get("NAV_VLM_MODEL")
                 or os.environ.get("EVAL_MODEL_NAME"))
        configured = bool(url and model)
        return cls(
            api_url=url, api_key=key, model=model,
            timeout=float(os.environ.get("NAV_VLM_TIMEOUT", "45")),
            enabled=_env_bool("NAV_VLM_ENABLED", configured))

    def agentic_chat(self, user_prompt, images=None):
        """决策层 agentic 循环低层接口：自由 prompt + 原始 JPEG 字节图像
        列表，返回解析后的 JSON dict；API 不可达自动回退 None（调用方
        走确定性规则），并打 warning。复用同一 HTTP/JSON 解析通路。"""
        if not self.enabled:
            return None
        prompt = str(user_prompt)
        image_parts = self._image_parts(images)
        payload = self._build_payload(
            prompt, image_parts,
            self.JSON_SYSTEM, self.json_mode)
        response = self._send(payload)
        parsed = self._extract_json(response)
        self._trace("decision", prompt, image_parts, response, parsed)
        return parsed

    def chat_text(self, user_prompt, images=None, max_tokens=None):
        """自由文本 VLM 调用（实例级描述等非 JSON 输出）。
        返回纯文本；客户端禁用或 API 不可达返回 None。"""
        if not self.enabled:
            return None
        prompt = str(user_prompt)
        image_parts = self._image_parts(images)
        payload = self._build_payload(
            prompt, image_parts,
            self.TEXT_SYSTEM, False)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        response = self._send(payload)
        parsed = self._extract_text(response)
        self._trace("text", prompt, image_parts, response, parsed)
        return parsed

    def _trace(self, kind, prompt, images, response, parsed):
        """Persist VLM I/O and, when requested, the exact transmitted images.

        ``images`` has already passed ``_image_parts`` and its image-budget
        truncation, so the snapshots match the payload rather than a larger
        pre-filter input list. Credentials and base64 payloads never enter the
        JSONL trace.
        """
        if not self.trace_path and not self.image_dir:
            return
        try:
            with self._trace_lock:
                self._trace_seq += 1
                image_meta = self._snapshot_images(
                    kind, images, self._trace_seq)
                record = {
                    "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": kind,
                    "model": self.model,
                    "api_url": self.api_url,
                    "prompt": prompt,
                    "images": image_meta,
                    "raw_response": response,
                    "parsed_output": parsed,
                    "ok": response is not None and parsed is not None,
                    "context": dict(self._trace_context),
                }
                if self.trace_path:
                    os.makedirs(
                        os.path.dirname(self.trace_path) or ".", exist_ok=True)
                    with open(self.trace_path, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(
                            record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            if not self._trace_warned:
                print(f"[VLMDecision] trace 写入失败: {exc}", flush=True)
                self._trace_warned = True

    def _snapshot_images(self, kind, images, sequence):
        """Build image metadata and optionally save byte-identical snapshots.

        NAV_VLM_TRACE_INLINE_IMAGES=1 时把图像 base64 内联进 trace 记录，
        使单个 JSONL 文件即可完整复盘 VLM 输入（不需要配套图像目录）。
        """
        image_meta = []
        inline = _env_bool("NAV_VLM_TRACE_INLINE_IMAGES", False)
        context = dict(self._trace_context)
        episode = self._safe_name(context.get("episode", "episode"))
        step = self._safe_name(context.get("step", "unknown"))
        call_kind = self._safe_name(kind)
        if self.image_dir:
            os.makedirs(self.image_dir, exist_ok=True)
        for index, (label, raw, mime) in enumerate(images):
            data = bytes(raw) if isinstance(raw, (bytes, bytearray)) else b""
            meta = {
                "label": label, "mime_type": mime, "bytes": len(data),
                "sha1": hashlib.sha1(data).hexdigest() if data else None,
            }
            if inline and data:
                meta["data_b64"] = base64.b64encode(data).decode("ascii")
            if self.image_dir and data:
                extension = ".png" if mime == "image/png" else ".jpg"
                filename = (
                    f"{episode}_step-{step}_call-{sequence:06d}_"
                    f"{call_kind}_{index:02d}-{self._safe_name(label)}"
                    f"{extension}")
                path = os.path.join(self.image_dir, filename)
                with open(path, "wb") as fp:
                    fp.write(data)
                meta["saved_path"] = path
            image_meta.append(meta)
        return image_meta

    @staticmethod
    def _safe_name(value):
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
        return cleaned[:96] or "unknown"

    @staticmethod
    def _image_parts(images):
        parts = []
        max_images = int(os.environ.get("NAV_VLM_MAX_IMAGES", "4"))
        for i, item in enumerate(images or []):
            if isinstance(item, tuple) and len(item) == 2:
                name, raw = item
            else:
                name, raw = f"attached_image_{i}", item
            if raw:
                data = bytes(raw) if isinstance(raw, (bytes, bytearray)) else raw
                is_png = (isinstance(data, bytes) and
                          data.startswith(b"\x89PNG\r\n\x1a\n"))
                mime = "image/png" if is_png else "image/jpeg"
                parts.append((str(name), raw, mime))
        if max_images > 0:
            return parts[:max_images]
        return parts

    def _build_payload(self, user_prompt, images, system_prompt, json_mode):
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if "NAV_VLM_ENABLE_THINKING" in os.environ:
            payload["enable_thinking"] = _env_bool(
                "NAV_VLM_ENABLE_THINKING", True)
        return payload

    def _send(self, payload):
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
            return response.json()
        except Exception as exc:
            if not self._warned:
                print(f"[VLMDecision] API 不可用，回退确定性决策: {exc}")
                self._warned = True
            return None

    def _chat_url(self):
        url = self.api_url.rstrip("/")
        return (url if url.endswith("/chat/completions")
                else f"{url}/chat/completions")

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
    def _extract_text(response):
        """自由文本响应提取；非 choices 结构或空内容返回 None。"""
        if not isinstance(response, dict) or "choices" not in response:
            return None
        try:
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) for item in content
                    if isinstance(item, dict))
            text = str(content).strip()
            text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text).strip()
            return text or None
        except (KeyError, IndexError, TypeError):
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
