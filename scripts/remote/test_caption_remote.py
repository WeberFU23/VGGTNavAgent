import base64
import json
import sys

import requests

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
from mapping.caption_store import CAPTION_PROMPT

images = sys.argv[1:] or ["/root/autodl-tmp/prompt_test_image_0006.jpg"]
models = requests.get("http://127.0.0.1:8000/v1/models", timeout=10).json()
model = models["data"][0]["id"]
for path in images:
    img_b64 = base64.b64encode(open(path, "rb").read()).decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 512,
    }
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"]
    usage = resp.json().get("usage", {})
    print(f"=== IMAGE: {path} (completion_tokens={usage.get('completion_tokens')}) ===")
    print(out)
    print()
print("=== PROMPT ===")
print(CAPTION_PROMPT)
