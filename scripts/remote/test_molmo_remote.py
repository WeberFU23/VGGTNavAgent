"""Molmo pointing 远端冒烟测试：发图 + 自然语言目标，打印原始输出并解析。

用法（远端，vllm 环境）:
    python test_molmo_remote.py <image> [goal ...]
"""
import base64
import io
import sys

import requests
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
from mapping.pointing import MOLMO_POINT_PROMPT, _parse_molmo_points

image_path = sys.argv[1]
goals = sys.argv[2:] or ["the red square"]

img = Image.open(image_path).convert("RGB")
w, h = img.size
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=88)
img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

models = requests.get("http://127.0.0.1:8000/v1/models", timeout=10).json()
model = models["data"][0]["id"]
print(f"model={model} image={image_path} ({w}x{h})")
for goal in goals:
    prompt = MOLMO_POINT_PROMPT.format(goal_text=goal)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 512,
    }
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions",
                         json=payload, timeout=300)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"]
    print(f"=== GOAL: {goal} ===")
    print("raw:", out)
    print("parsed:", _parse_molmo_points(out, w, h))
    print()
