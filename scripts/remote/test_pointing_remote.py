import base64
import io
import json
import sys

import requests
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
from mapping.pointing import POINT_PROMPT

image_path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/point_test_image_0000.jpg"
goals = sys.argv[2:] or ["a TV"]

img = Image.open(image_path).convert("RGB")
w, h = img.size
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=88)
img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

models = requests.get("http://127.0.0.1:8000/v1/models", timeout=10).json()
model = models["data"][0]["id"]
print(f"model={model} image={image_path} ({w}x{h})")
for goal in goals:
    prompt = POINT_PROMPT.format(goal_text=goal, width=w, height=h)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]}],
        "temperature": 0,
        "max_tokens": 512,
    }
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=180)
    resp.raise_for_status()
    out = resp.json()["choices"][0]["message"]["content"]
    print(f"=== GOAL: {goal} ===")
    print(out)
    print()
print("=== PROMPT (goal=a TV) ===")
print(POINT_PROMPT.format(goal_text="a TV", width=w, height=h))
