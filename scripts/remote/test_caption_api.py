import base64
import os
import sys

import requests

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
from mapping.caption_store import CAPTION_PROMPT

url = (os.environ.get("NAV_CAPTION_API_URL")
       or os.environ.get("NAV_VLM_API_URL", "")).rstrip("/")
key = (os.environ.get("NAV_CAPTION_API_KEY")
       or os.environ.get("NAV_VLM_API_KEY", ""))
model = os.environ.get("NAV_CAPTION_API_MODEL", "")
print("model:", model, "| url host:", url.split("//")[-1].split("/")[0])

img_path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/prompt_test_image_0006.jpg"
img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()
payload = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": CAPTION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
    "temperature": 0,
    "max_tokens": 1024,
}
endpoint = url if url.endswith("/chat/completions") else url + "/chat/completions"
resp = requests.post(endpoint, headers={"Authorization": f"Bearer {key}"},
                     json=payload, timeout=300)
print("HTTP", resp.status_code)
data = resp.json()
print("usage:", data.get("usage"))
print("=== OUTPUT ===")
print(data["choices"][0]["message"]["content"])
