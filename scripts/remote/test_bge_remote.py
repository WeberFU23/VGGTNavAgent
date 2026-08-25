import json
import sys

sys.path.insert(0, "/root/autodl-tmp/vggt_nav_agent")
from mapping.caption_store import BGEM3Embedder

captions = json.load(open("/root/autodl-tmp/bge_test_captions.json", encoding="utf-8"))
embedder = BGEM3Embedder("/root/autodl-tmp/models/bge-m3",
                         device="cpu")
doc_embs = embedder.encode([c["caption"] for c in captions])

for query in sys.argv[1:]:
    q = embedder.encode([query])[0]
    scores = doc_embs @ q
    print(f"=== QUERY: {query} ===")
    for i in scores.argsort()[::-1]:
        print(f"  {scores[i]:.4f}  {captions[i]['image']}")
    print()
