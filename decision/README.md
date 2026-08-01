# Event-driven VLM decision layer

This directory is method-side code. The benchmark remains model-agnostic and
continues to expose only RGB plus the natural-language task interface.

The VLM is a strategic supervisor, not a per-step motor policy. It is invoked
only for four events:

1. Parse the instruction into a faithful visual grounding phrase.
2. Choose a grounded memory candidate or continue exploration.
3. Verify arrival before emitting `TARGET_FOUND`.
4. Decide whether an `all` episode is exhausted after deterministic gating.

Low-level turning, forward motion, collision recovery, mapping, A*, and path
following remain deterministic. Every VLM response is strict JSON with a closed
decision enum. Invalid responses and API failures fall back to the deterministic
agent.

## OpenAI-compatible API configuration

```bash
export NAV_VLM_ENABLED=true
export NAV_VLM_API_URL=http://your-server:port/v1
export NAV_VLM_API_KEY=optional-token
export NAV_VLM_MODEL=your-vision-model

# Optional
export NAV_VLM_TIMEOUT=45
export NAV_VLM_MAX_TOKENS=300
export NAV_VLM_JSON_MODE=true
export NAV_VLM_CANDIDATE_LIMIT=4
export NAV_VLM_CANDIDATE_CONF=0.35
export NAV_VLM_VERIFY_CONF=0.50
export NAV_VLM_FINISH_CONF=0.60
```

`NAV_VLM_API_URL` may also be the full `/chat/completions` URL. If the serving
stack does not implement OpenAI `response_format`, set
`NAV_VLM_JSON_MODE=false`; the prompt still requires JSON and the client parses
and validates it.

For compatibility, missing `NAV_VLM_*` credentials fall back to
`EVAL_MODEL_API_URL`, `EVAL_MODEL_API_KEY`, and `EVAL_MODEL_NAME`. No API key or
raw response is logged.

## Information sent per event

- Instruction parse: text only.
- Candidate choice: current RGB plus at most four historical two-panel JPEGs
  (full context + red-mask close-up) and a compact JSON state/score summary.
- Arrival verification: current RGB plus the selected historical crop.
- Finish decision: current RGB plus compact agent-side memory counters.

No point cloud, raw trajectory, ground-truth pose/depth, semantic ID, or target
position is sent. Prompt definitions live in `prompts.py`; API and response
validation live in `vlm.py`; dependency-free contracts live in `types.py`.
