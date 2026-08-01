"""Strict prompts for the event-driven VLM navigation supervisor."""

import json


SYSTEM_PROMPT = """You are the strategic visual reasoning module of an embodied
multi-object navigation agent in Habitat HM3D.

Sensor and control contract:
- You receive RGB images and a natural-language instruction only.
- There is no ground-truth depth, GPS, compass, object ID, semantic map, or pose.
- Geometry, mapping, collision handling, path planning, and low-level discrete
  actions are handled by deterministic modules. Never invent coordinates,
  distances, unseen objects, or simulator state.
- A candidate evidence image is a two-panel historical view: the left panel is
  full-scene context and the right panel is a close-up. Pixels tinted red are
  the proposed segmentation mask. Judge the red object while using the left
  panel for relational clues such as "next to the sofa".
- TARGET_FOUND must refer to a visible instance matching the full instruction,
  including category, color, material, shape, and distinguishing attributes.
- In many/all modes, previously reported instances must not be selected again.

Task and strategic decision semantics:
- single/any: one verified matching instance completes the target search.
- many: report distinct matching instances until the instructed count is met.
- all: report every discoverable matching instance, then FINISH only when the
  agent-side visual memory and exploration evidence are exhausted.
- NAVIGATE(candidate_id) asks the deterministic planner to return to that known
  3D candidate; it does not immediately claim success.
- EXPLORE continues RGB collection and map growth.
- REPORT_FOUND causes benchmark signal 6 (TARGET_FOUND) at the current location.
- FINISH causes benchmark signal 0 and irreversibly ends a many/all episode.
- SCAN rotates in place using paid 30-degree turn actions for more visual proof.

Use only supplied evidence. Be conservative about false positives, but do not
reject a candidate merely because lighting or viewpoint differs. Return one
strict JSON object matching the requested schema, with no markdown or extra
text."""


def parse_instruction_prompt(instruction, target_mode, target_count):
    return f"""Event: PARSE_TARGET_INSTRUCTION

Natural-language instruction:
{json.dumps(str(instruction), ensure_ascii=False)}

Task mode: {target_mode}
Required count when known: {target_count}

Extract a short visual phrase suitable for CLIP retrieval and text-prompted
segmentation. Preserve visual attributes that distinguish the requested object.
The grounding query should contain the object noun plus intrinsic visible
attributes, but omit navigation verbs, quantifiers, and relational clauses such
as "near the table". Preserve the complete relational requirement in
target_description for later candidate verification. Do not add synonyms or
attributes that are absent from the instruction.

Return exactly:
{{
  "grounding_query": "short visual noun phrase",
  "target_description": "faithful description of what visually counts as a match",
  "confidence": 0.0
}}
"""


def candidate_prompt(instruction, target_spec, state, candidates):
    compact = []
    for i, candidate in enumerate(candidates):
        compact.append({
            "image_label": f"candidate_{i}",
            "candidate_id": candidate.get("candidate_id"),
            "clip_score": candidate.get("score"),
            "sam_score": candidate.get("sam_score"),
            "matched_prompt": candidate.get("matched_prompt"),
            "frame_id": candidate.get("frame_id"),
            "num_confident_3d_points": candidate.get("num_points"),
        })
    return f"""Event: CHOOSE_MEMORY_CANDIDATE_OR_EXPLORE

Instruction: {json.dumps(str(instruction), ensure_ascii=False)}
Parsed target: {json.dumps(target_spec, ensure_ascii=False)}
Navigation state: {json.dumps(state, ensure_ascii=False)}
Candidates: {json.dumps(compact, ensure_ascii=False)}

Image order:
1. current_observation: the agent's current RGB view.
2. candidate_0, candidate_1, ...: historical two-panel red-mask evidence, in
   the same order as the Candidates array. A candidate without an evidence image
   is less trustworthy, but its numeric scores alone are not proof of identity.

Choose NAVIGATE only when one candidate plausibly matches the full target.
Choose EXPLORE when no candidate is reliable or more evidence is needed.
CLIP and SAM scores are ranking hints, not calibrated probabilities or proof.
The current view need not contain a historical candidate—the purpose of NAVIGATE
is to return to a target remembered elsewhere. Judge candidate identity from its
red-mask crop and use the current view mainly for the exploration hint.
`exploration_hint` is a short macro suggestion for the deterministic explorer:
none, forward, turn_left, turn_right, or scan. It is advisory and may be
overridden by collision handling.

Return exactly:
{{
  "decision": "navigate" or "explore",
  "candidate_id": "one supplied candidate_id" or null,
  "rejected_candidate_ids": ["definite false-positive candidate IDs only"],
  "exploration_hint": "none|forward|turn_left|turn_right|scan",
  "confidence": 0.0,
  "reason": "brief evidence-based reason"
}}
"""


def arrival_prompt(instruction, target_spec, state):
    return f"""Event: VERIFY_ARRIVAL_BEFORE_REPORTING

Instruction: {json.dumps(str(instruction), ensure_ascii=False)}
Parsed target: {json.dumps(target_spec, ensure_ascii=False)}
Navigation state: {json.dumps(state, ensure_ascii=False)}

Image order:
1. current_observation: what the agent sees now.
2. selected_candidate: the historical red-mask crop that caused navigation.

The geometric controller believes it reached the candidate and an independent
text segmentation model detected the target in the current view. Decide whether
the current visible object and historical red-mask object match each other and
the full instruction. REPORT_FOUND has a benchmark penalty if wrong.

Return exactly:
{{
  "decision": "report_found" or "scan" or "reject",
  "candidate_id": null,
  "rejected_candidate_ids": [],
  "exploration_hint": "none",
  "confidence": 0.0,
  "reason": "brief visual reason"
}}
"""


def finish_prompt(instruction, target_spec, state):
    return f"""Event: DECIDE_ALL_MODE_FINISH

Instruction: {json.dumps(str(instruction), ensure_ascii=False)}
Parsed target: {json.dumps(target_spec, ensure_ascii=False)}
Navigation state: {json.dumps(state, ensure_ascii=False)}

The agent is in ALL mode. FINISH ends the whole episode permanently. The state
summarizes only agent-side memory; it contains no correctness feedback or true
remaining-target count. Finish only if exploration is late, multiple recent
memory queries found no new non-duplicate candidate, and the explored memory is
unlikely to yield another matching instance. Otherwise continue exploring.

Return exactly:
{{
  "decision": "finish" or "explore",
  "candidate_id": null,
  "rejected_candidate_ids": [],
  "exploration_hint": "none|forward|turn_left|turn_right|scan",
  "confidence": 0.0,
  "reason": "brief reason based only on supplied state and current RGB"
}}
"""
