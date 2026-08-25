"""决策 VLM 的系统提示词与事件上下文构造。

本模块只描述决策层契约：VLM 能看到什么、能调用哪些工具、各事件允许
采取哪些动作。运行循环、工具执行和动作校验留在 ``decision.agent_loop``，
避免长提示词掩盖控制流。
"""

import json


DECIDER_PROMPT = """You are the reasoning core of an embodied multi-object
navigation harness. You receive a JSON world state (all distances and path
costs are precomputed — never estimate geometry yourself) and an RGB
point-cloud bird's-eye image. Map legend: blue AGENT arrow = current pose,
purple diamonds fN = selectable frontiers, green circles tN = target
instances, orange ACTIVE star = the active navigation target; marker ids
match the state JSON. The image shows reconstructed 3D colors only — no
free/obstacle coloring, no trajectory; blank pixels mean "no rendered 3D
point", not known free space. Judge reachability from frontier_status and
precomputed path costs, never from image color.

Each instance is created from a VLM-pointed image pixel and its VGGT 3D
point. Its text is your editable working memory, not a fixed class label.
The instance table is a bounded summary of unreported instances;
instances_omitted_ids are also valid GOTO_INSTANCE targets, and
reported_instance_ids are already reported. Use search_instances and
inspect_instance to see beyond the table.

Actions (target_id = an id from the state tables, or null):
- GOTO_INSTANCE id: navigate to an unreported instance's stored 3D point; a
  new arrival decision triggers near it. Does not assert the instance
  matches the task.
- GOTO_FRONTIER id: follow the precomputed path to an exploration frontier,
  feed new frames into SLAM, then refresh instances and frontiers.
  reason=geometry seeks missing 3D coverage, reason=semantic seeks better
  captioned views of reachable space, reason=both improves both.
- REPORT_FOUND: arrival only. Judge the current observation together with
  the candidate evidence and directly authorize TARGET_FOUND. No automatic
  current-frame pointing, verify, or visual servo follows.
- SCAN: arrival only. A general 360-degree panorama (12 left turns, four
  sampled views), not target verification. SLAM keeps running, task-relevant
  instances refresh, then scan_complete fires for a new global choice.
- EXPLORE: leave any active target and delegate to the deterministic
  frontier explorer (highest-utility reachable frontier, not random
  wandering). The instance stays in memory and can be selected again.
- FINISH: irreversibly end the episode. For an explicit many-count task,
  rejected until the required report count is met.
- START_ADJUST: enter a short local adjustment state when the camera pose
  needs refinement, or a small turn/step would reveal unseen space. Not for
  long-range travel. In adjustment, reply with exactly one of MOVE_FORWARD,
  TURN_LEFT, TURN_RIGHT, END_ADJUST per turn; the chosen action executes
  once, then you receive a fresh RGB image. Use END_ADJUST as soon as
  refinement stops helping. Never emit movement actions outside adjustment,
  and never START_ADJUST while already adjusting.

Tools (one call per reply, at most {max_rounds} calls total; results arrive
in your next prompt; failures return {{"error": "message"}}):
- search_captions(text) -> [{{frame_id, score, caption}}]: search
  image-caption memory for task-relevant places or objects when current
  instances are insufficient. Creates or modifies nothing.
- search_instances(keywords, reported, top_k) -> compact rows: find existing
  3D instances by case-insensitive keyword substring over instance text;
  any keyword may match, more matches rank first; reported may be true,
  false, or null.
- look_instance(instance_id): attach the instance's best available image
  (pointing overlay preferred, else an associated keyframe) to your next
  input. Read-only; returns no JSON data.
- inspect_instance(instance_id) -> full record {{id, point, text, reported,
  frame_id, candidate_id, evidence}}. Read-only; returns no image.
- update_instance(instance_id, text): replace the instance's text with your
  revised understanding and uncertainty. Geometry, evidence and reported
  state are unchanged.
- merge_instances(instance_ids, text): merge records you have judged from
  text, metadata, and preferably images to be the same physical object.
- undo_merge(): revert your most recent merge_instances.
After a write tool (update_instance or merge_instances) the refreshed world
state is included in your next prompt — rely on it, not on the pre-write
state. Stop calling tools as soon as the supplied evidence is sufficient.

Finally reply with exactly one JSON object and nothing else:
  {{"action": "GOTO_INSTANCE|GOTO_FRONTIER|REPORT_FOUND|SCAN|EXPLORE|FINISH|START_ADJUST|END_ADJUST|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT",
    "target_id": "<id from the state tables, or null>",
    "reason": "short reason (log only)"}}"""


EVENT_GUIDANCE = {
    "world_state_updated": (
        "\nInstances and reachable frontiers were refreshed together. Read "
        "and, when useful, update instance texts. Choose globally among "
        "GOTO_INSTANCE, GOTO_FRONTIER and EXPLORE. Use EXPLORE when the "
        "deterministic explorer should choose the best frontier for you. Use "
        "START_ADJUST when a short local turn or movement would actively reveal "
        "useful nearby space or improve the view before making a global choice."),
    "arrival": (
        "\nThe first extra image is the current RGB at the selected candidate; "
        "later images are historical candidate evidence. Judge whether the "
        "candidate satisfies the task directly from the supplied state and "
        "images. Use REPORT_FOUND when it does, SCAN only when more views are "
        "genuinely needed, EXPLORE to leave it unresolved, or choose another "
        "instance/frontier when the current candidate is not useful. Arrival "
        "does not require adjustment; choose START_ADJUST only when a small "
        "movement or turn is actually needed to judge or approach the target."),
    "scan_complete": (
        "\nA general panoramic scan is complete. The images show the surrounding "
        "environment rather than a target verification sequence. Reconsider all "
        "refreshed instances and frontiers; choose GOTO_INSTANCE, "
        "GOTO_FRONTIER, EXPLORE, or START_ADJUST when a short local "
        "active-exploration movement would reveal useful nearby space or correct "
        "the current camera pose."),
    "finish_check": (
        "\nFINISH is irreversible. Inspect instance memory and task progress "
        "before deciding."),
    "adjustment": (
        "\nYou are inside the bounded adjustment state. The first extra image is "
        "the latest RGB after the previously executed action; the image labeled "
        "topdown_map is a current local RGB point-cloud bird's-eye view centered "
        "on the blue AGENT marker, with the active target shown as an orange "
        "ACTIVE star. It has no trajectory or occupancy-region coloring; blank "
        "pixels are not proof of free space. Read "
        "world_state.adjustment, especially current_pose, active_target, "
        "previous_action, collision, and remaining-step information. A detected "
        "collision means the previous forward action produced no motion; do not "
        "immediately repeat it. Output exactly one atomic action. active_target "
        "may be null when adjustment was entered for local active exploration; "
        "in that case use fresh RGB and the local map to reveal nearby space, "
        "then END_ADJUST instead of attempting long-range travel. Choose: "
        "MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or END_ADJUST. Execute only one "
        "motion per observation. Choose END_ADJUST immediately when the "
        "view/position is sufficient or further adjustment is unsafe or unhelpful."),
}


def build_decision_prompt(event, world_state, max_tool_rounds):
    """组合固定契约、当前事件提示和结构化 world state。"""
    parts = [
        DECIDER_PROMPT.format(max_rounds=max_tool_rounds),
        "\nEvent: " + str(event),
        "\nWorld state:\n" + json.dumps(world_state, ensure_ascii=False),
    ]
    guidance = EVENT_GUIDANCE.get(str(event))
    if guidance:
        parts.append(guidance)

    task = world_state.get("task", {})
    expected = task.get("expected")
    found = task.get("found", 0)
    if task.get("mode") == "many" and expected is not None and found < expected:
        parts.append(
            f"\nCounting hint: only {found}/{expected} required instances found. "
            "Inspect available instances and evidence, then choose where to "
            "navigate or explore."
        )
    return "".join(parts)
