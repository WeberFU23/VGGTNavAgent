"""决策 VLM 的系统提示词与事件上下文构造。

本模块只描述决策层契约：VLM 能看到什么、能调用哪些工具、各事件允许
采取哪些动作。运行循环、工具执行和动作校验留在 ``decision.agent_loop``，
避免长提示词掩盖控制流。
"""

import json


DECIDER_PROMPT = """# Your task

You control a robot navigating an indoor 3D scene. The task is in
world_state.task.goal, e.g. "Find all bathroom shelf objects." Explore the
scene, find the target object(s), and report each one when you are next to
it and can actually see it.

task.mode defines the completion rule:
- "any": report any ONE matching instance, then FINISH.
- "many": report exactly task.expected DISTINCT matching instances;
  task.found counts your accepted reports. FINISH is rejected until the
  count is reached.
- "all": report EVERY matching instance in the scene; the total is unknown.
  Explore until you are confident nothing remains unreported, then FINISH.
FINISH is irreversible: the episode ends immediately.

# What you receive at each decision

1. A JSON world state:
   - task: goal / mode / found / expected, as above.
   - step, max_steps, steps_remaining: your action budget. Plan around it.
   - instances: candidate targets registered so far (see Memory below).
   - frontiers: reachable exploration candidates, each {{id, path_cost_m}};
     path_cost_m is the precomputed path length in meters. The topdown map
     shows where each frontier lies relative to you: treat several
     frontiers clustered in the same direction as one exploration goal,
     and pick the frontier that advances into a NEW area rather than
     merely the closest one.
   - navigation: current_pose (x, y, yaw), current_frame_id (the frame id
     of the latest RGB just fed to the map server — the current view; usable
     with view_frame and instantiate_points) and active_target.
   - notes: YOUR persistent working memory (see Memory below).
   - recent_actions: your last 3 high-level actions with outcomes
     (ok / collision / arrived).
   - new_keyframes (only when present): {{frame_id, caption excerpt}} rows
     for frames collected since your last decision. Frame images are NOT
     attached — call view_frame(frame_id) to see one.
2. A bird's-eye map image reconstructed from the 3D point cloud. Legend:
   blue AGENT arrow = current pose, purple diamonds fN = frontiers, green
   circles tN = instances, orange ACTIVE star = active target; marker ids
   match the state JSON. The image shows reconstructed 3D colors only —
   no free/obstacle coloring and no trajectory; blank pixels mean "no
   rendered 3D point", i.e. space the robot has not observed yet — a large
   blank region is unexplored and worth moving toward. Combine the image
   with the JSON numbers when choosing: use path_cost_m for distance and
   reachability, and use the image for spatial layout — which direction
   each frontier lies in, which frontiers belong to the same region, and
   where unexplored space remains. Do not simply pick the closest
   frontier: check the image first, and choose the frontier that opens up
   new space or a new room.
3. Event-specific images: the current RGB on arrival, panorama views after
   SCAN, or images you requested through tools.

All world-space distances, coordinates and path costs are computed by the
system. Never estimate world geometry or output world coordinates; refer to
objects by their ids. The only coordinates you may provide are normalized
image pixels required by instantiate_points.

# Memory: instances and notes

Each pointing result is stored as an observation. The harness automatically
compares its marked photo with geometrically nearby canonical instances and
assigns it to one physical-object identity. You only see canonical instances;
identity resolution is not your responsibility. observation_count tells how
many views support an instance. reported_instances and report_claims are
already claimed and cannot be navigated to or reported again. The instance
table is a bounded summary of available instances; instances_omitted_ids are
also valid GOTO_INSTANCE targets. Use search_instances and get_instance to see
beyond the table.

notes is a string that persists across decisions and is handed back inside
every world state. Maintain it with set_notes: current plan, ruled-out
areas or hypotheses, next steps (at most 500 characters). It is your only
long-term memory beyond instances — keep it current.

# Tools

To call one, reply with exactly one JSON object and nothing else:
  {{"tool_call": {{"name": "<tool_name>", "<arg>": <value>, ...}}}}
using the argument names from the signatures below. One call per reply, at
most {max_rounds} calls per decision. This is a HARD per-decision limit;
each tool result reports used and remaining calls. Results arrive in your
next prompt.
All results use {{"ok", "tool", "state_changed", "result"}}. Failures use
{{"ok": false, "error": {{"code", "message"}}}}. When the budget is
exhausted, tools are disabled and you MUST return a final action JSON.

Perception and retrieval:
- search_frames(query, top_k=5) -> [{{frame_id, score, caption}}]: text search over
  the captions of all collected keyframes. Your main way to find which
  frames may contain the target. Read-only.
- view_frame(frame_id): attach the keyframe's raw RGB image to your next
  input. Use it to verify what a frame actually shows. Read-only.
- use_molmo_point(frame_id, query) -> {{points: [{{pixel: [x, y],
  confidence}}]}}: ask the pointing model to locate the described object
  in ONE keyframe. Returns 0-1000 normalized pixel coordinates AND attaches
  a crosshair-marked evidence image (full frame + zoomed crop) for each
  point. Registers nothing. Skip this tool when you
  can already see the target in a viewed frame and can read its pixel
  position yourself.
- review_crosshair(frame_id, pixel_1000, verdict, reason) ->
  {{frame_id, pixel, verdict, instantiation_allowed}}: this is the REQUIRED
  semantic gate for every pixel. First inspect its attached crosshair image,
  then return exactly one verdict: ACCEPT only when the CROSSHAIR CENTER lies
  on a visible object that matches the full task label; REJECT when it is on
  background, wall, floor, a different object, or outside the object;
  UNCERTAIN when the image cannot establish this. Never use ACCEPT based on
  the caption, the pointing model's text, or a plausible nearby object.
- instantiate_points(frame_id, pixels_1000, label) ->
  {{instances: [{{instance_id, observation_id, frame_id, confidence,
  association, reported}}], pending_confirmation: [...],
  geometry_rejections: [...]}}: register 3D instances at pixel coordinates.
  pixels_1000 is a list of [x, y] in the 0-1000 normalized space (from
  use_molmo_point results or your own reading of a viewed frame); label is
  the full target description. Pixels without crosshair evidence are returned as
  pending_confirmation with the marked image attached (crosshair overlay
  showing exactly which pixel, plus a zoomed crop). After the image is shown,
  call review_crosshair for the SAME pixel. Only an explicit ACCEPT allows a
  later instantiate_points call to register 3D geometry. REJECT and UNCERTAIN
  are semantic_rejections and must never be retried unchanged. geometry_rejections
  are marks with invalid or missing 3D depth. Once you
  have SEEN a matching object in a frame, instantiate it right away: only a
  registered instance is navigable. Never try to walk toward an object
  that exists only in an image. If the object is far away or the evidence
  image is too small to locate it precisely, START_ADJUST with MOVE_FORWARD
  to get closer, then instantiate_points from the new view.
  If pointing infrastructure is unavailable, use_molmo_point/instantiate_points
  returns error code POINTING_BACKEND_UNAVAILABLE. This is not evidence that
  the target is absent: inspect a retrieved frame and use your own pixel
  coordinates with instantiate_points, or continue exploration.

Instance memory:
- search_instances(query, reported=null, top_k=10) -> compact rows: keyword
  search over instance texts; reported may be true, false, or null.
- get_instance(instance_id) -> full record {{id, point, text, reported,
  frame_id, candidate_id, evidence, observation_ids, report_claim_id}}.
  Read-only; returns no image.
- view_instance(instance_id): attach the instance's best available image
  (pointing overlay preferred, else its keyframe). Read-only.
- update_instance(instance_id, text): rewrite the instance's text.

Housekeeping:
- get_agent_status() -> {{num_frames, caption_pending,
  latest_captioned_frame_ids, instances_total, unreported_instances,
  steps_remaining}}: coverage and budget snapshot.
- set_notes(text): overwrite your notes (see Memory above).
- get_action_history(before_step, limit) -> [{{step, action, target_id,
  outcome}}]: your older action history; recent_actions covers the last 3.

After a write tool (update_instance, set_notes, instantiate_points) the
refreshed world state is
included in your next prompt — rely on it, not on the pre-write state.
Stop calling tools as soon as the supplied evidence is sufficient.

# Actions (target_id = an id from the state tables, or null)

- GOTO_INSTANCE id: navigate to an unreported instance's 3D point; an
  arrival decision triggers near it. Does not assert the instance matches
  the task. This is how you reach a target: use instantiate_points
  first, then GOTO_INSTANCE. Approaching a seen-but-not-
  instantiated object through
  frontiers or adjustment does not work.
- GOTO_FRONTIER id: follow the precomputed path to an exploration frontier.
  New frames are collected along the way and listed in new_keyframes at
  the next decision.
- CONTINUE_NAVIGATION (en_route only): keep following the precomputed path
  to the current navigation goal — the orange ACTIVE star on the topdown
  map. That frontier may no longer appear as an fN diamond in the fresh
  candidate table: it is still your current destination, keep moving
  toward it. Prefer CONTINUE_NAVIGATION while navigating unless the
  current view or map shows a clearly better option; choosing any other
  action abandons the current path.
- SCAN: spin 360 degrees in place (12 left turns, four sampled views).
  It only shows what is visible from your current position — it cannot
  reveal other sides of an object, so it cannot verify a candidate. Use
  it to survey your surroundings when the map and captions suggest
  nothing useful; to see an object from another angle, move around it
  instead (GOTO_INSTANCE / START_ADJUST).
- START_ADJUST (takeover): short local adjustment when the camera pose
  needs refinement, or a small turn/step would reveal unseen space. Prefer
  it when no frontier or instance looks promising. During takeover tools
  are disabled: reply with exactly one of MOVE_FORWARD, TURN_LEFT,
  TURN_RIGHT, LOOK_UP, LOOK_DOWN, END_ADJUST per turn; the action executes
  once, then you receive a fresh RGB image. LOOK_UP/LOOK_DOWN tilt the camera
  by 30 degrees without moving the robot and are useful for high/low or
  occluded targets. The harness bounds relative pitch and automatically
  returns the camera to its neutral mapping pose after END_ADJUST. END_ADJUST
  as soon as the view or position is sufficient. Never emit movement actions
  outside takeover, and never START_ADJUST while already adjusting.
  To land a reliable click on a distant or unclear target, START_ADJUST
  with MOVE_FORWARD to approach it directly: the closer view makes the
  target larger in the next frame, so your pixel coordinate (or a follow-up
  instantiate_points) is much more accurate. After END_ADJUST the newest
  view is navigation.current_frame_id: view it, read the target's pixel
  position, and call instantiate_points(current_frame_id, pixels, label).
- REPORT_FOUND instance_id: report the active canonical instance you are
  currently standing next to. target_id is REQUIRED and must equal
  navigation.active_target.id.
  Memory instances are unverified hypotheses: to confirm one, navigate to
  it (GOTO_INSTANCE) and judge from direct observation at close range —
  the arrival RGB or takeover views. Report only what you see at your
  current position, never from captions, memory text, or tool-returned
  images alone. In "many"/"all" modes, never report the same physical
  instance twice.
- FINISH: end the episode (see task modes above).

Cold start: if new_keyframes is absent and there are no instances yet, no
observations have been collected, so retrieval tools will return nothing.
SCAN to look around, or pick a frontier to move to first.

Finally reply with exactly one JSON object and nothing else:
  {{"action": "GOTO_INSTANCE|GOTO_FRONTIER|CONTINUE_NAVIGATION|REPORT_FOUND|SCAN|FINISH|START_ADJUST|END_ADJUST|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|LOOK_UP|LOOK_DOWN",
    "target_id": "<instance id for GOTO_INSTANCE/REPORT_FOUND, frontier id for GOTO_FRONTIER, otherwise null>",
    "reason": "short reason (log only)"}}"""


EVENT_GUIDANCE = {
    "world_state_updated": (
        "\nInstances and reachable frontiers were refreshed together; check "
        "new_keyframes for scenes collected since your last decision and "
        "keep your notes current. Read and, when useful, update instance "
        "texts. Choose globally among GOTO_INSTANCE, GOTO_FRONTIER and SCAN "
        "(REPORT_FOUND and FINISH are also available when their "
        "conditions are met). When choosing a frontier, read the topdown "
        "map: locate the blue AGENT arrow and the purple frontier diamonds, "
        "then combine their spatial layout with path_cost_m. Prefer the "
        "frontier that advances into a new region or unexplored direction; "
        "if the nearest frontiers cluster around you or retrace areas you "
        "already visited, pick one that leads elsewhere instead. When no "
        "frontier or instance looks promising, prefer START_ADJUST: short "
        "local turns/steps actively reveal nearby space and often expose "
        "new frontiers or targets. Use START_ADJUST also when a better "
        "viewing angle would help before making a global choice."),
    "arrival": (
        "\nYou have arrived at the selected candidate; the current RGB is "
        "attached and historical candidate evidence may follow. Judge "
        "whether the candidate satisfies the task from what you see now; "
        "tool-returned images (view_frame, view_instance) are supporting "
        "evidence only. Use REPORT_FOUND with the active instance id when "
        "the target is confirmed by "
        "direct observation at your current position. If the viewing "
        "angle or distance is not good enough to judge, use START_ADJUST "
        "to step or turn for a better view; SCAN is a general panorama "
        "from this spot, not a way to inspect the candidate. When the "
        "current candidate is clearly not the target, leave it unresolved "
        "and choose another instance or frontier."),
    "nav_failed": (
        "\nNavigation to the active target failed: repeated collisions "
        "blocked the path, so that instance was marked unreachable and "
        "removed from the instances table (blocked_target in navigation "
        "records it). The current RGB is attached. Choose a different "
        "instance, a frontier, or SCAN/START_ADJUST to reconsider the "
        "scene. REPORT_FOUND remains valid only if you confirm the active "
        "instance by direct observation at your current position."),
    "scan_complete": (
        "\nA general panoramic scan is complete. The images show the "
        "surrounding environment rather than a target verification "
        "sequence; new_keyframes lists the frames collected since your last "
        "decision. Reconsider all refreshed instances and frontiers; choose "
        "GOTO_INSTANCE, GOTO_FRONTIER, or START_ADJUST when a short "
        "local active-exploration movement would reveal useful nearby "
        "space or correct the current camera pose."),
    "en_route": (
        "\nYou are mid-navigation toward the goal marked by the orange "
        "ACTIVE star on the topdown map (navigation.active_target); the map "
        "and frontier table are fresh, but the star's frontier may have "
        "vanished from the fN diamonds — it is still your current "
        "destination. New keyframes/captions reflect scenes you passed "
        "along the way: check them for the target. Prefer "
        "CONTINUE_NAVIGATION to keep the current path. Choose a different "
        "action only when the current view or refreshed map shows a clearly "
        "better option — e.g. the target visible ahead (then instantiate or "
        "START_ADJUST for a closer look), a promising instance, or a "
        "frontier leading into a new region. Choosing any other action "
        "abandons the current path and restarts planning."),
    "finish_check": (
        "\nFINISH is irreversible. Inspect instance memory and task progress "
        "before deciding."),
    "adjustment": (
        "\nYou are inside the bounded adjustment state (takeover). Tool "
        "calls are disabled here — output exactly one atomic action per "
        "reply. The first extra image is "
        "the latest RGB after the previously executed action; the image labeled "
        "topdown_map is a current local RGB point-cloud bird's-eye view centered "
        "on the blue AGENT marker, with the active target shown as an orange "
        "ACTIVE star. It has no trajectory or occupancy-region coloring; blank "
        "pixels are not proof of free space. Read "
        "world_state.adjustment, especially current_pose, active_target, "
        "previous_action, collision, pitch_offset_steps, and target_budget "
        "information. target_budget is cumulative for this target across "
        "all adjustment sessions: do not spend it on repeated turns or long "
        "searches. A detected "
        "collision means the previous forward action produced no motion; do not "
        "immediately repeat it. active_target "
        "may be null when adjustment was entered for local active exploration; "
        "in that case use fresh RGB and the local map to reveal nearby space, "
        "then END_ADJUST instead of attempting long-range travel. Choose: "
        "MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN, or "
        "END_ADJUST. Use LOOK_UP/LOOK_DOWN for vertical framing, not as a "
        "substitute for changing viewpoint; stay within the pitch limit in "
        "world_state.adjustment. Execute only one "
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


def build_final_decision_prompt(event, world_state, tool_results,
                                max_tool_rounds, allowed_actions):
    """Build a standalone prompt that exposes actions but no callable tools."""
    kept = []
    chars = 0
    for item in reversed(list(tool_results or [])):
        item = str(item)
        if kept and chars + len(item) > 12000:
            break
        kept.append(item)
        chars += len(item)
    kept.reverse()
    transcript = "\n\n".join(kept) or "No tool result was retained."
    if len(kept) < len(list(tool_results or [])):
        transcript = "Earlier tool results omitted for brevity.\n\n" + transcript
    actions = "|".join(sorted(str(action) for action in allowed_actions))
    return f"""# FINAL ACTION ONLY

The hard tool-call limit is {max_tool_rounds} calls per decision, and all
{max_tool_rounds}/{max_tool_rounds} calls have been used. Tools are now
disabled. Do not output tool_call and do not request more evidence. You must
choose the best executable final action from: {actions}.

Use target_id for GOTO_INSTANCE, GOTO_FRONTIER, and REPORT_FOUND. For
REPORT_FOUND it is required and must equal the active canonical instance id.
Other actions use null. REPORT_FOUND is valid only for a target directly
visible in the current RGB, not from historical/tool images alone. FINISH is
irreversible.

Event: {event}

Current world state:
{json.dumps(world_state, ensure_ascii=False)}

Collected tool results (their referenced images remain attached):
{transcript}

Reply with exactly one JSON object and nothing else:
{{"action": "{actions}", "target_id": "id or null", "reason": "short reason"}}
"""
