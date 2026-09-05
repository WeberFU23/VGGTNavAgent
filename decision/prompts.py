"""决策 VLM 的系统提示词与事件上下文构造。

本模块只描述决策层契约：VLM 能看到什么、能调用哪些工具、各事件允许
采取哪些动作。运行循环、工具执行和动作校验留在 ``decision.agent_loop``，
避免长提示词掩盖控制流。
"""

import json


DECIDER_PROMPT = """# Your task

You control a robot navigating an indoor 3D scene. The task is in
world_state.task.goal, e.g. "Find all bathroom shelf objects." Explore the
scene, find the target object(s), and report each one when you are near it.

SCORING: you are credited for every DISTINCT correctly reported instance,
weighted toward recall. Unreported targets score zero — a target you
confirmed but never reported is a total loss. Reporting is cheap (one
decision) and does not end the episode, so report each confirmed instance as
soon as you are near it; never postpone a report to "finish exploring
first". A report that misses is a small precision penalty; a target never
reported is a full recall loss — when in doubt between reporting a
near/confirmed instance and walking away to explore more, REPORT.

task.mode defines the FINISH rule only:
- "any": report any ONE matching instance, then FINISH.
- "many": report exactly task.expected DISTINCT matching instances;
  task.found counts your accepted reports. FINISH is rejected until the
  count is reached.
- "all": find as many as the step budget allows. The scene total is UNKNOWN
  and you are NOT required to find all of them; "completeness" is not a
  checkpoint you wait for. Report iteratively: report what you confirm near,
  keep exploring, report again. When steps run out you keep only what you
  reported.
FINISH is irreversible: the episode ends immediately.

# What you receive at each decision

1. A JSON world state:
   - task: goal / mode / found / expected, as above.
   - step, max_steps, steps_remaining: your action budget. Plan around it.
   - instances: candidate targets registered so far (see Memory below).
   - frontiers: reachable exploration candidates with path cost, branch id,
     geometry/semantic gain, failure count and novelty. frontier_branches
     groups frontiers by the LOCAL A* path prefix and records successful
     moves, collisions and new keyframes. It is not a permanent room map:
     prefer an untried branch, and avoid a recently blocked/revisited branch
     unless its gain is materially better.
   - navigation: current_pose (x, y, yaw), current_frame_id (the frame id
     of the latest RGB just fed to the map server — the current view; usable
     with view_frame and instantiate_points) and active_target.
   - notes: YOUR persistent working memory (see Memory below).
   - recent_actions: your last 3 high-level actions with outcomes
     (ok / collision / arrived).
   - rejected_spots: pixels on keyframes that were semantically REJECTED
     ({{frame_id, pixel, count, reason}}). The system HARD-BLOCKS re-proposing
     the same spot. Never point at a rejected (frame_id, pixel) again — the
     object may be small or far there; get closer first, then propose from
     the new viewpoint.
   - revisit_targets: keyframes whose 3D depth could not be validated
     ({{frame_id, attempts, dist_m}}). The system is navigating (or has
     navigated) you near the frame's camera pose for a closer look. When
     dist_m is small, propose the target again from your current view.
   - new_keyframes (only when present): {{frame_id, caption excerpt}} rows
     for frames collected since your last decision. Frame images are NOT
     attached — call view_frame(frame_id) to see one.
   - relevant_frames (only when semantic retrieval is available): task-directed
     top caption matches over all collected keyframes. Treat them as hypotheses,
     not detections. Before repeatedly exploring with no usable instance, inspect
     plausible unreviewed rows with view_frame and propose_candidates.
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

Each accepted proposal (from som_pick/instantiate_points) is stored as an
observation. If its 3D point lies near existing instances it is NOT turned
into an instance directly:
you receive a duplicate_review entry with evidence images and must judge
identity yourself with resolve_duplicate (DUPLICATE merges, NEW creates).
The evidence images are wide full-frame views (no zoomed crops) so you keep
the global context; dist_m is the 3D distance between the two stored points,
and pointing localization noise is roughly ±1m — dist_m < 1 with the same
object category, or both markers pointing at the same object in the wide
views, usually means one physical object seen twice.
Whenever you later realize two existing instances are actually the same
physical object (from images, texts, or map positions), call merge_instances
right away — you do not have to wait for a duplicate_review.
observation_count tells how
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
- propose_candidates(frame_id, query) -> {{masks: [{{mask_id, centroid,
  bbox, area_frac}}]}} plus an attached numbered overlay image: segment the
  whole frame into object regions with SAM (no pointing model involved).
  centroid/bbox are 0-1000 normalized, matching the numbers printed on the
  overlay. Then call som_pick with the ids of the regions matching the target.
  Rejected regions (rejected_spots) are filtered out automatically; NEVER
  re-propose a frame whose regions were rejected, and never propose the
  same frame repeatedly from the same viewpoint.

SAM TARGET-SIZE DISCIPLINE: SAM only yields clean, usable object masks
when the object occupies a substantial share of the frame — a chair
roughly 3m away fills a large area, while one 6m+ away either yields no
mask or masks that merge with the rug/floor/wall behind it. Regions below
~0.5% of frame area (area_frac < 0.005) are unreliable: masks in that
range, or an overlay with no mask where you expect the object, mean you
are too far. Do NOT pick or commit such candidates, and do NOT re-propose
another distant frame hoping for better luck — DISTANCE is the problem,
not the frame. Move closer FIRST (GOTO_INSTANCE for a registered
instance; otherwise START_ADJUST with MOVE_FORWARD, or navigate toward
that area), until the target region looks large in the view, THEN
propose. The same logic applies when a prior commit was REJECTED because
the mask fell on the floor/rug/wall beside the object: walk closer and
re-propose from the new viewpoint instead of repeating from afar.
- commit_candidates(reviews, label) -> {{instances, accepted, rejected,
  uncertain, geometry_rejections, duplicate_review}}: reviews is a list of
  {{candidate_id, verdict: ACCEPT|REJECT|UNCERTAIN, reason}}. Only ACCEPT
  proposals are batch-resolved into navigable instances. UNCERTAIN remains a
  non-navigable proposal for later evidence; it is never a navigation target.
  geometry_rejections list ACCEPTED candidates whose 3D depth could not be
  validated — the system AUTOMATICALLY navigates you near that keyframe's
  camera pose for a closer look (see revisit_targets). Do not re-commit the
  same candidate unchanged; wait until you arrive and propose again from the
  new view.
  An ACCEPTED candidate whose 3D point lies near existing instances is NOT
  created immediately: it appears under duplicate_review with its observation_id
  and the neighbors' ids/distances, with wide full-frame evidence images
  attached (dup_new_obs<N> for the new observation, dup_existing_<id> for
  neighbors).
- resolve_duplicate(observation_id, decision, duplicate_of=null, text="") ->
  {{instance_id, resolved}}: verdict for each duplicate_review entry.
  DUPLICATE merges the observation into the existing instance duplicate_of
  (same physical object — compare the wide evidence images and map positions;
  near-identical map locations usually mean the same object even when texts
  differ); NEW creates a separate instance. Until resolved, the observation
  is not navigable and not an instance.
- merge_instances(instance_id, other_instance_id, text="") ->
  {{merged, into, reported}}: merge two EXISTING instances you have judged
  to be the same physical object. other_instance_id is absorbed into
  instance_id (observations, evidence and reported state are kept; optional
  text replaces the surviving instance's text). Use it any time — e.g. after
  comparing view_instance images or noticing two ids at the same map spot.
- review_crosshair(frame_id, pixel_1000, verdict, reason) ->
  {{frame_id, pixel, verdict, instantiation_allowed}}: this is the REQUIRED
  semantic gate for every pixel. First inspect its attached crosshair image,
  then return exactly one verdict: ACCEPT only when the CROSSHAIR CENTER lies
  on a visible object that matches the full task label; REJECT when it is on
  background, wall, floor, a different object, or outside the object;
  UNCERTAIN when the image cannot establish this. Never use ACCEPT based on
  the caption or a plausible nearby object.
- som_pick(frame_id, mask_ids, query, goal_index=null) -> {{proposals: [{{candidate_id,
  frame_id, mask_id, pixel}}]}}: register the picked regions as reviewable
  proposals; each mask's centroid becomes the candidate pixel and the mask
  itself is used for depth sampling. Evidence panels are attached; review
  them and commit with commit_candidates exactly as with propose_candidates.
  goal_index: only in image-goal mode — the index N of the goal_image_N this
  proposal is meant to match.
- instantiate_points(frame_id, pixels_1000, label, goal_index=null) ->
  {{instances: [{{instance_id, observation_id, frame_id, confidence,
  association, reported}}], pending_confirmation: [...],
  geometry_rejections: [...]}}: FALLBACK path — use it only when SAM found
  no matching region on a near frame but you can read the target's pixel
  position yourself from a viewed frame. pixels_1000 is a list of [x, y]
  in the 0-1000 normalized space (your own reading of a viewed frame);
  label is the full target description. Pixels without crosshair evidence
  are returned as pending_confirmation with the marked image attached
  (crosshair overlay showing exactly which pixel, plus a zoomed crop).
  After the image is shown, call review_crosshair for the SAME pixel. Only
  an explicit ACCEPT allows a later instantiate_points call to register 3D
  geometry. REJECT and UNCERTAIN are semantic_rejections and must never be
  retried unchanged. geometry_rejections are marks with invalid or missing
  3D depth. Once you have SEEN a matching object in a frame, register it
  right away: only a registered instance is navigable. Never try to walk
  toward an object that exists only in an image. If the object is far away
  or the evidence image is too small to locate it precisely, START_ADJUST
  with MOVE_FORWARD to get closer, then retry.
  If SAM is unavailable, propose_candidates returns error code
  SAM_UNAVAILABLE. This is not evidence that the target is absent: move
  closer and retry, or use instantiate_points with pixels you read
  yourself, or continue exploration.
  goal_index has the same meaning as in som_pick: pass it whenever this
  instantiation is meant to match a specific goal_image_N.

Instance memory:
- search_instances(query, reported=null, top_k=10) -> compact rows: keyword
  search over instance texts; reported may be true, false, or null.
- get_instance(instance_id) -> full record {{id, point, text, reported,
  frame_id, candidate_id, evidence, observation_ids, report_claim_id}}.
  Read-only; returns no image.
- view_instance(instance_id): attach the instance's best available image
  (evidence overlay preferred, else its keyframe). Read-only.
- update_instance(instance_id, text): rewrite the instance's text.

Housekeeping:
- get_agent_status() -> {{num_frames, caption_pending,
  latest_captioned_frame_ids, instances_total, unreported_instances,
  steps_remaining}}: coverage and budget snapshot.
- set_notes(text): overwrite your notes (see Memory above).
- get_action_history(before_step, limit) -> [{{step, action, target_id,
  outcome}}]: your older action history; recent_actions covers the last 3.

After a write tool (update_instance, set_notes, instantiate_points,
commit_candidates) the
refreshed world state is
included in your next prompt — rely on it, not on the pre-write state.
Stop calling tools as soon as the supplied evidence is sufficient.

# Actions (target_id = an id from the state tables, or null)

- GOTO_INSTANCE id: navigate to an unreported instance's 3D point; an
  arrival decision triggers near it. Does not assert the instance matches
  the task. This is how you reach a target: use instantiate_points
  first, then GOTO_INSTANCE. Approaching a seen-but-not-
  instantiated object through
  frontiers or adjustment does not work. The harness follows the whole
  path by itself and returns control to you on arrival or failure.
- GOTO_FRONTIER id: follow the precomputed path to an exploration frontier.
  The harness executes the whole path by itself — you are NOT consulted
  again until the path completes, fails, or a passing keyframe caption
  strongly matches the task (which interrupts the leg early). New frames
  collected along the way are listed in new_keyframes at that next
  decision. Prefer a frontier in an untried branch rather than
  repeatedly selecting the first marker in a stalled branch.
- SCAN: spin 360 degrees in place (12 left turns, four sampled views).
  It only shows what is visible from your current position — it cannot
  reveal other sides of an object, so it cannot verify a candidate. Use
  it to survey your surroundings when the map and captions suggest
  nothing useful; to see an object from another angle, move around it
  instead (GOTO_INSTANCE / START_ADJUST).
- START_ADJUST (takeover): short local movement under your direct
  control. Legitimate uses: approaching or re-centering a target you can
  already see; recovering a view blocked at point-blank range; or a
  BOUNDED local probe (a few steps through a doorway or past an occluder)
  when no frontier leads where you need. Before entering, decide what
  this session should reveal. Accountability: if your last adjustment
  session revealed nothing new (no new keyframe, frontier, or target),
  do NOT start another one — choose GOTO_FRONTIER or SCAN instead.
  Repeated zero-progress adjustments waste your step budget and are the
  most common failure mode. During takeover tools
  are disabled: reply with exactly one action per turn — MOVE_FORWARD,
  TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN, or END_ADJUST; the action
  executes, then you receive a fresh RGB image. MOVE_FORWARD requires a
  "steps" field: an integer from 1 to
  world_state.adjustment.max_forward_steps saying how many 0.25m forward
  steps to execute in a row (one action type per reply — you cannot combine
  forward and turn in one decision). Steps execute one by one and stop
  early on collision, so prefer several steps over repeated single-step
  replies when the path is clearly free. Turns and tilts always execute
  once. LOOK_UP/LOOK_DOWN tilt
  the camera by 30 degrees without moving the robot and are useful for
  high/low or
  occluded targets. The harness bounds relative pitch and automatically
   returns the camera to its neutral mapping pose after END_ADJUST. END_ADJUST
   is accepted only after its stated success condition has measurable progress:
   a fresh view for verify_instance, a successful move for clear_path, or a
   new mapping keyframe for inspect_sector. Never emit movement actions
  outside takeover, and never START_ADJUST while already adjusting.
  To land a reliable region on a distant or unclear target, START_ADJUST
  with MOVE_FORWARD to approach it directly: the closer view makes the
  target larger in the next frame, so SAM segmentation of it is reliable.
  Whenever the target is small in the current view or its region was
  rejected (see rejected_spots), you MUST get closer before proposing
  again — never retry the same frame from the same viewpoint. After
  END_ADJUST the newest view is navigation.current_frame_id: propose on it
  (propose_candidates) and pick the matching mask with som_pick.
- REPORT_FOUND instance_id: report the active canonical instance you are
  standing right next to. target_id is REQUIRED and must equal
  navigation.active_target.id (or an instance whose dist_m shows you are
  within ~0.5m of it).
  Success is judged by DISTANCE with a tight radius: the benchmark
  accepts a report only from a valid close viewing position of the
  target — practically, get as close to the object as you reasonably
  can while keeping it in view. The stored 3D point is approximate, so
  GOTO_INSTANCE arrival alone is often NOT close enough: after arrival,
  check the current view; if the target is visible a few steps away,
  close in with START_ADJUST small steps until it is right in front of
  you, then report. Report only when you can see the target now or just
  saw it at arm's length during this approach. If no plausible target
  is near the arrival point, leave the instance unresolved and move on.
  In "many"/"all" modes, never report the same physical instance twice.
- FINISH: end the episode (see task modes above).

Cold start: if new_keyframes and relevant_frames are absent and there are no instances yet, no
observations have been collected, so retrieval tools will return nothing.
SCAN to look around, or pick a frontier to move to first.

Finally reply with exactly one JSON object and nothing else:
  {{"action": "GOTO_INSTANCE|GOTO_FRONTIER|REPORT_FOUND|SCAN|FINISH|START_ADJUST|END_ADJUST|MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|LOOK_UP|LOOK_DOWN",
    "target_id": "<instance id for GOTO_INSTANCE/REPORT_FOUND, frontier id for GOTO_FRONTIER, otherwise null>",
    "steps": <integer 1..max_forward_steps; required for MOVE_FORWARD, omit otherwise>,
    "reason": "short reason (log only)"}}"""


EVENT_GUIDANCE = {
    "world_state_updated": (
        "\nFIRST check the current observation: if it already shows a target "
        "object (for image-goal tasks: anything matching a goal_image_N), "
        "instantiate it NOW (propose_candidates + som_pick, or "
        "instantiate_points) before choosing any movement — an object seen "
        "but never instantiated is lost, and walking away may cost you the "
        "view. Then: instances and reachable frontiers were refreshed "
        "together; check "
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
        "frontier or instance looks promising, prefer SCAN or relocate to "
        "the least-bad frontier; START_ADJUST only if your previous "
        "adjustment session made measurable progress (a new keyframe, "
        "frontier, or target) and a few more local steps would clearly "
        "finish what it started."),
    "arrival": (
        "\nYou have arrived at the selected candidate. The current RGB is "
        "attached and historical candidate evidence may follow. The stored "
        "3D point is approximate and reports are only accepted from close "
        "viewing positions of the target, so check the "
        "current view first: if the target is visible a few steps away, "
        "START_ADJUST and close in until it is right in front of you, then "
        "REPORT_FOUND. If no plausible target is near the arrival point, "
        "leave the instance unresolved and choose another instance or "
        "frontier. If this arrival is a geometry "
        "revisit (revisit_targets shows the frame), you are close to the "
        "target's viewpoint: view the current frame and propose/instantiate "
        "it from this near distance."),
    "nav_failed": (
        "\nNavigation to the active target failed: repeated collisions "
        "blocked the path, so that instance was marked unreachable and "
        "removed from the instances table (blocked_target in navigation "
        "records it). The current RGB is attached. Choose a different "
        "instance, a frontier, or SCAN to reconsider the "
        "scene (START_ADJUST only if your previous adjustment session made "
        "measurable progress). REPORT_FOUND remains valid if you are still near that "
        "instance (dist_m within ~0.5m): reports are scored by distance, not "
        "by what you can see."),
    "scan_complete": (
        "\nA general panoramic scan is complete. The images show the "
        "surrounding environment rather than a target verification "
        "sequence; new_keyframes lists the frames collected since your last "
        "decision. Reconsider all refreshed instances and frontiers; choose "
        "GOTO_INSTANCE or GOTO_FRONTIER; START_ADJUST only if your previous "
        "adjustment session made measurable progress and a short local "
        "movement would clearly finish what it started."),
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
        "END_ADJUST. MOVE_FORWARD must include \"steps\": an integer from 1 "
        "to world_state.adjustment.max_forward_steps; each step moves 0.25m "
        "and steps run one by one until done or a collision stops them — "
        "choose several when the way ahead is clearly free, one when unsure. "
        "One action type "
        "per reply; never combine moving and turning in one decision. "
        "Use LOOK_UP/LOOK_DOWN for vertical framing, not as a "
        "substitute for changing viewpoint; stay within the pitch limit in "
        "world_state.adjustment. If this session is revealing nothing new "
        "(no new keyframe, frontier, or target), END_ADJUST now rather "
        "than cruising — and do not start another adjustment session "
        "afterwards; pick a frontier or SCAN instead. Choose END_ADJUST "
        "immediately when the "
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
    if task.get("goal_type") == "image":
        parts.append(
            "\nImage-goal mode: the targets are the outlined objects in the "
            "attached goal_image_N photos (one photo per target; only "
            "not-yet-found targets are attached). task.goal_descriptions "
            "gives a text description per goal_image_N; task.goals_unfound "
            "lists the indexes still to find. You can also spot targets "
            "yourself: whenever you see an outlined object's match in the "
            "current observation or a viewed frame, instantiate it "
            "immediately (som_pick or instantiate_points) and pass "
            "goal_index=N. Before REPORT_FOUND on an image-goal instance, "
            "compare your evidence image against the corresponding "
            "goal_image_N — they must show the same physical object."
        )
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
Other actions use null. MOVE_FORWARD additionally requires "steps": an
integer 1..world_state.adjustment.max_forward_steps (0.25m each, executed
one by one until done or a collision stops them). REPORT_FOUND is scored by
DISTANCE with a tight radius (you must stand at a close viewing position
of the target): arrival at the stored point is usually not close enough —
if you are a few steps away, close in first and report only when the
target is right in front of you. Do not report purely from
historical/tool images. FINISH
is irreversible.

Event: {event}

Current world state:
{json.dumps(world_state, ensure_ascii=False)}

Collected tool results (their referenced images remain attached):
{transcript}

Reply with exactly one JSON object and nothing else:
{{"action": "{actions}", "target_id": "id or null",
  "steps": <required for MOVE_FORWARD only>, "reason": "short reason"}}
"""
