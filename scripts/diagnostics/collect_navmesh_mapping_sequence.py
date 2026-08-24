"""Collect a deterministic, same-floor mapping trajectory without a VLM.

This is an oracle diagnostic utility, not a benchmark agent. Habitat's navmesh
provides a repeatable route through several separated regions while RGB remains
the only observation sent to VGGT-SLAM. The output pairs every mapping frame
with Habitat ground truth, actions and collisions so SLAM drift can be measured
independently of exploration quality.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


ACTION_NAMES = {1: "move_forward", 2: "turn_left", 3: "turn_right"}


def _quaternion_xyzw(rotation):
    """Convert Habitat/numpy quaternion variants to an xyzw array."""
    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        imag = np.asarray(rotation.imag, dtype=np.float64).reshape(-1)
        if imag.size >= 3:
            return np.array([imag[0], imag[1], imag[2], rotation.real],
                            dtype=np.float64)
    values = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if values.size != 4:
        raise ValueError("agent rotation must be a quaternion")
    return values


def _forward_from_quaternion(rotation):
    """Return Habitat agent forward (-Z in the local frame) in world XYZ."""
    x, y, z, w = _quaternion_xyzw(rotation)
    axis = np.array([x, y, z], dtype=np.float64)
    forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return ((w * w - np.dot(axis, axis)) * forward
            + 2.0 * np.dot(axis, forward) * axis
            + 2.0 * w * np.cross(axis, forward))


def _signed_heading_error(forward_xyz, desired_xyz):
    """Signed yaw error; positive means Habitat TURN_LEFT."""
    forward = np.asarray(forward_xyz, dtype=np.float64)[[0, 2]]
    desired = np.asarray(desired_xyz, dtype=np.float64)[[0, 2]]
    forward /= np.linalg.norm(forward) + 1e-12
    desired /= np.linalg.norm(desired) + 1e-12
    cross_y = forward[1] * desired[0] - forward[0] * desired[1]
    dot = float(np.clip(np.dot(forward, desired), -1.0, 1.0))
    return float(math.atan2(cross_y, dot))


def _navigation_action(forward_xyz, desired_xyz, turn_threshold_deg=15.0):
    error = _signed_heading_error(forward_xyz, desired_xyz)
    threshold = math.radians(float(turn_threshold_deg))
    if error > threshold:
        return 2, error
    if error < -threshold:
        return 3, error
    return 1, error


def _path_stays_on_floor(points, floor_y, tolerance):
    points = np.asarray(points, dtype=np.float64)
    return bool(len(points) and np.isfinite(points).all() and
                np.max(np.abs(points[:, 1] - float(floor_y))) <= tolerance)


def _similarity_align(source, target):
    """Least-squares Sim(3) alignment, returning aligned source and RMSE."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or len(source) < 3:
        return source.copy(), None
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src, dst = source - src_mean, target - dst_mean
    covariance = dst.T @ src / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(src * src, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) /
                  max(variance, 1e-12))
    aligned = scale * (source - src_mean) @ rotation.T + dst_mean
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return aligned, rmse


def _shortest_path(pathfinder, habitat_sim, start, end):
    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32).tolist()
    query.requested_end = np.asarray(end, dtype=np.float32).tolist()
    if not pathfinder.find_path(query) or \
            not np.isfinite(query.geodesic_distance):
        return None
    return {"distance": float(query.geodesic_distance),
            "points": np.asarray(query.points, dtype=np.float64)}


def _sample_same_floor_route(pathfinder, habitat_sim, start, rng,
                             sample_count, waypoint_count, floor_tolerance,
                             min_separation):
    """Farthest-first route over reachable paths contained in one height band."""
    start = np.asarray(start, dtype=np.float64)
    candidates, seen_cells = [], set()
    for _ in range(max(int(sample_count), 1)):
        point = np.asarray(pathfinder.get_random_navigable_point(),
                           dtype=np.float64)
        if point.size != 3 or not np.isfinite(point).all() or \
                abs(point[1] - start[1]) > floor_tolerance:
            continue
        cell = tuple(np.floor(point[[0, 2]] / 0.75).astype(np.int64))
        if cell in seen_cells:
            continue
        path = _shortest_path(pathfinder, habitat_sim, start, point)
        if path is None or path["distance"] < min_separation or \
                not _path_stays_on_floor(
                    path["points"], start[1], floor_tolerance):
            continue
        seen_cells.add(cell)
        candidates.append(point)

    selected, current = [], start
    while candidates and len(selected) < max(int(waypoint_count), 1):
        ranked = []
        for index, point in enumerate(candidates):
            path = _shortest_path(pathfinder, habitat_sim, current, point)
            if path is None or not _path_stays_on_floor(
                    path["points"], start[1], floor_tolerance):
                continue
            spread = min(float(np.linalg.norm(
                point[[0, 2]] - ref[[0, 2]])) for ref in [start] + selected)
            if spread < min_separation:
                continue
            score = path["distance"] + 0.75 * spread + rng.uniform(0.0, 1e-6)
            ranked.append((score, index))
        if not ranked:
            break
        _score, chosen_index = max(ranked, key=lambda item: item[0])
        current = candidates.pop(chosen_index)
        selected.append(current)
        candidates = [point for point in candidates if np.linalg.norm(
            point[[0, 2]] - current[[0, 2]]) >= min_separation]
    return selected


def _next_path_point(path_points, position, lookahead):
    position = np.asarray(position, dtype=np.float64)
    for point in np.asarray(path_points, dtype=np.float64)[1:]:
        if np.linalg.norm(point[[0, 2]] - position[[0, 2]]) >= lookahead:
            return point
    return np.asarray(path_points[-1], dtype=np.float64)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _save_rgb(path, rgb):
    from PIL import Image
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(
        path, quality=95)


def _sensor(observations, name):
    return observations.get(name) if isinstance(observations, dict) else None


def _reset_environment(env, episode, dataset):
    try:
        return env.reset(episode=episode)
    except TypeError:
        if hasattr(env, "_dataset"):
            env._dataset = dataset
        if hasattr(env, "_current_episode"):
            env._current_episode = episode
        return env.reset()


def _render_trajectory_comparison(path, gt_positions, slam_positions,
                                  slam_frame_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = np.asarray(gt_positions, dtype=np.float64)
    slam = np.asarray(slam_positions, dtype=np.float64)
    frame_ids = np.asarray(slam_frame_ids, dtype=np.int64)
    valid = (frame_ids >= 1) & (frame_ids <= len(gt))
    matched_slam, matched_gt = slam[valid], gt[frame_ids[valid] - 1]
    aligned, rmse = _similarity_align(matched_slam, matched_gt)
    sim3_scale = None
    if len(matched_slam) >= 3:
        source_energy = float(np.sum(
            (matched_slam - matched_slam.mean(0)) ** 2))
        aligned_energy = float(np.sum((aligned - aligned.mean(0)) ** 2))
        if source_energy > 1e-12:
            sim3_scale = math.sqrt(aligned_energy / source_energy)
    figure, axis = plt.subplots(figsize=(10, 9), dpi=170)
    axis.plot(gt[:, 0], gt[:, 2], color="#1565c0", linewidth=2.2,
              label="Habitat GT")
    if len(aligned):
        axis.plot(aligned[:, 0], aligned[:, 2], color="#ff1744",
                  linewidth=1.8, label="VGGT-SLAM (Sim3 aligned)")
    axis.scatter(gt[0, 0], gt[0, 2], color="#00a86b", s=42, label="start")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Habitat X (m)")
    axis.set_ylabel("Habitat Z (m)")
    title = "GT vs VGGT-SLAM trajectory"
    axis.set_title(title if rmse is None else f"{title} — ATE {rmse:.3f} m")
    axis.grid(True, alpha=0.18)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return rmse, aligned, sim3_scale


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default=os.environ.get("BENCH_DIR"))
    parser.add_argument("--config", default="evaluation/main/hm3d_config.yaml")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--scene-root", default=os.environ.get("SCENE_ROOT"))
    parser.add_argument("--scene-id")
    parser.add_argument("--episode-id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--waypoints", type=int, default=6)
    parser.add_argument("--floor-height-tolerance", type=float, default=0.30)
    parser.add_argument("--min-waypoint-separation", type=float, default=2.0)
    parser.add_argument("--waypoint-radius", type=float, default=0.45)
    parser.add_argument("--path-lookahead", type=float, default=0.35)
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0)
    parser.add_argument("--max-consecutive-turns", type=int, default=6)
    parser.add_argument("--max-collisions-per-waypoint", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--save-rgb-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="只验证并保存 navmesh 路线，不连接或重置 mapping server")
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.benchmark_dir or not args.scene_root:
        raise SystemExit("BENCH_DIR/--benchmark-dir and SCENE_ROOT/--scene-root are required")
    benchmark_dir, scene_root = (Path(args.benchmark_dir).resolve(),
                                 Path(args.scene_root).resolve())
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else \
        benchmark_dir / "dataset_semantic"
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = benchmark_dir / config_path
    sys.path.insert(0, str(benchmark_dir))

    from habitat.config import read_write
    from habitat.core.env import Env
    import habitat_sim
    from evaluation.dataset import load_custom_dataset
    from evaluation.main.run_eval import (
        attach_resolved_scene_paths, configure_hm3d_scene_root,
        load_habitat_config)
    from mapping.client import MappingClient
    from runtime_paths import run_debug_path

    config = load_habitat_config(str(config_path))
    configure_hm3d_scene_root(config, scene_root)
    if args.gpu_device_id is not None:
        with read_write(config.habitat.simulator.habitat_sim_v0):
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = \
                args.gpu_device_id
    scene_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if args.scene_id:
        scene_dirs = [path for path in scene_dirs if path.name == args.scene_id]
    if not scene_dirs:
        raise SystemExit("no matching benchmark scene data")
    scene_dir = scene_dirs[0]
    episodes_path, queries_path, goals_path = (
        scene_dir / "episodes.json", scene_dir / "queries.json",
        scene_dir / "goals.json")
    dataset = load_custom_dataset(
        str(episodes_path), str(queries_path) if queries_path.exists() else None,
        str(goals_path) if goals_path.exists() else None, goal_type="description")
    if args.episode_id:
        dataset.episodes = [episode for episode in dataset.episodes
                            if episode.episode_id == args.episode_id]
    if not dataset.episodes:
        raise SystemExit("no matching episode")
    dataset.episodes = dataset.episodes[:1]
    attach_resolved_scene_paths(dataset, scene_root)
    episode = dataset.episodes[0]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(
        run_debug_path("navmesh_mapping", f"{scene_dir.name}_{timestamp}"))
    rgb_dir, trace_path = output_dir / "rgb", output_dir / "trajectory_trace.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix with existing diagnostics: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rgb_every > 0:
        rgb_dir.mkdir(parents=True, exist_ok=True)

    rng, env = np.random.default_rng(args.seed), Env(config)
    client = MappingClient(args.host, args.port, timeout=240.0)
    gt_positions, gt_rotations, actions, collisions = [], [], [], []
    completed_waypoints = abandoned_waypoints = 0
    try:
        observations = _reset_environment(env, episode, dataset)
        pathfinder = env.sim.pathfinder
        seed_pathfinder = getattr(pathfinder, "seed", None)
        if callable(seed_pathfinder):
            seed_pathfinder(args.seed)
        start = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        route = _sample_same_floor_route(
            pathfinder, habitat_sim, start, rng, args.sample_count,
            args.waypoints, args.floor_height_tolerance,
            args.min_waypoint_separation)
        if len(route) < 2:
            raise RuntimeError(f"same-floor route has only {len(route)} waypoint(s)")
        with open(output_dir / "planned_waypoints.json", "w",
                  encoding="utf-8") as stream:
            json.dump({"episode_id": episode.episode_id, "scene": scene_dir.name,
                       "start": start, "floor_height_tolerance":
                      args.floor_height_tolerance, "waypoints": route}, stream,
                      ensure_ascii=False, indent=2, default=_jsonable)
        if args.plan_only:
            print(json.dumps({
                "plan_only": True,
                "scene": scene_dir.name,
                "episode_id": episode.episode_id,
                "planned_waypoints": len(route),
                "output": str(output_dir / "planned_waypoints.json"),
            }, ensure_ascii=False, indent=2))
            return
        server_state = client.get_state()
        semantic_flags = (
            "caption_enabled", "pointing_enabled", "embedding_enabled")
        semantic_state = server_state.get("semantic") or {}
        enabled_semantics = [name for name in semantic_flags
                             if bool(semantic_state.get(
                                 name, server_state.get(name)))]
        if enabled_semantics:
            raise RuntimeError(
                "diagnostic collector requires a --no-semantic mapping server; "
                f"enabled: {enabled_semantics}")
        client.reset_map()
        client.set_episode(f"navmesh_{scene_dir.name}_{episode.episode_id}")
        target_index = consecutive_turns = waypoint_collisions = 0
        server_busy = False

        for step in range(max(int(args.max_steps), 1)):
            if target_index >= len(route):
                break
            if server_busy and not client.wait_idle(timeout=60.0):
                raise RuntimeError("mapping server stayed busy for 60 seconds")
            state = env.sim.get_agent_state()
            position = np.asarray(state.position, dtype=np.float64)
            rotation = _quaternion_xyzw(state.rotation)
            rgb = np.asarray(_sensor(observations, "rgb"), dtype=np.uint8)[..., :3]
            if args.save_rgb_every > 0 and step % args.save_rgb_every == 0:
                _save_rgb(rgb_dir / f"rgb_{step:06d}.jpg", rgb)
            feed = client.feed_frame(rgb)
            server_busy = bool(feed.get("busy"))
            gt_positions.append(position.copy())
            gt_rotations.append(rotation.copy())

            target = route[target_index]
            if np.linalg.norm(target[[0, 2]] - position[[0, 2]]) <= \
                    args.waypoint_radius:
                completed_waypoints += 1
                target_index += 1
                consecutive_turns = waypoint_collisions = 0
                continue
            path = _shortest_path(pathfinder, habitat_sim, position, target)
            if path is None or not _path_stays_on_floor(
                    path["points"], start[1], args.floor_height_tolerance):
                abandoned_waypoints += 1
                target_index += 1
                consecutive_turns = waypoint_collisions = 0
                continue
            steering_point = _next_path_point(
                path["points"], position, args.path_lookahead)
            action, heading_error = _navigation_action(
                _forward_from_quaternion(state.rotation),
                steering_point - position, args.turn_threshold_deg)
            consecutive_turns = consecutive_turns + 1 if action in (2, 3) else 0
            if consecutive_turns > args.max_consecutive_turns:
                abandoned_waypoints += 1
                target_index += 1
                consecutive_turns = waypoint_collisions = 0
                continue
            before = position.copy()
            observations = env.step(ACTION_NAMES[action])
            after = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
            displacement = float(np.linalg.norm(after - before))
            collision = action == 1 and displacement < 0.05
            waypoint_collisions = waypoint_collisions + 1 if collision else 0
            actions.append(action)
            collisions.append(collision)
            row = {"step": step, "episode_id": episode.episode_id,
                   "waypoint_index": target_index, "waypoint": target,
                   "path_distance": path["distance"],
                   "steering_point": steering_point,
                   "heading_error_deg": math.degrees(heading_error),
                   "action": {"id": action, "name": ACTION_NAMES[action]},
                   "position_before": before, "position_after": after,
                   "rotation_xyzw": rotation, "displacement": displacement,
                   "collision": collision,
                   "mapping": {"frame_id": feed.get("frame_id"),
                               "is_keyframe": feed.get("is_keyframe"),
                               "busy": feed.get("busy")}}
            with open(trace_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False,
                                        default=_jsonable) + "\n")
            if waypoint_collisions >= args.max_collisions_per_waypoint:
                abandoned_waypoints += 1
                target_index += 1
                consecutive_turns = waypoint_collisions = 0
            if getattr(env, "episode_over", False):
                break

        client.wait_idle(timeout=120.0)
        client.flush_map()
        client.wait_idle(timeout=300.0)
        slam_poses, slam_frame_ids = client.get_all_poses()
        slam_poses = np.asarray(slam_poses if slam_poses is not None else [],
                                dtype=np.float64).reshape(-1, 4, 4)
        slam_xyz = slam_poses[:, :3, 3] if len(slam_poses) else np.empty((0, 3))
        rmse, aligned, sim3_scale = _render_trajectory_comparison(
            output_dir / "trajectory_gt_vs_slam.png", gt_positions, slam_xyz,
            slam_frame_ids)
        np.savez_compressed(
            output_dir / "trajectory_data.npz",
            gt_positions=np.asarray(gt_positions),
            gt_rotations_xyzw=np.asarray(gt_rotations),
            actions=np.asarray(actions, dtype=np.int8),
            collisions=np.asarray(collisions, dtype=bool),
            slam_poses=slam_poses,
            slam_frame_ids=np.asarray(slam_frame_ids, dtype=np.int64),
            slam_positions_aligned=np.asarray(aligned))
        summary = {"diagnostic_only": True, "uses_habitat_navmesh": True,
                   "uses_vlm": False, "scene": scene_dir.name,
                   "episode_id": episode.episode_id,
                   "planned_waypoints": len(route),
                   "completed_waypoints": completed_waypoints,
                   "abandoned_waypoints": abandoned_waypoints,
                   "mapping_frames": int(len(gt_positions)),
                   "steps": len(actions),
                   "forward_actions": int(sum(a == 1 for a in actions)),
                   "turn_actions": int(sum(a in (2, 3) for a in actions)),
                   "collisions": int(sum(collisions)),
                   "keyframe_poses": int(len(slam_poses)),
                   "trajectory_ate_rmse_m": rmse,
                   "trajectory_sim3_scale_m_per_unit": sim3_scale,
                   "mapping_state": client.get_state()}
        with open(output_dir / "summary.json", "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2,
                      default=_jsonable)
        print(json.dumps(summary, ensure_ascii=False, indent=2,
                         default=_jsonable))
        print(f"saved diagnostics to {output_dir}")
    finally:
        client.close()
        env.close()


if __name__ == "__main__":
    main()
