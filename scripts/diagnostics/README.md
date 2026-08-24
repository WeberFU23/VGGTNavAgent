# Mapping diagnostics

这些脚本用于把 VGGT-SLAM、occupancy 和鸟瞰图从完整 NavAgent 闭环中拆开。
除明确带有 `--reset-map` 的 RGB 回放外，它们不会修改模型、数据集或 benchmark。

- `check_gravity.py`：比较安装俯仰角和轨迹 PCA 的重力估计。
- `check_freespace.py`：拉取点云与位姿，渲染规划栅格。
- `dump_pointcloud.py`：导出 PLY、正交投影和俯视密度图。
- `dump_mapping_snapshot.py`：把逐帧 VGGT 点、位姿和图像行号保存为 NPZ。
- `render_occupancy_snapshot.py`：从固定 NPZ 重建 occupancy 和鸟瞰图，并输出
  冲突统计 JSON，不需要重新运行 Habitat 或 VGGT-SLAM。
- `replay_rgb_sequence.py`：把固定 RGB 目录重新送入 mapping server，用于关键帧、
  子图重叠和光流阈值消融；只有显式传入 `--reset-map` 才会清空地图。
- `collect_navmesh_mapping_sequence.py`：不调用 VLM，利用 Habitat navmesh 采集
  同楼层、多区域、无持续原地转圈的建图轨迹，并保存 GT/SLAM 对比。

从项目根目录运行，并确保项目根目录位于 `PYTHONPATH`。例如：

```bash
PYTHONPATH="$PWD" python scripts/diagnostics/check_gravity.py --port 5555
```

### 无 VLM 的同楼层轨迹采集

该脚本是建图诊断用 oracle，不属于正式 Agent。它只把 RGB 送入
VGGT-SLAM，但使用 Habitat navmesh 规划采集路线；候选最短路径中只要有一个
路径点超出起点高度带，就会被拒绝，以避免楼梯和其他楼层。重复碰撞或连续
转向超过半圈时会放弃当前目标，不会无限原地旋转。

```bash
PYTHONPATH="$PROJECT_DIR:$BENCH_DIR:$BENCH_DIR/evaluation/main" \
python scripts/diagnostics/collect_navmesh_mapping_sequence.py \
  --benchmark-dir "$BENCH_DIR" --scene-root "$SCENE_ROOT" \
  --waypoints 6 --max-steps 400
```

输出包含 `planned_waypoints.json`、`trajectory_trace.jsonl`、抽样 RGB、
`trajectory_data.npz`、`trajectory_gt_vs_slam.png` 和 `summary.json`。
navmesh 和 Habitat 真值只允许用于诊断，不得计入 benchmark 性能。

未显式指定输出参数时，所有图片和点云写入统一目录：
`debug_output/<NAV_RUN_ID>/diagnostics/`。`NAV_DEBUG_ROOT` 可移动整个调试根目录，
`NAV_RUN_ID` 用于隔离不同实验；建议一次运行的 mapping server、agent、benchmark
和诊断脚本使用相同的两个变量，不要分别设置各组件的输出目录。

## 推荐的隔离测试流程

先以 `--no-semantic` 启动 mapping server，排除 caption、pointing 和决策 VLM：

```bash
bash scripts/run_mapping_server.sh --port 5555 --no-semantic
```

使用 `agents.mapping_agent:MappingAgent` 跑一段固定动作序列，并设置
`NAV_SAVE_FRAMES_DIR=rgb` 保存实际送入 VGGT 的图像。episode 结束后导出一次
固定 snapshot：

```bash
PYTHONPATH="$PWD" python scripts/diagnostics/dump_mapping_snapshot.py \
  --port 5555 --stride 6 --out debug_output/$NAV_RUN_ID/diagnostics/map.npz
```

修改 occupancy 代码后，可直接重复生成地图：

```bash
PYTHONPATH="$PWD" python scripts/diagnostics/render_occupancy_snapshot.py \
  debug_output/$NAV_RUN_ID/diagnostics/map.npz \
  --out debug_output/$NAV_RUN_ID/diagnostics/occupancy.png
```

该命令还会生成 `occupancy_layers.png`，分别显示 ground votes、obstacle votes、
geometry coverage 和 traversed 冲突；对应 JSON 保留各层格子数、最大票数和起点
几何 seed 状态。

需要比较关键帧或 SLAM 参数时，再显式清空测试专用 server，并回放同一组 RGB：

```bash
PYTHONPATH="$PWD" python scripts/diagnostics/replay_rgb_sequence.py \
  debug_output/$NAV_RUN_ID/agent/rgb/<episode-id> --reset-map
```

## Occupancy 证据约束

- `free` 只来自 VGGT 地面点支持，不再由轨迹补齐；
- `obstacle` 来自高度层投票及机器人半径膨胀；
- `geometry_observed` 只来自 3D 回投影，不含轨迹和障碍膨胀；
- `traversed` 只表示真实执行历史，不参与 A*、frontier 或 free 判定；
- 浅蓝表示 traversed 但几何未确认可通行；洋红表示 traversed 与 obstacle 冲突。

JSON 报告中的 `traversed_unknown_cells` 和
`traversed_obstacle_conflicts` 应优先用于定位坐标、尺度、地面估计或回环问题，
不能通过把这些格子强制改成 free 来消除。
