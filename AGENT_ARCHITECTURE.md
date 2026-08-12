# VGGT-Nav Agent 架构与弱点总结

> 更新于 2026-08-12。新增语义记忆架构（Phase 1–5 代码完成，待 AutoDL 远端
> smoke 实测；旧 CLIP+SAM3 链路保留在 `NAV_SEMANTIC_BACKEND=clip_sam`（默认）
> 后面，可一键回退）。

## 〇、语义记忆架构（新链路，`NAV_SEMANTIC_BACKEND=semantic_memory`）

```
server 端（vggtslam, py3.11）
  子图处理完成 -> CaptionWorker（异步最低优先级，GPU 忙让路）
    -> Qwen2.5-VL-3B(4bit, vLLM) 生成查询无关详细 caption
    -> BGE-M3 embedding -> CaptionStore（按 episode 落盘）
  ground_object(短语) -> retrieve_captions 文文检索 top-K(默认10,召回优先)
    -> PointingGrounder(Qwen2.5-VL-7B) verify_frame 逐条属性复核（滤假阳性）
    -> point 输出目标像素（point 优先，bbox 交叉验证，多实例）
    -> 11x11 patch 先按 VGGT confidence 过滤再取中位数 -> 3D 点
agent 端（habitat, py3.9）
  命中 -> ObservationLedger 分级置信度：单帧观测 -> belief 锚点
    （frontier 打分先验）；>=2 帧不同位姿独立观测 -> confirmed 进 TSP
  到达例程：0.8m 内 -> pointing+VQA 复核 -> 末端视觉伺服（不看坐标看
    图像，目标近且居中才 TARGET_FOUND，超 8 步退回坐标判定）
  决策层（NAV_DECIDER=vlm）：事件驱动 DecisionLoop，世界状态 JSON
    （dist/path_cost 全预计算）+ 俯视标注地图 PNG -> 受约束 JSON 决策，
    FINISH 硬条件状态机强制，非法回退确定性规则，全程 JSONL trace
```

关键设计：导航端不只用拍照位姿（保留 pointing 像素 -> 点云采深度 ->
3D 点这一跳）；计数信几何不信文本；VLM 不记账不算几何；决策层不进
控制回路；caption 召回优先、精度靠查询条件化复核。

### Flag 总表

| Flag | 默认 | 含义 |
|---|---|---|
| `NAV_SEMANTIC_BACKEND` | `clip_sam` | `clip_sam`（旧）/ `semantic_memory`（新） |
| `NAV_DECIDER` | `rules` | `rules` / `vlm`（LLM 决策层，需 API） |
| `NAV_ORACLE_GEOMETRY` | `0` | GT 位姿消融钩（仅实验分支，打大红 warning） |
| `NAV_VLLM_URL` / `NAV_VLLM_API_KEY` | `http://127.0.0.1:8000/v1` / `EMPTY` | vLLM OpenAI 兼容服务 |
| `NAV_CAPTION_MODEL_PATH` | 空 | Qwen2.5-VL-3B 权重（空 = caption worker 不启动） |
| `NAV_POINTING_MODEL_PATH` | 空 | Qwen2.5-VL-7B 权重（空 = pointing 不可用） |
| `NAV_EMBED_MODEL_PATH` | 空 | BGE-M3 权重（空 = 检索不可用） |
| `NAV_CAPTION_STORE_PATH` | `mapping_diag/caption_store` | caption 落盘目录 |
| `NAV_RETRIEVE_TOP_K` | `10` | caption 检索 top-K（召回优先） |
| `NAV_POINT_PATCH` | `11` | pointing 深度采样 patch 边长 |
| `NAV_POINT_MIN_CONF` | `0.5` | pointing 置信准入/到达复核阈值 |
| `NAV_CONFIRM_MIN_OBS` | `2` | 独立帧观测数升级 confirmed |
| `NAV_MIN_TARGET_PIXELS` | `32` | 小/远目标像素边长下限（两段式兜底） |
| `NAV_MAX_DEPTH_STD_M` | `0.5` | patch 深度方差上限（超出降级 belief） |
| `NAV_SERVO_MAX_STEPS` / `NAV_SERVO_AREA_RATIO` / `NAV_SERVO_CENTER_TOL` | `8` / `0.04` / `0.25` | 末端视觉伺服 |
| `NAV_DECIDER_LOG` | `mapping_debug/decision_trace.jsonl` | 决策 trace |
| `NAV_DECIDER_MAX_TOOL_ROUNDS` | `3` | 决策层工具调用上限 |
| `NAV_FINISH_UNEXPLORED_MAX` | `0.15` | FINISH 硬条件：未探索占比阈值 |

远端部署：权重统一 `/root/autodl-tmp/models/`（ModelScope 镜像下载），
推理走一个 vLLM 服务（客户端优先级队列：决策 > pointing > caption，
同帧结果缓存）；显存预算 VGGT 1B ~10GB + 3B(4bit) ~3GB + 7B(4bit)
~8GB（24GB 单卡）；决策 LLM 走 API（key 走环境变量），不可达自动回退
规则决策。smoke 矩阵：`scripts/run_smoke_matrix.sh`。

**待办（Phase 5 收尾）**：新链路四模式 smoke 不劣于旧链路后，删除
`mapping/semantic.py`（CLIP/SAM3）与 `_ground_object_clip_sam` 旧链路。

---

> 以下为 2026-08-02 版旧架构（clip_sam 链路）记录与弱点分析。


## 一、整体架构

```
┌───────────────────────────── agent 端 (habitat env, python3.9) ─────────────────────────────┐
│ 每步 act()                                                                                   │
│   ├─ feed_frame(rgb) ──────────────► VGGT-SLAM server（子图/关键帧/稠密点云/位姿）           │
│   ├─ explore 模式：随机游走 + frontier 探索（骨架拓扑 + 自由空间栅格 + A*）                    │
│   ├─ 每 NAV_QUERY_INTERVAL 步 ground_object(目标短语)                                        │
│   └─ nav 模式：沿 A* 路径跟随 → 到达后 360° 扫描确认 → TARGET_FOUND                          │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                    │  socket 协议 (mapping/protocol.py, 端口 5555)
┌───────────────────────────── server 端 (vggtslam env, python3.11) ─────────────────────────┐
│  feed_frame → 光流关键帧筛选 → 攒满子图 → 后台线程 VGGT 前向 + 因子图优化（含回环）          │
│  CLIP（常驻，openai/clip-vit-base-patch32）：关键帧向量随子图处理顺带计算                    │
│  SAM3（查询时懒加载实例化）：ground_object 时对 CLIP top-K 帧按需分割                        │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 关键机制

1. **VGGT-SLAM 建图定位（一切空间信息的来源）**
   - agent 不读仿真器 GPS/深度/姿态（RGB-only track），所有位姿、点云、栅格、导航都来自 server。
   - 每步喂帧 → 关键帧（视差筛选）→ 每 ~16 帧一个子图 → VGGT 1B 前向 + 图优化（回环可用时）。
   - `get_all_poses`（位姿）、`get_frame_points`/`get_map_points`（稠密点云）驱动：重力对齐、栅格 A*、骨架拓扑、frontier 探索、实例记忆坐标刷新。

2. **查询时实例化的语义层**
   - **CLIP 常驻**：server 启动即加载；关键帧 CLIP 向量在子图处理时顺带算好（不是查询时才编码）。
   - **SAM3 按需实例化**：`Sam3Grounder` 只建空实例，首次 `ground_object`/`ground_frame` 才 `build_sam3_image_model()` 加载，之后常驻。
   - `ground_object(短语)` 流程：CLIP 检索 top-K 关键帧（默认 `NAV_GROUND_TOP_K=5`，分数平坦时多捞几帧）→ 对 top-K 帧跑 SAM3（多 prompt 变体取最高分）→ mask 投影到 VGGT 点云得 3D 目标点。

3. **多目标状态机**
   - instance memory（confirmed/visited/黑名单，合并半径 0.75m）+ belief（CLIP 先验空间锚点）+ 骨架挂载 + 开放路径 TSP 排序（planner.plan_multi）。
   - 支持 any / many / all；`_should_finish` 纯规则判定（无 confirmed 未访问实例 + frontier 空 + 地图不增长 + 接近最大步数）。

4. **VLM 战略层（当前禁用）**
   - 远端无真实 API（`NAV_VLM_ENABLED=0`），全部走 deterministic fallback：候选取最高 SAM、目标短语正则剥词、FINISH 用规则条件。
   - VLM 的 parse_instruction / choose_candidate / verify_arrival / decide_finish 只通过了本地 mock 单测，从未端到端验证。

### 实测成绩（NAV_VLM_ENABLED=0）

| Episode | 结果 |
|---|---|
| ep0000004 (any basket) | F1=1.0，reached=[133]（起点旁 2m 即目标，天时地利） |
| sink (all) | step 48 检出 SAM=0.828，Precision=1.0，不再误 FINISH |
| ep0000005 (many basket, 层 B) | 多次重跑均 0 命中（见弱点 #1/#2） |

## 二、主要弱点（按影响排序）

### 1. CLIP 语义检索弱（感知层核心瓶颈）
- 所有 `ground_object` 的 top-K 分数压在 **0.21–0.25 平坦区间**；含目标帧（basket 133）也只有 0.216–0.22；无关帧（fid 150）以 0.262 假阳性霸榜。
- 后果：目标在视野里也常召回失败；即使召回也可能是假阳性帧 → SAM 出边缘 mask → 假阳性导航（ep5 step 93 sam=0.500）。
- 类别敏感：sink 类能 0.898，basket 类 0.2x。

### 2. 探索方向随机、无语义引导
- frontier 只按 面积/(距离) × (1+CLIP 信念) 打分，无"目标可能在哪个区域"的先验。
- 实测 agent 起点距目标仅 3.2m 却一路西行、从不去目标区（东侧），500 步覆盖失败；随机环视信息有限（已回滚）。
- 后果：大量零命中的真相是"没走到"而非"没识别"。

### 3. VLM 决策层缺位
- 属性理解缺失：`exactly two` / `gray` / `fabric` 无法区分 → many/all 退化。
- 候选仲裁缺失：假阳性候选直接采信（无证据推理）。
- FINISH 保守：many/all 常跑满 500 步。

### 4. SAM3 措辞敏感 + 单一阈值
- 同一帧 `basket`→0.648，`laundry basket`/`hamper`→NO MASK。
- `NAV_MIN_SAM=0.5` 一刀切：假阳性恰好卡在阈值边缘，无按类别/尺度的校准。

### 5. SLAM 尺度/几何误差（空间决策的地基误差）
- Sim(3) 对齐 RMS 0.33–1.0m；"目标进视野"几何判定与目视矛盾；在线尺度标定 ~4.35 m/unit。
- 后果：3D 目标点、0.75m 实例合并、0.8m 到达判定都压在误差边缘；`resolve_candidate` 不稳会"SAM 命中却拿不到导航点"。

### 6. 单点成绩的假鲁棒
- ep0000004 的 F1=1.0 是"起点即目标 + any 模式"的偶然，同场景 many 即零命中。

## 三、下一步候选方向

1. **CLIP 检索质量**：开放词汇检测器第二通道（OWL-ViT / Grounding-DINO），或检索分数校准。
2. **探索方向先验**：区域优先级 / 覆盖驱动的 frontier 打分，减少"没走到"。
3. **启用 VLM**：配 API 后验证 parse/choose/verify 闭环；在此之前先做假阳性抑制（按 SAM 分数分级 + 到达后扫描拉黑）。
4. **SAM 校准**：类别级阈值 / 多尺度验证。
