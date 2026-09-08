# 论文实验日志：输出与使用口径

对应 [PAPER_OUTLINE.md](PAPER_OUTLINE.md)。本次改动仅补充诊断采集与归档，不更改提示词、工具反馈、动作规则、成功判定或评分公式，也不增加模型调用。

## 1. 输出文件

正常通过 benchmark 的 `run_eval.py` / `evaluate.sh` 运行即可，无需额外开启日志参数。

| 输出 | 内容 |
|---|---|
| 原有结果 JSON | 保留 `metrics` 和 `episodes`；新增顶层 `manifest`、`episode_journal`、`journal_errors`，以及每个 episode 的 `paper` |
| 同名 `*.episodes.jsonl` | 每完成一个 episode 追加一行，包含 `run_id`、manifest 和完整 episode。批次中断后可保留已完成结果；重复使用路径时根据 `run_id` 分开读取 |
| Agent `decision_trace.jsonl` | schema 2：结构化决策输入、原始模型输出、完整工具结果、候选快照、最终校验动作 |
| Agent `vlm_calls.jsonl` / `vlm_inputs/` | 原有完整 prompt、模型返回与实际发送的图像；增加唯一调用 ID 和决策关联 ID；图片文件名增加会话 ID，防止重跑覆盖 |

`manifest` 记录 benchmark/agent 的 Git revision、工作区变更指纹、配置覆盖、运行参数和 Python 版本。agent 的实际决策模型及主要生效参数还在 `paper.agent_trace` 中。配置记录排除 API key、token、认证头和服务 URL；不会导出整个环境。

JSONL checkpoint 在最终跨 episode 聚合之前写入，Stopping Regret 的跨 episode 回退成本可能尚未补齐。中断后恢复统计时，应按选定 run 取出各行 `episode`，再调用原来的 `aggregate_metrics`，不要直接平均 journal 中的所有字段。

## 2. 各项论文分析读取什么

| 论文分析 | 主要字段 |
|---|---|
| 主结果 | 原有 `sr / precision / recall / f1 / spl_multi` 和聚合 `metrics` |
| 规模变化 | 原有目标数量、场景、描述类型；`paper.episode_definition` 保存完整任务定义，`paper.scene_navigable_area_m2` 保存可用的 navmesh 面积 |
| All-Object Navigation（AllON）停止 | 原有 `report_events / finish_step / all_required_found_step / finished_by_agent / timed_out / stopping_regret_*`；`paper.completion` 辅助归类 |
| 发现池与路径诊断 | 原有搜索覆盖（SC）、TSQ/OQ/PE/RTSR、精确性字段；`paper.steps` 保存逐步原始池与匹配结果、报告前真值完成集合和动态认领配额 |
| `U_t` | 原有 `u_t_series` 保持不变；高层决策时刻的重算使用 `paper.agent_trace.decisions[].agent_snapshot.world_pool` 和 `paper.route_reference.decisions[].candidate_matches` |
| 目标非贪心 | `decisions[].world_state.instances` 中实际呈现的 ID/路径代价，`instances_omitted_ids / instances_unreachable_ids`，及 `output / validation / decision_origin` |
| 首选路线价值 | 当时的候选匹配、完成集合和剩余配额；`paper.route_reference` 的固定目标距离矩阵与各决策起点距离向量 |
| 预算行为 | 最新 `world_state` 的任务进度、预算、候选/frontier 和事件类型；原始 `model_outputs`、处理后 `output`、逐步实际动作及 steering 结果 |
| 轨迹案例 | 原有轨迹/报告；决策工具链、图像、实例/观测身份、报告 claim、`agent_trace.steering_events / steps / observations` |

## 3. 决策快照与动作关联

`decision_id` 标识一次高层决策，同一环境步内发生多次决策也有不同 ID。`context` 带 run/episode/scene/step/event；VLM 调用通过 `context.decision_id` 和 `decision_call_index` 关联。模型调用自己的 `call_id` 也跨进程唯一。

- `world_state`：最终动作前最新的模型可见状态，包含成功写工具后已有的状态刷新。采集器不会为了日志额外刷新地图、计算 A* 或请求 RPC。
- `agent_snapshot.instances`：所有 canonical instance 的稳定 ID、当时的原始点、描述、报告状态、可达性标记、goal index 和证据关联。该快照只进入日志，不进入 prompt。
- `agent_snapshot.world_pool`：同一时刻的估计世界坐标，加 `instance_id`；无变换时明确标记 `transform_unavailable`，原始实例仍保留。
- `model_outputs`：每次模型调用的解析输出，含重试和被拒绝的提议。
- `tools`：调用参数和截断前的完整结构化结果；实际送入模型的反馈仍遵守原有截断规则。
- `output`：DecisionLoop 校验/改写后的高层动作；`decision_origin` 区分 VLM、harness 改写和 fallback。
- `steering_events`：目标/frontier 导航是否成功应用，以及应用后的目标与模式；高层选择不等同于执行成功。
- `agent_trace.steps` 与 `paper.steps`：分别记录 agent 返回动作及 benchmark 实际接受的动作，辅助识别继续跟随、恢复、报告和停止。

逐步 `step` 为 **从 0 开始、执行本次返回动作前的观测编号**。原有轨迹记录同时包含移动后的点，应根据其字段单独对齐。

## 4. 目标选择离线比较

`paper.route_reference` 保存数据，不预先给出贪心/VLM 胜负，不运行另一条策略。

- `target_ids` 指定 `pair_distances_m` 行列顺序。该矩阵为参与目标访问决策的匹配目标的并集；**每次优化只能使用该决策当时的候选集合**，不能使用并集中后来才发现的目标。
- 每个决策保留 `presented_instance_ids / omitted_instance_ids / candidate_matches / reached_target_ids_before / position_world / start_distances_m`，并通过 `decision_id` 关联完整快照。
- 最近目标应按当时 `world_state.instances[].path_cost_m` 确定；没有呈现代价的 omitted 候选不能被默认为最近目标比较中的已知可排序候选。
- 路线重评分使用两种首选共用的 **固定、吸附到 navmesh 的目标中心之间的 geodesic 图**，起点是该决策时刻 benchmark 的世界坐标。这样首选后续优化使用一致的固定节点。
- 这一静态图口径明确写在 `basis` 中，**不等同于主榜报告判定的最近 viewpoint 范围，也不替换已有 SPL 指标**。原始目标位置、全部 viewpoints 和任务信息均保存在 `episode_definition`，供后续检查分析口径。
- 匹配保留阈值内所有候选真值 ID、距离、最近匹配及 `matched/ambiguous/unmatched` 状态；不能将匹配歧义、误识别或不可达的选择无说明删除。
- `null` 距离表示不可达或不可用，不是 0；矩阵附缺失计数。变换缺失、日志缺失和无适用选择事件也要单独报告覆盖率。

Many-Object Navigation（ManyON）应检查池是否满足剩余配额，AllON 只评价已知池的访问；计算 `J(最近首选) - J(VLM首选)`，不解释成完整贪心 agent 的任务成绩。

## 5. 不干扰评测的措施与边界

- 运行中只复制已存在的状态，不新增感知、建图、几何刷新、随机采样或模型调用；原有 `get_target_pool()` 默认契约不变。
- 真值匹配增强、静态距离图与 checkpoint 写入均在原有 episode 指标固定后进行，不计入原有 `episode_elapsed_seconds`，也不占下一 episode 的计时。
- 仍存在轻量的在线复制及决策日志写入开销，不能声称物理耗时完全为零。benchmark 记录 `online_capture_seconds`，后处理记录 `postprocess_seconds`；前者不包含 agent 自身的日志开销。原有 timeout 规则没有更改。
- 诊断接口/文件写入错误不会修改动作或得分；`paper.status / errors / agent_trace_status`、agent 的采集错误和文件警告用于判断数据是否完整。缺少第三方接口时保留主榜结果，标记诊断接口不支持。
- 已有动态认领配额逻辑保留；仅增加其记录，不修改配额或判分。
- 不支持用这些日志恢复完整 SLAM/仿真器状态并执行另一策略；它们服务于本文约定的离线分析。

测试覆盖了模型输入/工具反馈/动作不变、ManyON/AllON 原有指标逐字段相等、冻结快照不被后续状态修改、同一步多决策关联、缺失接口/磁盘失败容错、跨 run checkpoint 和凭据排除。实际 Habitat/VGGT 服务运行仍需在具备服务和数据的环境中验证。






## 3. 三项行为分析的固定口径

### A. `U_t`：实际目标选择机会

- 定义：时刻 t 已实例化、匹配合法目标且尚未报告的目标数。
- 报告出现 `U_t >= 2` 的 episode 比例，以及高层决策时刻比例；可附 `U_t >= 3` 子集。
- 原始 `u_t_series` 按环境步采集，需与高层决策时刻对齐；不把连续路径执行重复计为独立选择，排除 ManyON 配额完成后的尾段。
- 从“池内有多个目标”进一步筛出当前可选、可达且任务相关的目标选择事件，并报告筛选后的覆盖率。
- `U_t` 不覆盖仅存在于 caption/视觉记忆、尚未实例化的目标；低值不能证明任务没有规划空间。

### B. 目标非贪心与离线首选路线比较

**问题：在当时已知的目标中，VLM 的首选是否比最近目标更有利于完成剩余访问？**

1. 取至少两个尚未报告候选的目标访问决策，冻结最终动作前最新的当前位置、候选集、已报告集合及剩余配额。工具写入可能更新状态，不能只取本轮最初输入。
2. 以当时提供的路径代价定义最近目标，与 VLM 所选目标比较；对并列/近似并列设置固定容差。排除强制改写与 fallback。
3. 对路线比较，构造两种首选共用的固定距离图，明确使用估计几何还是评测几何，以及报告位置/到达范围口径。不得纳入未来新发现的目标。
4. 计算 `J_t(a)`：从当前位置先访问 a，再在固定池内完成剩余访问要求的最短总路程。两种首选使用相同的后续优化方式。
5. 计算 `Delta_t = J_t(最近目标) - J_t(VLM目标)`：正值表示 VLM 首选更好，负值表示最近目标更好，容差内为持平。

| 任务条件 | 路线比较口径 |
|---|---|
| ManyON，候选池覆盖剩余配额 | 固定首选后，优化剩余子集与顺序，并满足类别配额 |
| ManyON，仅剩一个目标需求 | 单独报告；同样合法可达时最近目标对纯距离目标已最优 |
| ManyON，搜索覆盖不足 | 报告比例，跳过完整剩余任务路线比较，不把较少目标的路线与完整任务混比 |
| AllON | 访问冻结池内所有尚未报告目标；只解释已知目标的排序，不评价未来探索或停止 |

- 报告有效机会/可比较样本覆盖率、目标非贪心率，以及非贪心事件中的胜/平/负比例和 `Delta_t` 分布。
- 错误实例、不可达或缺少距离的选择单独计数，不能无说明删除；如用真值合法性筛选，明确为条件化诊断。
- 优先在可精确求解的候选规模上做，近似解单独标识；按 episode/场景处理重复决策的相关性。
- **允许结论：首选对应的可实现路线代价更好/更差。禁止结论：VLM 后来实际走出了该最优路线，或完整贪心 agent 的 SR/F1/SPL 更好/更差。**

### C. 预算相关动作变化

- 按 `steps_remaining / max_steps` 分四段，以原始预算为分母，不按实际结束时间归一化。
- 统计高层决策事件，统一动作类别；工具核验若单独分析，应使用独立分母，不与高层动作频次混算。
- ManyON/AllON 分开，报告每段有效决策数及 episode 数；注意晚段只剩长轨迹的样本构成变化。
- 样本允许时按任务进度、可选实例/frontier 数和决策事件类型分层；ManyON 排除配额完成后的尾段，AllON 可使用当时可见的已报告数等状态。
- 主动 FINISH 与预算/超时截断分开，VLM 原始提议与 harness 改写动作分开；不把评测器截断当模型决策。
- 结论表述为“与预算阶段/压力相关的行动调整”；分布不变也不能证明预算没有作用。

## 4. 运行与材料清单

- **闭环运行：** 完整参考 agent 的主评测；预算允许时补一个决策 VLM 的配对运行。
- **日志离线分析：** 本文八项分析；路线比较需额外离线求解，规模分析可能需要 navmesh 面积缓存，不需模型调用。
- **暂不运行：** frontier/记忆/SAM/尺度等结构消融、完整贪心 agent、多预算参数扫描。
- **归档：** 逐 episode 结果 JSON、`decision_trace.jsonl`、含完整 prompt 的 `vlm_calls.jsonl`、配套图像与动作日志，以及模型/配置/代码版本和 run/episode 标识。
- **数据依赖：** `decision_trace` 只有状态摘要，目标代价与最新候选应从完整 VLM 输入恢复；静态路线计算还需当时位置、候选坐标/身份及相应距离信息。缺失时报告不可分析数量，不用未来状态补齐当时知识。

相关设计：[PAPER_OUTLINE.md](PAPER_OUTLINE.md)（benchmark 定义与论文框架）、[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)（agent 系统架构）。