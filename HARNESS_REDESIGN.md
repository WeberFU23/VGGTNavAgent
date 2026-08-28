# Harness 化改造方案：VLM 自主工具链导航 Agent

状态：核心工具链已落地；当前权威架构见 `AGENT_ARCHITECTURE.md`。本文保留
设计动机与实施记录。已落地的基础改动：world-state 精简
（去掉 recent_events/map_coverage/termination、实例行只留
id/text/path_cost_m、新增 steps_remaining）、检索 top-k 默认 2、
pointing bbox 约束采样。

## 1. 目标与定位

把现在的"固定管线 + VLM 决策点"系统，改造为"harness + 工具箱"系统：
仿照 coding agent，VLM 是唯一的规划者，系统只做三件事——**增强感知、
管理记忆、可靠执行**。VLM 自行决定何时检索、何时核实图像、何时实例化、
何时导航、何时接管微调；不再有任何"到点必做"的固定语义流程。

非目标：不追求 VLM 逐步控制运动；不引入连续动作空间；不改变 VGGT-SLAM
建图内核。

## 2. 设计红线（不可让渡的确定性职责）

1. **运动执行层不放手**：A*、路径跟随、碰撞恢复、离散动作执行全部留在
   `agents/navigator.py`。VLM 只在事件点介入，绝不逐帧调用。
2. **VLM 永不输出世界坐标**：世界坐标、深度、路径代价由工具返回并来自
   VGGT/几何模块。VLM 只引用 ID；唯一例外是 `instantiate_points` 所需的
   0-1000 归一化图像像素。
3. **感知自动化**：caption 生成保持在子图完成后由 caption worker 自动
   触发，不是 VLM 的工具。VLM 的工具是"检索"，不是"记得做 caption"。
   BGE 索引同步自动维护。
4. **覆盖可兜底**：frontier 机制保留，几何/语义覆盖层继续由确定性模块
   维护并呈现在鸟瞰图上；覆盖统计经 `get_agent_status` 按需查询；VLM 不
   探索时系统能检测停滞。
5. **记忆一致性由 harness 保证**：Observation 重放幂等、Canonical Instance
   视觉关联、ReportClaim 一次性报告以及 candidate 重投影均由确定性接口
   维护，决策 VLM 不能直接改 3D 坐标或 reported 状态。

## 3. 工具箱设计

### 3.1 保留的现有工具（不变）

| 工具 | 说明 |
|---|---|
| `search_frames(query, top_k=5)` | BGE 检索关键帧 caption |
| `search_instances(query, reported=null, top_k=10)` | 实例文本关键词检索 |
| `get_instance(instance_id)` | 实例完整记录 |
| `update_instance(instance_id, text)` | 只更新 canonical instance 的工作文本 |

### 3.2 改造的工具

| 工具 | 现状 | 改造 |
|---|---|---|
| `view_instance(instance_id)` | 返回 pointing evidence crop | 保留，证据图带可追踪 image label |
| `SCAN` | 固定 12 步环视动作 | 保留为**动作**（改变朝向，不是纯信息工具），但不再有任何路径强制触发，VLM 自主决定何时环视 |
| `START_ADJUST` | 事件内进入微调 | 保留名称：VLM 显式接管，进入有界离散动作模式（白名单 `MOVE_FORWARD/TURN_LEFT/TURN_RIGHT/END_ADJUST`，`NAV_ADJUST_MAX_STEPS` 上限不变）。**adjustment 期间禁止工具调用**：每轮只出动作+收新 RGB，保持反应式控制节奏 |

### 3.3 新增工具（本方案的核心）

| 工具 | 输入 | 返回 | 实现 |
|---|---|---|---|
| `view_frame(frame_id)` | 关键帧 ID | 该帧原始 RGB（下一轮附图） | server 已有 `get_frame_image`，仅需在决策层包成工具 |
| `point_frame(frame_id, query)` | 帧 ID + 目标描述 | 0-1000 归一化像素 | 只做 pointing，不写实例 |
| `instantiate_points(frame_id, pixels_1000, label)` | 帧 ID + 像素 + 标签 | Observation/canonical instance ID 与关联结果 | 跳过检索和 pointing，直接深度反投影；入库时自动执行三层去重 |
| `ground_target(query, frame_id=null, top_k=2)` | 目标描述 + 可选帧 ID | Observation/canonical instance ID 与关联结果 | 有 frame_id 时严格定帧 pointing+实例化；无 frame_id 时才执行 BGE 检索；入库自动关联 |
| `get_agent_status()` | 无 | mapping/caption/实例/预算状态 | 聚合现有 `get_state` + world-state 字段 |
| `set_notes(text)` | 自由文本（上限约 500 字） | 更新后的 notes | VLM 自己维护的跨决策工作记忆：当前计划、已知结论、待办。harness 只存一个字符串，随每次决策注入 |
| `get_action_history(before_step, limit)` | 分页参数 | `[{step, action, target_id, outcome}]` | 全量动作流水（harness 确定性记录并落盘），不主动下发，VLM 需要回顾早期经历时按需查询 |

工具分层意图：`ground_target` 是高层便捷工具，
`search_frames → view_frame → point_frame → instantiate_points` 是分解后的
精细工具链。VLM
按情况选择：目标明确且着急用前者；需要核实、多目标、或检索结果可疑时
用后者逐步确认。**这正是"很多步骤不一定必要"的落地方式。**

重复实例不再交给决策 VLM 手工清理。入库采用三层机制：同帧重放先做
Observation 幂等；跨帧先以 3D 半径召回少量 canonical 候选，再让专用 VLM
直接比较新旧标注照片，只有 `SAME` 才关联；报告时为 active canonical
instance 创建唯一 ReportClaim。距离只负责召回，类别文本不作为合并条件，
`UNCERTAIN` 会新建实例以避免误合并。

### 3.4 跨决策记忆的组织（不依赖 server 端对话状态）

- **每次决策调用无状态**：prompt = 契约 + 当轮 fresh world-state +
  注入的记忆字段。不累积历史 messages——旧 world-state 里的坐标在
  回环优化后会失效，累积历史既耗 token 又误导决策；
- **VLM 的工作记忆**：`notes` 字段随 world-state 下发，VLM 用
  `set_notes` 自行改写，维护"我在做什么、已排除什么、下一步打算"
  这类提炼后的结论；
- **动作流水**：harness 确定性记录每步 `{step, action, target_id,
  outcome}`（outcome 由执行层给出：arrived / blocked / collision 等）。
  world-state 只带**最近 3 步**（`recent_actions`），更早的全量历史
  存起来不主动给，经 `get_action_history` 按需查询；
- **世界现状**：永远取当轮最新的 world-state，绝不从历史中恢复；
- **事件内多轮工具调用**是一段天然短对话，事件结束即丢弃。

### 3.5 动作集（不变，仍为最终输出）

`GOTO_INSTANCE / GOTO_FRONTIER / REPORT_FOUND / SCAN / EXPLORE / FINISH /
START_ADJUST(→takeover) / END_ADJUST + 微调三动作`。动作与工具的区分保持：
工具改信息、动作改位置/状态。

### 3.6 观察的采集与按需交付（编号制）

仿照 coding agent"不默认塞文件内容、read 时才给"的上下文策略：

- **采集保持连续**：移动过程中（GOTO 路径跟随、takeover 离散动作）RGB
  连续喂给 mapping server，光流关键帧筛选与 caption worker 照旧——沿途
  视野不丢。改变的不是"何时采集"，而是"何时通知 VLM"；
- **RGB-caption 对编号入库**：每个关键帧 = 一个 `frame_id` + 一张 RGB +
  一条 caption（生成中则为 null），存 caption store，全部可按号取回；
- **编号随返回交付**：每次工具调用返回或动作完成后，把这段移动新产生
  的关键帧以 `[{frame_id, caption 摘要(截断), pose}]` 列表带给 VLM。
  VLM 先凭 caption 摘要筛选，真正感兴趣才 `view_frame` 看图；
- **不默认推 RGB**：普通决策的输入是鸟瞰图 + world-state + 编号摘要，
  当前/历史 RGB 只在 VLM 调用 `view_frame` 时进入上下文
  （受 `NAV_VLM_MAX_IMAGES` 上限约束）；
- **例外一（takeover）**：接管微调期间每步仍推送最新 RGB——这是反应式
  控制的反馈回路，不适用按需拉取；
- **例外二（报告亲见约束）**：prompt 契约要求 `REPORT_FOUND` 前必须
  `view_frame` 亲见目标或处于 takeover 中直接观察到目标；先软约束，
  是否硬校验待评测后定；
- **caption 同步返回**：`search_frames` / `ground_target(frame_id=null)` / 移动完成的
  编号通知，在返回前强制等待 caption 队列清空（复用
  `NAV_CAPTION_WAIT_S` 有界等待，上限约 30s）；超时仍返回但标注
  `caption_pending: N`，VLM 可稍后重查。保证"交付给 VLM 的编号，
  caption 一定已就绪"是常态语义。

## 4. 决策循环改造

### 4.1 现状

事件驱动（`world_state_updated / arrival / scan_complete / finish_check /
adjustment`），每事件允许的动作集合硬编码（`AGENT_ARCHITECTURE.md` §7），
工具循环已在 `decision/agent_loop.py` 存在。

### 4.2 目标形态

- 事件入口保留（这是低频决策与高频执行分离的载体），但**放宽每事件的
  动作白名单**：除 `finish_check` 保留 FINISH 相关约束外，其余事件统一
  允许全部动作，由 prompt 契约而非硬编码约束行为；
- 事件输入改为"鸟瞰图 + world-state + 新关键帧编号摘要"（§3.6），
  不再默认附当前/历史 RGB（takeover 除外）；
- 工具调用轮次硬上限为 7；`NAV_DECIDER_MAX_TOOL_ROUNDS` 可调低但不能调高；
  prompt 在初始契约和每轮结果中显示上限与剩余次数。达到上限后切换到不含
  可调用工具的 final-action-only 阶段，残留 `tool_call` 不进入动作校验；
- world-state 增加预算与记忆字段：`steps_remaining`（相对
  `--max-steps`）、`notes`（VLM 自己维护的工作记忆，§3.4）、
  `recent_actions`（最近 3 步动作+结果，更早的经 `get_action_history`
  查询），让 VLM 做规划时有成本意识和连续性。覆盖统计**不进**
  world-state，经 `get_agent_status` 按需查询；
- **冷启动**：episode 开始时没有任何关键帧/caption/实例，首个事件的
  prompt 明确提示此时检索类工具都会返回空，应先 `EXPLORE` 或 `SCAN`
  积累观察；
- **takeover 期间禁止工具调用**（见 §3.2），每轮只输出一个离散动作；
- prompt 重写为 harness 风格：角色（自主导航 agent）→ 可用工具及
  典型用法（含 3.3 的分层建议）→ 记忆工具 → 动作契约 → 当前事件与
  world-state。不再写"推荐链路"式的流程脚本，只给工具能力和约束。

### 4.3 停滞与预算兜底（harness 职责）

- 连续 N 次决策未产生位移且未调用任何工具 → 在下次 world-state 中注入
  `stagnation_warning` 事件提示；
- `steps_remaining` 低于阈值（如 20%）→ prompt 注入收尾提示（优先核实
  已有实例或直接 FINISH）；
- 决策 VLM 连续非法输出 → 现有确定性降级（planner 回退）不变。

## 5. 明确删除的固定行为

| 现状行为 | 改造后 |
|---|---|
| 探索中按 `NAV_QUERY_INTERVAL` 定时 ground 目标 | 删除；VLM 自主决定何时 `ground_target`/`instantiate_points` |
| SCAN 后强制刷新 caption/pointing/实例 | caption 本就自动；实例刷新改为 VLM 调用工具触发 |
| 到达后固定 evidence 展示 + 动作白名单 | 到达事件只带 candidate 编号与摘要，图像由 VLM 用 `view_frame`/`view_instance` 按需取（报告亲见约束见 §3.6）；动作不限死 |
| 探索阶段 pointing 全靠 agent 侧定时触发 | 全部由 VLM 工具调用驱动 |

## 6. 实施步骤（按可独立验证切片）

1. **server 加 `point_frame` RPC**（对指定帧 pointing + 实例化，复用
   `_locate_frame`/`_resolve_point`/`_register_point_candidate`），
   client 加对应方法；单测覆盖。
2. **决策层加 `view_frame` / `point_frame` / `instantiate_points` /
   `ground_target` / `get_agent_status` 工具**，接入 `agent_loop` 工具循环；
   单测覆盖。
3. **观察编号制落地**：移动完成后向决策事件附新关键帧编号摘要；
   `search_frames`/`ground_target`/编号通知改为返回前等待 caption 队列
   清空（有界）；world-state 加预算与覆盖字段；prompt 重写为 harness
   风格（先写 prompt 草案给用户审）。
4. **删除定时 ground 管线**（`NAV_QUERY_INTERVAL` 路径），放宽事件
   动作白名单；保留 `ground_top_k` 等配置给 `ground_target` 用。
5. **对照评测**：TEEsavR23oF 同一批 episode，新旧两版各跑一遍，
   对比成功率、步数、决策 API 调用次数；日志体系（vlm_calls /
   vlm_caption / vlm_pointing / decision_trace）不动，直接可复盘。

每步独立可回滚；1–2 是纯增量，可先行。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 决策 API 调用次数膨胀（成本/延迟） | 工具轮次上限 + 预算字段 + 评测时统计调用数 |
| VLM 不做系统性探索，覆盖漏区 | frontier 持续呈现在地图/状态；停滞检测注入提示；`EXPLORE` 兜底保留 |
| VLM 忘记核实就报告（precision 下降） | prompt 契约要求 report 前必须亲见目标（`view_frame` 或 takeover 中直接观察，§3.6）；是否硬校验待评测后定 |
| 工具链太长导致上下文膨胀 | 实例表有界摘要保留；view_frame 图片计入 `NAV_VLM_MAX_IMAGES` 上限 |
| 7B 本地 pointing 精度瓶颈不变 | 不在本方案解决；bbox 约束采样已兜底，Molmo 替换留作后续 |

## 8. 验收标准

- 同一批 TEEsavR23oF episode 上：成功率不低于现版本，平均步数不显著
  增加（容忍 +10%），决策 API 平均调用数可量化；
- 全部现有单测通过，新增工具各有单测；
- 日志能完整回答"VLM 每次为什么查、查到什么、为什么去那里"。
