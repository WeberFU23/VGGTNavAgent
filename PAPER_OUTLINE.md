# 论文

## 1. 正文章节安排

### 1. Introduction

Embodied navigation is a foundational capability for embodied tasks ranging from household assistance to warehouse order-picking because agents typically need to reach a target location before executing downstream manipulations. As embodied intelligence advances, a series of benchmarks have emerged to quantify how well agents navigate.

Existing embodied navigation benchmarks have progressively extended the task horizon. Starting from single-goal episodes [Habitat ObjectNav], a line of works has moved toward sequential multi-goal navigation, in which the agent is assigned the next goal upon reporting completion of the current one, while retaining its memory of the environment across goals [MultiON, GOAT, LH-VLN]. This lifelong setting substantially extends the temporal scope of agent operation, making environmental memory retention and long-horizon efficiency important evaluation targets. However, regardless of how long the task horizon grows, the task structure itself remains fixed: goals, their order, and count are always given to the agent in advance.

Real-world service and household tasks, however, expose a property the line leaves untouched: how much of the task the agent must decide for itself. For instance, before hosting guests, a robot is asked to "find three chairs and bring them over"; before doing laundry, to "collect all the clothes scattered around the house"; in a restaurant, to "round up all the dishes." Serving such requests requires the agent to decide which subset of candidate instances to visit and in what order, to track which instances it has already found so that each is reported exactly once, and to judge whether the search is complete when the total count is not given. Existing sequential multi-goal benchmarks exercise none of these capabilities: (i) the goal order is prescribed by the instruction, so the agent never selects a subset or ranks its visits, whereas unordered multi-instance tasks induce a combinatorial space of subset choices and visit orderings; (ii) working-memory demands remain weak, as the agent tracks only the current sub-goal rather than the growing set of already-found instances; (iii) termination is never in question, since the goal count is known and a scheduler hands over the next sub-task; and (iv) existing metrics score the outcome of ordered execution but cannot attribute failure to a specific cause: whether the agent failed to recognize the targets, chose a poor subset, ordered a suboptimal visit, or stopped at the wrong time. We argue that these capabilities constitute a largely unexplored dimension for navigation benchmarks — decision autonomy: beyond how long agents operate, this dimension asks how much of a multi-object task the agent must plan and decide on its own.

To close this gap, we present XXX-Bench, a benchmark for multi-object navigation built around two tasks. In Many-Object Navigation (ManyON), the agent must find $k$ distinct instances that match the goal, in a visiting order of its own choosing, within a step budget. In All-Object Navigation (AllON), the total number of instances is withheld, and the agent must find every instance and actively signal completion once it judges the search to be complete. In both tasks, goals can be specified by a category name, a natural-language description, or a reference image. XXX-Bench comprises 750 episodes built on 36 scenes from the HM3D dataset, with controlled target counts, scene scales, and description types.

The evaluation protocol in XXX-Bench mirrors the four gaps identified above. A compact leader-board (Success Rate, Success Rate by Length, and de-duplication-aware Precision/Recall/F1) scores the final outcome, while process-level diagnostics attribute performance to its sources: a multiplicative decomposition of path inefficiency into search-side (coverage, search quality) and planning-side (target selection, ordering) factors measures how well the agent exploits its ordering freedom; a repeated-target-selection rate probes instance memory; and a stopping-regret measure evaluates termination decisions in AllON. Choice-opportunity statistics further verify that genuine decision spaces actually arise rather than being degenerate. Each diagnostic corresponds to a failure mode we observe in practice, rather than a purely theoretical construct.

We further contribute an auditable reference agent that serves both as a baseline and as a measurement instrument for the diagnostics above. The agent follows a harness design: the VLM performs only high-level cognition — planning, retrieval, verification, and arbitration — while deterministic modules handle perception, memory, and execution, all exposed as callable tools with fully traced decisions. Experiments show that XXX-Bench is far from saturated: the reference agent reaches only [X]\% success on AllON. Its failures are attributable rather than opaque — choice opportunities with multiple instantiated candidates arise in [X]\% of decision steps, non-greedy target choices carry measurable cost consequences, efficiency losses decompose cleanly into search-side and planning-side terms, and stopping emerges as a genuine decision point with non-trivial regret. 

To summarize up, our contributions are threefold:
\begin{itemize}
\item \textbf{Task and data.} We propose ManyON and AllON, two multi-object navigation tasks that operationalize decision autonomy — free visit order, mandatory deduplication, and, in AllON, self-determined termination — and construct XXX-Bench with 750 episodes across 36 HM3D scenes, supporting category, language, and image goal specifications.
\item \textbf{Evaluation and diagnostics.} We design a two-level protocol that pairs leaderboard metrics with process-level diagnostics, enabling failures to be attributed to search, selection, ordering, memory, or stopping, rather than collapsed into a single opaque score.
\item \textbf{Reference system and empirical findings.} We contribute an auditable harness-style VLM agent and show that XXX-Bench is unsaturated, creates genuine and consequential decision spaces, and supports precise localization of where current agents fall short.
\end{itemize}


### 2. Related Work

**2.1 Embodied Navigation Benchmark**

Existing embodied navigation benchmarks have mainly evolved along two lines, each substantially broadening what navigation systems are evaluated on. The first line expands how goals are specified: from object categories in a closed label set [Habitat ObjectNav], to open-vocabulary categories [HM3D-OVON], to goals given by category names, natural-language descriptions, or reference images [GOAT], and even to sounding targets localized from acoustic cues [SoundSpaces, AVLEN]. This line has freed navigation from fixed label sets, allowing goals to be expressed in whatever form users naturally provide. Yet while evaluation has shifted toward goal grounding and perceptual generalization, the underlying task structure — reaching one designated target at a time — remains unchanged. The second line extends the task horizon: from single-goal episodes to sequential multi-goal navigation, in which the agent is assigned the next goal upon reporting completion of the current one while retaining its memory of the environment across goals [MultiON, GOAT, LH-VLN]. This lifelong setting extends the temporal scope of agent operation, making environmental memory retention and long-horizon efficiency important evaluation targets.

Despite expanding different axes, both lines rest on a common structural assumption: the task structure is externally given. The goals, their order, and their count are prescribed by the benchmark, and the metrics score the outcome of executing that prescription rather than the decisions that produced it. Our work is complementary to both lines. We hold goal specification and task horizon as orthogonal axes — AutoNav-Bench supports category, language, and image goals in a long-horizon setting — and vary instead the degree of decision autonomy: which instances to visit, in what order, with what memory of past visits, and when to stop. This isolates a capability that neither line exercises or measures, and allows our tasks to be combined with any point along the two established axes.

**2.2 VLM for Exploration and Navigation**

- VLM 在解决具身导航问题上有前景，但对空间的理解和记忆是 VLM 的弱点，需要外部的表征来辅助 VLM。现有方法从表示形式和交互范式两个互补方向展开，但均未解决开放世界多实例收集的核心挑战。
- 在场景表示方面可分为三种范式(1) 3D Scene Graph：MSGNav、SG-Nav和 PSG-Nav 将环境建模为物体节点和关系边，使 agent 能利用物体间的语义和空间关系进行导航。然而，scene graph 的预构建开销大，且将复杂空间关系简化为离散文本描述或RGB图片，难以支持精细的空间推理。3D-Mem 提出用 Memory Snapshot 替代 scene graph，虽然保留了更丰富的视觉信息，但其高层决策仍基于离散图片，缺乏对全局空间拓扑的认知。(2) BEV / 语义地图：TopV-Nav将鸟瞰图与物体 bounding box 作为 MLLM 的输入，利用其空间推理能力进行零样本目标导航。3DGSNav、IntentNav和 VLFM也在 BEV 或语义地图上标注目标点或前沿点来引导 VLM 决策。虽然这种压缩简化了空间规划，但在需要实例级视觉验证时，VLM 无法从压缩后的 BEV 中恢复原始视觉证据。(3) 3D 特征场 / Query-based：MTU3D  提出在线 query 表示学习，从 RGB-D 流中实时提取物体 query 和前沿 query，统一了 grounding 与 exploration 的优化目标。但其表示是压缩后的隐式向量，语义不可读，且依赖深度传感器和已知位姿，限制了在纯 RGB 场景下的适用性。本文基于 VGGT 的单目重建直接维护原始点云，配合 VLM 生成的自然语言 caption，在 query 时刻动态实例化语义区域——既避免了全图预构建的开销，又保留了完整几何与显式可读语义，使 VLM 能够直接基于点云密度和 caption 检索进行空间聚类推理与覆盖评估。从结构化场景表示上。
- 在交互范式方面，现有方法通过外部记忆或工具调用扩展 VLM 能力，但交互方式较为僵化。MemoNav、RAVEN 等将历史观测以帧、片段或 embedding 形式存入外部记忆，需要时检索召回——记忆内容由感知流水线自动写入，VLM 只能被动接受召回结果，无法追问"某个候选是否为新实例"，更不能修正记忆本身。 AgenticNav 将深度查询、视觉回忆、移动等感知与动作原语封装为工具，让 VLM 在"思考-行动"循环中主动与环境交互；但这些工具只作用于环境与即时感知，场景记忆仍由流水线被动累积——VLM 不能对记忆本身追问、核实与修正，例如裁决两个观测是否同一实例、撤销一次错误的记忆写入。值得注意的是，harness 式的工具组织在操作领域已被验证有效（Guava），但在导航领域尚未与可写的场景记忆结合。本文的工具作用对象不止环境与即时感知，还包括两层可读写的记忆：一是 VGGT-SLAM 在线重建的点云空间记忆——frontier、覆盖状态、路径代价与 BEV 态势都由它提取和预计算，目标候选经 SAM 分割反投影实例化为其中的 3D 位置；二是对 VLM 开放写入的语义记忆——感知产出经 VLM 审核后才成为可导航实例，重复实例可合并、文本可修订、notes 可自行维护。VLM 据此按需索取信息、核实候选并参与记忆维护，形成"调查-核实-入账"的主动闭环，从而在多实例收集任务中做出有全局意识的主动决策。

### 3. Benchmark 

**3.1 任务定义**

- 任务规则定义（观测输入、动作输出、成功标准、agent规格）。
- ManyON：数量已知、顺序未知。
- AllON：总数未知，要求找全并主动停止。

**3.2 数据构建与统计**

- 数据集生成pipline：场景来源、目标标注、查询生成、可达性筛选、自然语言生成。
- 数据统计：场景数、episode 数、目标数量、目标分布、目标类型。

**3.3 评测指标体系**

| 层级 | 指标 | 定义目的 |
|---|---|---|
| 主结果 | SR、F1、SPL-multi；补充 Precision/Recall | 衡量成功、报告准确完整性及成功条件下的路径效率 |
| 过程诊断 | 搜索覆盖 SC、搜索质量 SQ、TSQ、OQ、PE、RTSR、停止相关量、NE、OSR | 区分总分背后的不同表现特征 |
| 行为证据 | `U_t`、目标非贪心与路线比较、预算分段动作分布 | 观察选择机会、首选目标价值和行动调整 |

- 正文给出核心定义；求解近似、匹配阈值、缺失值和聚合规则放附录。

### 4. Reference Agent

参考 agent 不是来刷分的 SOTA 竞争者，它承担两个功能：**可解性 / 有效性证明**（任务是可以被一个合理系统做出真实决策的）和**诊断探针**（它的机制与留痕恰好产出 §5 所有诊断指标的数据）。全节每一小节都应能回扣这两个功能之一。

**4.1 Design Rationale：为什么必须是 harness**（动机，全节的智力核心）

从 ManyON/AllON 的任务需求出发，论证两个极端都不行：端到端 VLM 扛不住长时程上下文与精确几何，纯启发式写不出"哪个候选最像目标、能不能停"的通用规则。harness 恰好切走中间地带。五条论证压缩成一段设计原则：

- 长时程：episode 长达数百步，端到端 VLM 逐步决策成本高、视觉上下文膨胀，早期观察被挤出窗口，且其中大部分决策并不重要；
- 记账：要求可靠维护"找到过哪些、是否重复、哪里还没探索"，VLM 在长上下文中自行记账不可靠、不可审计；
- 精确几何：尺度、路径代价、可达性、去重半径都是连续数值，VLM 的数值估计不可靠；
- 语义判断：候选选择、属性判别、终止判断依赖语义常识与不确定推理，启发式写不出通用规则；
- 搜索与重访：已看过的区域需要能被语义回访，冷启动需要无目标探索。

harness 的定位由此导出：感知增强、记忆管理与动作执行由确定性模块承担并包装为工具，VLM 只做高层认知（规划、检索、核实、裁决）。

**4.2 Memory Architecture：三层记忆 + world_state 视图**（v2 改造的成果，显性呈现）

- 空间记忆：VGGT-SLAM 点云 / 占据栅格 / 关键帧库 / 尺度标定；
- 语义记忆：caption_store；instance_store 的 Observation / Instance / ReportClaim 三层；
- 工作记忆：agent_notes、action_log、decision_window；
- 代码方与 VLM 的可写性分明：底层结构由代码维护保证正确性，语义层对 VLM 开放写入（审核候选、合并重复、修订文本、维护 notes）；
- world_state 不是独立记忆，而是每次决策对三层记忆的确定性重建视图。这是"记忆是 harness 强项"这条论点的具体形态，也是和 MemoNav 式被动检索的区别所在。

**4.3 Spatial Memory：VGGT-SLAM 作为空间记忆载体**

不只介绍建图，重点是"全部空间状态同源"：frontier、覆盖、路径代价、BEV、实例 3D 坐标都从同一个在线点云派生；尺度标定、鬼影清除、反投影实例化由确定性模块吸收误差，VLM 永不输出坐标和尺度。感知 → 反投影实例化链条（caption 检索 → 查看帧 → pointing / SAM 选 mask → 反投影取 3D）在本节给出。误差处理压缩、细节放附录。

**4.4 Decision Loop：工具集 + 调查—核实—入账 + 事件驱动**（与 related work 对位的差异化贡献）

架构图放本节开头总览。

- 工具集按读 / 写 / 执行分组：读（caption 检索、查看帧、实例检索与检查、地图状态），写（实例化、文本修订、重复合并、notes），执行（GOTO、SCAN、REPORT、FINISH、ADJUST）；
- 调查—核实—入账：感知产出只是候选，经 VLM 审核（接受 / 拒绝 / 不确定）才成为可导航实例；重复实例可合并、文本可修订、notes 自维护。VLM 按需索取信息并参与记忆维护，而不是被动接收召回——与 related work 对 AgenticNav 的批评口径一致（那里写对比，这里给机制细节）；
- 事件驱动决策密度：GOTO 执行到底，仅到达 / frontier 耗尽 / 检索命中等事件点交还 VLM，决策密度降至每 300 步 12–20 次，VLM 的上下文只花在真正的决策点上（正式写之前用正式实验日志核数字）。

**4.5 Instrumentation：机制 ↔ 考点 ↔ 指标对齐**（全节的收尾，通往 §5 的桥）

benchmark 主导论文里最关键、也最容易被漏掉的一段。一张小表：

| 机制 | benchmark 考点 | 产生的指标 / 证据 |
|---|---|---|
| 实例池 + 预计算路径代价 | 目标选择成为显式决策点 | U_t 的结构来源（§5.6）；首选路线代价比较 |
| FINISH 显式化 | 终止成为可观测决策而非隐式超时 | §5.4 停止分析 |
| 三层去重 + 报告幂等 | 重复报告 FP 的防线 | 诊断指标归因 |
| 全程留痕（决策、工具、审核、尺度） | — | §5 所有诊断指标的数据来源 |

写完这段，reader 自然明白为什么 experiments 里那些指标"有数可算"。

**写作注意（降级或不写）**

- 28 个配置项、API 重试、截断口径、尺度标定细节 → 附录或实现细节小节，一句话带过；
- 不要声称 agent 本身的创新性超过"一个合理的 reference 设计"——它的贡献是和 benchmark 联合呈现的，单独拔高反而给 reviewer 攻击面；
- mermaid 图换成一张正式架构图（三层记忆居中，工具 / 执行两侧，world_state 视图标注），最好再加一个 episode 内 loop 的时序小图或 running example。

### 4. Reference Agent (English)

The reference agent is not a SOTA competitor chasing scores; it serves two functions: **solvability / validity evidence** — the tasks admit genuine decision-making by a reasonable system — and a **diagnostic probe** — its mechanisms and traces produce exactly the data behind every diagnostic metric in §5. Every subsection should tie back to one of these two functions.

**4.1 Design Rationale: Why a Harness** (motivation; the intellectual core of the section)

Starting from the requirements of ManyON/AllON, we argue that both extremes fail: end-to-end VLMs cannot sustain long-horizon context or precise geometry, while purely heuristic pipelines cannot express general rules for "which candidate is most likely the target" or "whether it is safe to stop". A harness occupies exactly this middle ground. Five arguments, compressed into one design principle:

- Long horizon: an episode runs for hundreds of steps. Step-by-step end-to-end VLM decisions are expensive, the visual context grows without bound, early observations are pushed out of the window — and most of those decisions do not matter.
- Bookkeeping: the task requires reliably maintaining "what has been found, what is a duplicate, what remains unexplored". A VLM keeping such ledgers inside its own context is neither reliable nor auditable.
- Precise geometry: scale, path costs, reachability and deduplication radii are continuous quantities that VLMs estimate unreliably.
- Semantic judgment: candidate selection, attribute discrimination and termination decisions rely on semantic common sense and reasoning under uncertainty — heuristics cannot write general rules for these.
- Search and revisit: regions already seen must be semantically revisit-able, and cold start requires goal-free exploration.

The harness positioning follows directly: perception augmentation, memory management and action execution are delegated to deterministic modules and exposed as tools; the VLM handles only high-level cognition (planning, retrieval, verification, adjudication).

**4.2 Memory Architecture: Three-Layer Memory + the World-State View** (the outcome of the v2 redesign, presented explicitly)

- Spatial memory: the VGGT-SLAM point cloud / occupancy grid / keyframe store / scale calibration.
- Semantic memory: caption_store; the three levels of instance_store — Observation / Instance / ReportClaim.
- Working memory: agent_notes, action_log, decision_window.
- Writability is split cleanly between code and VLM: underlying structures are maintained by code for correctness, while the semantic layer is open to VLM writes (auditing candidates, merging duplicates, revising texts, maintaining notes).
- world_state is not an independent memory but a deterministically rebuilt view over the three layers at every decision. This is the concrete form of the claim that memory is the harness's strength, and the substantive difference from MemoNav-style passive retrieval.

**4.3 Spatial Memory: VGGT-SLAM as the Spatial Memory Carrier**

Not just a mapping introduction — the point is that all spatial state shares one source: frontiers, coverage, path costs, the BEV image and instance 3D coordinates are all derived from the same online point cloud. Scale calibration, ghost-layer removal and back-projection instantiation absorb reconstruction errors inside deterministic modules; the VLM never outputs coordinates or scale. The perception → back-projection instantiation chain is presented here (caption retrieval → view frame → pointing / SAM mask selection → back-project to 3D). Keep error handling brief; details go to the appendix.

**4.4 Decision Loop: Tools + Investigate–Verify–Commit + Event-Driven Control** (the differentiated contribution aligned with related work)

Place the architecture figure at the start of this subsection as an overview.

- Tools are grouped as read / write / execute: read (caption retrieval, view frames, instance search and inspection, map status), write (instantiation, text revision, duplicate merging, notes), execute (GOTO, SCAN, REPORT, FINISH, ADJUST).
- Investigate–verify–commit: perceptual output is only a candidate; it becomes a navigable instance only after VLM review (accept / reject / uncertain). Duplicates can be merged, texts revised, notes self-maintained. The VLM pulls information on demand and participates in memory maintenance instead of passively receiving recalls — consistent with the criticism of AgenticNav in related work (the contrast goes there; the mechanism details go here).
- Event-driven decision density: GOTO executes to completion and control returns to the VLM only at event points (arrival / frontier exhaustion / retrieval hits). Decision density drops to 12–20 decisions per 300 steps, so VLM context is spent only on real decision points (verify these numbers against the formal experiment logs before writing).

**4.5 Instrumentation: Mechanism ↔ Capability ↔ Metric Alignment** (the close of the section; the bridge to §5)

The most critical — and most easily omitted — paragraph in a benchmark-led paper. A small table:

| Mechanism | Benchmark capability | Resulting metric / evidence |
|---|---|---|
| Instance pool + precomputed path costs | Target selection becomes an explicit decision point | Structural source of U_t (§5.6); first-choice route cost comparison |
| FINISH made explicit | Termination is an observable decision, not an implicit timeout | §5.4 stopping analysis |
| Three-level deduplication + report idempotency | Defense line against duplicate-report false positives | Failure attribution to the review / dedup stages |
| Full tracing (decisions, tools, reviews, scale) | — | Data source for every diagnostic in §5 |

After this paragraph, the reader naturally understands why the experiments have "numbers to compute" at all.

**Writing notes (demote or omit)**

- The 28 configuration items, API retries, truncation policy and scale-calibration details → appendix or an implementation-details subsection, one sentence each at most.
- Do not claim the agent itself is more innovative than "a reasonable reference design" — its contribution is presented jointly with the benchmark; inflating it only gives reviewers attack surface.
- Replace the mermaid sketch with a proper architecture figure (three-layer memory at the center, tools and execution on the sides, world_state annotated as a view), ideally plus a small timeline of the in-episode loop or a running example.

### 5. Experiments

**5.1 Setup**

- 测评的方法（3dmem、MTU3D、ours）

**5.2–5.8 分析**

| 小节 | 放哪些实验与指标 | 推荐呈现 | 结论与指标的作用 |
|---|---|---|---|
| **5.2 Overall Performance** | ManyON/AllON 分别报告 SR、F1、SPL-multi、P/R、样本数 | 主结果表 | 建立参考性能；用部分完成指标区分“同样失败但完成程度不同”。单系统低分不能独自证明 benchmark 有效或普遍困难 |
| **5.3 Scaling with Task Difficulty** | 按目标数量、场景规模、描述类型分组的 SR、Recall、SPL-multi | 少量分桶曲线及置信区间 | 展示性能如何随目标数量变化；复杂的任务对agent架构要求更高；控制/分层报告每个episode |
| **5.4 Completion versus Stopping in AllON** | 找齐率、正式 SR；找齐并停止/找齐被截断/未找齐主动停/未找齐被截断四类比例；找齐后额外步数、提前停止遗漏数；Stopping Regret 作补充 | 四类结果分布 + 停止成本图 | 分开观察搜索完成与宣布完成；原始量解释问题，Regret 概括代价。未找齐被截断不能直接归因为停止错误；找齐后继续探索不一定意味着当时已有充分停止证据 |
| **5.5 Decoupling SPL ：Searching and Planning** | 搜索质量 SQ、TSQ、OQ、PE 四因子 + RTSR（单列） | 开篇给出分解等式；log 损失堆叠分解图（乘性转加性）；因子几何均值表；搜索/规划两轴散点 | 核心等式 $SPL\text{-}multi = S \times (SQ \times PE) \times (TSQ \times OQ)$：完成度由 $S$ 承担，效率项乘法归因到搜索侧（发现候选 + 探索移动开销）与规划侧（子集选择 + 访问排序），"同样的低 SPL"由此可区分"没找到"与"没排好"。PE 在确定性执行器下主要反映探索移动，归搜索侧而非低层执行；AllON 无 TSQ（分解只三项）、搜索欠覆盖时 SQ 记 null 并以 SC 单列、精确/近似 TSP、池不足/未完成单独标识；四因子是归因不是因果失败占比；RTSR 属目标记忆，不进乘法链 |
| **5.6 Target Choice Opportunities and Value** | `U_t` 机会覆盖率；目标非贪心率；离线首选路线比较的胜/平/负及代价差 | 机会覆盖图 + 路线代价差分布 | `U_t` 说明实际出现多个候选；路线差异说明选择具有代价后果；比较 VLM 首选是否比最近目标更有利。具体口径见第 3 节 |
| **5.7 Budget-Related Action Changes** | 按原始预算剩余比例统计探索、目标访问、核验/调整、报告、主动停止；ManyON/AllON 分开 | 分段动作分布 + 样本量；必要时按进度分层 | 展示行为是否随预算阶段变化。进度和候选条件相近时仍有变化，支持与预算压力相关的调整；不声称单凭分布变化证明因果或更优策略 |

### 6. Discussion and Conclusion （todo）

- 用 5.2–5.8 的证据回答三个问题——任务是否制造了自主选择机会（`U_t`）、选择是否有代价后果（路线比较）、停止与预算是否构成真实决策压力（停止分解、预算行为漂移）。判据是各环节都被实证制造出决策空间，而不是参考 agent 分数低。
- SPL 乘法分解把效率损失归因到搜索侧与规划侧；失败现象与诊断指标一一对应，指明改进落点。
- ManyON/AllON 补上多目标导航"并行搜索规划"的评测空白；回收三项贡献（任务与数据、评测与诊断、参考系统与实证发现）。
- and more



## 2. 论文定位与主线

**核心贡献是 benchmark；agent 是基线并提供诊断。**

叙事主线：任务提出自主目标选择、任务进度维护与完成判断需求 → 指标定义如何量化agent的能力 → agent 提供baseline → 实验展示性能、决策机会和具体瓶颈。