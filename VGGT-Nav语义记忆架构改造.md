# 任务：将 vggt_nav_agent 语义层从 CLIP+SAM3 改造为语义记忆架构

你是一名资深工程师，负责改造位于 `/home/wenbofu/vggt_nav_agent/` 的多目标具身导航项目。开工前先通读仓库内的 `AGENT_ARCHITECTURE.md`、`README.md` 以及下文列出的关键源码，理解现状后再动手。

---

## 0. 背景（必读）

**Benchmark**：基于 HM3D val 场景的多目标视觉导航评测。不给定目标顺序和数量。四种 target_mode：single/any/many（数量已知）/all（数量未知，需主动 FINISH）。GOAT-style API：action 6 = TARGET_FOUND（登记发现、不结束 episode），action 0 = FINISH。成功判定：到达目标 navigation viewpoint 的 geodesic 距离 ≤ 0.25m。

**系统现状**（详见 AGENT_ARCHITECTURE.md）：
- client/server 双进程。agent 端（habitat env，python3.9）：状态机、frontier 探索、A*、实例记忆、TSP。server 端（vggtslam env，python3.11）：VGGT-SLAM 建图定位（RGB-only，深度和位姿都是 VGGT 预测，Sim(3) 尺度靠 0.25m 步长在线标定）+ CLIP（常驻）+ SAM3（懒加载）。socket 协议在 `mapping/protocol.py`，端口 5555。
- 当前语义链路：`ground_object(短语)` → CLIP 检索 top-K 关键帧 → SAM3 分割 → mask 投影点云得 3D 点。已确认失败模式：CLIP 分数平坦（0.21–0.25）不可分、假阳性霸榜；SAM3 措辞敏感、单一阈值无校准。

**改造目标（一句话）**：用语义记忆架构替代 CLIP+SAM3——VLM 给每个关键帧写 caption 建立语义记忆，检索从"图文跨模态匹配"变成"文文匹配"；用 VLM pointing 替代 SAM3 mask 投影做目标定位；新增 LLM 决策层接管高层规划与终止决策。导航栈不动。

---

## 1. 硬约束（不许动的部分）

以下模块**禁止修改**，只许调用：
- VGGT-SLAM 建图定位本体（子图、因子图优化、回环、尺度标定）
- 导航跟随、A*、frontier 提取与打分骨架、实例记忆合并逻辑（0.75m 半径）、TSP（`planner.plan_multi`）
- socket 协议骨架（允许新增消息类型，不许破坏既有消息的字段与语义）
- RGB-only 红线：agent 端任何代码不许读仿真器的 GPS/深度/姿态真值
- 双环境隔离：agent 端 python3.9、server 端 python3.11，新增依赖注意版本兼容

**兼容性要求**：所有新功能由 config flag 控制，旧链路（CLIP+SAM3）保留在 flag 后面直到新链路在 smoke test 上跑通，之后再删除。默认配置先保持旧行为，逐阶段切换。

---

## 2. 开工前必做

1. 通读并列出你实际读过的文件：`mapping/protocol.py`、`ground_object`/`ground_frame` 全链路、`Sam3Grounder`、实例记忆与状态机、`planner.plan_multi`、`_should_finish`、VLM stub（`parse_instruction`/`choose_candidate`/`verify_arrival`/`decide_finish`）、config/env 变量的集中定义处。
2. **先输出一份文件级改动计划**（新增哪些文件、修改哪些文件的哪些函数、每个 flag 的名字和默认值），等我确认后再写代码。计划里标注每个改动属于下面哪个 Phase。

---

## 3. 分阶段实施

### Phase 1：语义记忆——caption 生成与检索（server 端）

**目标**：每个关键帧异步生成详细 caption，建立可检索的语义记忆库；检索从 CLIP 图文匹配改为文文匹配。

- **Caption worker**：挂在现有"子图处理顺带算 CLIP 向量"的挂点上，异步执行、最低优先级（GPU 忙时让路）。模型用 Qwen2.5-VL-3B（4bit，经 vLLM 服务调用，见 §5 部署）。Prompt 要求输出：场景类型/房间、可见物体清单（含颜色、材质等属性细粒度描述）、物体间空间关系。Caption 是**查询无关**的，宁可详细。
- **存储**：`caption_store`，每条记录 = {frame_id, 位姿, caption 文本, embedding}。Embedding 用 BGE-M3（不用 CLIP text encoder——77 token 截断 + bag-of-words 特性对长 caption 检索质量差）。落盘持久化，支持按 episode 清空。
- **检索接口**：`retrieve_captions(goal_text, k) -> List[{frame_id, caption, score, pose}]`，余弦相似度 top-K。召回优先、宁可多捞（K 默认取比现有 `NAV_GROUND_TOP_K=5` 更大，如 10）。
- **验收**：单测（mock caption 与 embedding）；集成验证：对一个已有 episode 的关键帧集合，用 "basket"/"gray fabric sofa" 等查询检查 top-K 是否包含人工标注的含目标帧，对比 CLIP 基线的召回率。

### Phase 2：pointing 定位——替代 SAM3（server 端）

**目标**：`ground_object` 链路中，用 VLM pointing 替代 SAM3 mask 投影，输出格式不变（3D 点列表 + 置信度），下游实例记忆零改动。

新 `ground_object(goal_text)` 流程：
1. `retrieve_captions(goal_text)` 粗筛 top-K 候选帧；
2. **查询条件化复核**：每个候选帧带 goal_text 原文重新看原图，逐条核对属性（如 gray、fabric），滤掉假阳性帧；
3. **pointing**：对通过复核的帧，VLM 输出目标像素 point（**point 优先**，bbox 只作交叉验证——point 落在 bbox 外则降置信）。一次调用允许返回多个实例点（many/all 计数用）。模型固定用 **Qwen2.5-VL-7B**（4bit，与 caption 的 3B 同系列，少维护一套权重与推理代码）；
4. **深度采样**：point 像素周围 patch（如 11×11）从 `get_frame_points` 取深度，**先按 VGGT confidence 过滤低分点再取中位数**，得 3D 点；
5. 返回：List[{frame_id, pixel, point_3d, confidence}]。

- **验收**：构造 20–50 张带人工标注目标像素的测试帧，point 命中率（point 落在目标实例 mask 内）≥ 80%；端到端：`ground_object` 在无 SAM3 的情况下返回合理 3D 点，实例记忆正常写入。

### Phase 3：到达例程改造 + 分级置信度（agent 端为主）

**目标**：删除 `NAV_MIN_SAM=0.5` 阈值逻辑，改为 pointing + VQA 复核；低置信观测做探索先验。

- **到达例程**：内部到达判定（0.8m）后 → 360° 扫描 → 每帧 pointing + 逐条属性 VQA 复核 → 确认才 action 6；否决则拉黑该位置（沿用现有黑名单机制）。
- **分级置信度**：单帧观测 → belief 锚点（只做 frontier 打分先验）；≥2 帧独立观测（不同位姿）→ confirmed，进入 TSP。合并逻辑不动，只改置信度来源。
- **末端视觉伺服**：确认目标在视野后，最后一段逼近不看坐标看图像——每步 VLM 判断"目标是否足够近且居中"（目标占比阈值 + 居中程度），满足后停。目的：把 benchmark 的 0.25m 判定与 SLAM 的 0.33–1.0m 误差解耦。伺服过程设步数上限（如 8 步），超限退回坐标判定。
- **小/远目标两段式兜底**：pointing 置信低 / patch 深度方差过大 / 目标像素占比过小（<32×32）时，不登记发现，降级为 belief 锚点当探索先验，逼近后复核。
- **验收**：mock VLM 单测覆盖确认/否决/超时三分支；any 模式 smoke test 不劣于现有成绩（ep0000004 F1=1.0 不许回退）。

### Phase 4：决策层 + 俯视标注地图（agent 端）

**目标**：VLM 决策层接管高层规划，替代/增强纯规则的 `_should_finish` 与无先验的 frontier 打分。

**4a. 俯视标注地图渲染**：占据栅格 → 俯视图 PNG——free 白 / obstacle 黑 / unknown 灰；叠加历史轨迹、当前位姿箭头、confirmed 实例（编号实线圈）、belief 锚点（编号虚线圈）、frontier（编号标记）。编号与 JSON 状态中的 id 严格一一对应。回环导致地图显著变形后重渲染（实例/锚点本来就 ID 锚定在点上，随图动）。渲染放 server 或 client 均可，注意环境依赖最小化。

**4b. 决策输入组装器**：每次决策时组装——
- 世界状态 JSON：任务账本（goal 原文、mode、found 数、expected 数）、实例表（id/类别/属性/区域/置信/dist/A* path_cost **预计算好**——VLM 不算几何）、frontier 表（id/dist/语义线索，线索来自 caption 检索）、近期事件（最近 10 条原文 + 更早压缩统计）、终止账本（未探索面积占比、未复核锚点数）；
- 俯视标注地图 PNG；
- goal_text 检索的 top-K caption 摘要。

**4c. Agentic 决策循环**：事件驱动触发（到达/新实例确认/扫描完成/frontier 耗尽/VERIFY 返回），不是每步调。开放两个只读工具：`query_memory(text)` → caption 检索；`look_at(frame_id)` → 返回该帧图像。最多 3 轮工具调用后必须下决策。

**4d. 受约束输出**：JSON schema 强制（guided decoding 或 prompt + 校验重试）：
```json
{"action": "GOTO_INSTANCE | GOTO_FRONTIER | VERIFY | FINISH",
 "target_id": "<必须存在于当前状态中的 id>",
 "reason": "<简短理由，不参与执行，仅日志>",
 "confidence": 0.0}
```
状态机校验：id 不存在 → 拒绝重试；**FINISH 硬条件**（未探索占比低于阈值 且 无未复核高置信锚点，many 模式另需簇数 ≥ N）不满足 → 自动降级为继续探索；最终非法 → 回退现有确定性规则。全部决策 trace 写 JSONL 日志（时间步、输入摘要、输出、校验结果），供 failure analysis。

**4e. many 模式计数校验**：簇数 < N 时，决策层被提示在已确认实例周边补探索，而不是直接去新 frontier。

- **验收**：mock 决策模型下端到端跑通四种 mode；`NAV_VLM_ENABLED=0`（规则兜底）与 `=1` 均可运行；日志完整可追溯。

### Phase 5：消融开关与旧链路清理

- 三个独立 flag：`NAV_SEMANTIC_BACKEND=semantic_memory|clip_sam`（语义层）、`NAV_DECIDER=vlm|rules`（决策层）、`NAV_ORACLE_GEOMETRY=0|1`（GT 位姿+深度替换 VGGT 的 oracle 实验钩，仅离线 ablation 用，注意这只允许在实验分支，不得进入 RGB-only 主评测路径）。
- 新链路在 single/any/many/all 四模式 smoke test 全部不劣于旧链路后，删除 SAM3 与 CLIP 检索代码（含懒加载逻辑），更新 AGENT_ARCHITECTURE.md。

---

## 4. 关键设计细则（实现时不能违背的"为什么"）

1. **导航端不能只用拍照位姿**：VLM 可能隔 4m 认出目标，导航到拍照位置会判 FP（benchmark 卡 viewpoint 0.25m）。必须保留"图像定位 → 点云采深度 → 3D 点"这一跳。
2. **计数信几何不信文本**：实例数量以几何独立的簇数为准，caption/VLM 报的数字只当提示。
3. **VLM 不记账、不算几何**：dist/path_cost/覆盖率全部由状态机预计算后喂入；VLM 只做基于账本的判断。唯一真源是确定性状态机。
4. **决策层不进控制回路**：底层跟随、避障、重规划永远确定性执行；VLM 只在离散事件点介入。
5. **point 优先于 bbox**：下游只需一个落在物体上的像素采深度（合并半径 0.75m、到达判定 0.8m），远低于检测级 IoU 要求。
6. **caption 召回优先**：粗筛宁可多捞，精度靠查询条件化复核保证——caption 是查询无关的，会漏细粒度属性。

---

## 5. 开发与部署约束

**工作方式：本阶段只改代码，不在本地跑模型。** 开发机上不下载任何模型权重、不执行 GPU 推理，全部模型调用以 mock/接口 stub 形式开发和单测；代码完成后由我到 AutoDL 远端服务器实测。因此：

- **模型权重统一下载到 AutoDL 远端服务器**，本地下载属于浪费带宽，不要尝试。涉及的权重：
  - `Qwen/Qwen2.5-VL-3B-Instruct`（caption）
  - `Qwen/Qwen2.5-VL-7B-Instruct`（pointing）
  - `BAAI/bge-m3`（文文检索 embedding）
- 远端下载方式：在 AutoDL 上优先走 ModelScope 镜像（`pip install modelscope` 后 `modelscope download`，或设置 `HF_ENDPOINT=https://hf-mirror.com` 用 huggingface-cli），避免直连 HuggingFace 超时。权重统一放 `/root/autodl-tmp/models/` 下（AutoDL 系统盘小，数据盘才够用），代码里模型路径全部走环境变量配置（如 `NAV_CAPTION_MODEL_PATH`、`NAV_POINTING_MODEL_PATH`、`NAV_EMBED_MODEL_PATH`），不写死绝对路径。
- 远端推理统一走一个 vLLM 服务，带优先级队列：**决策 > pointing > caption**（caption 异步，GPU 忙时排后）。同帧结果做缓存（caption/pointing/决策附图复用）。
- 显存预算：VGGT 1B ~10GB + caption 3B(4bit) ~3GB + pointing 7B(4bit) ~8GB，按 24GB 单卡规划；决策 LLM 走 API（key 从环境变量读，代码里不许硬编码），API 不可达时自动回退规则决策并打 warning。
- 远端目前无 API 可用：决策层的 VLM 调用必须先以 mock 方式跑通闭环，API 配置留接口。
- 代码里需要为远端实测预留的东西：模型加载失败的明确报错与重试、vLLM 服务地址走环境变量、所有新模块在权重缺失时可被 mock 替换（单测不依赖真实权重）。

---

## 6. 交付物与验收总表

1. 文件级改动计划（开工前，待确认）；
2. 各 Phase 代码 + 单测（mock 外部模型）；
3. 集成验证报告：四模式 smoke test 对比表（旧链路 vs 新链路，sr/f1/spl_multi 至少不劣化）、pointing 命中率统计、一次完整 episode 的决策 trace 样例；
4. 更新后的 AGENT_ARCHITECTURE.md。

**总验收红线**：ep0000004（any basket）F1=1.0 不许回退；任何 Phase 失败时能用 flag 一键切回旧链路。
