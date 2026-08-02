# VGGT Nav Agent — 项目介绍

基于 VGGT-SLAM 的 RGB-only 具身导航智能体，运行在 [Habitat HM3D](https://aihabitat.org/) 仿真环境中，用于完成多目标语义导航任务（如"找到红色的椅子"）。

## 核心设计理念

该项目最显著的特点是 **纯 RGB 输入，不读取深度、GPS、指南针或仿真器真值位姿**。导航完全依赖：

1. **VGGT-SLAM**（基于 VGGT-1B 模型的单目 SLAM）在线建图与定位
2. **CLIP + SAM3** 语义记忆与目标定位
3. **确定性控制器**（A* 栅格规划 + 路径跟随）执行低层动作
4. **VLM（视觉语言模型）作为战略监督层**，仅在关键决策点介入

## 项目结构

```
├── agents/                  # 智能体实现
│   ├── mapping_agent.py     # 基础 agent：随机探索 + 喂图建图
│   ├── nav_agent.py         # 导航 agent：多目标状态机 + VLM 战略层
│   └── navigator.py         # 导航执行：重力对齐、占据栅格、A*、路径跟随
├── decision/                # VLM 战略决策层
│   ├── types.py             # 数据契约：TargetSpec、StrategicDecision
│   ├── prompts.py           # 事件驱动 VLM prompt 定义
│   ├── vlm.py               # OpenAI 兼容 VLM 客户端
│   └── README.md            # 设计文档
├── mapping/                 # SLAM 建图模块（client/server 架构）
│   ├── client.py            # 客户端：TCP 通信，喂帧、查询位姿/点云
│   ├── server.py            # 服务端：VGGT-SLAM 在线建图 + 语义层
│   ├── online_solver.py     # 显存优化的 OnlineSolver（帧存 CPU）
│   ├── semantic.py          # CLIP 关键帧记忆 + SAM3 实例定位
│   ├── scale_calibration.py # 在线尺度标定（地图单位 → 米）
│   └── protocol.py          # TCP 二进制协议（跨 py3.9/py3.11 兼容）
├── tests/                   # 单元测试
│   ├── test_navigator.py    # 导航模块测试（栅格、A*、路径跟随、重力对齐）
│   └── test_vlm_decision.py # VLM 决策层测试（prompt 契约、API 兼容、故障回退）
├── scripts/                 # 启动/工具脚本
│   ├── run_mapping_server.sh
│   ├── setup_vggtslam.sh
│   └── dump_pointcloud.py
├── mapping_keyframes/       # 关键帧临时落盘目录
└── VGGT-SLAM/               # VGGT-SLAM 上游仓库（git submodule）
```

## 系统架构

项目采用 **双 conda 环境分离** 的设计：

| 环境 | Python | 角色 |
|---|---|---|
| `habitat` (3.9) | 客户端 | Habitat 仿真 + agent 逻辑 |
| `vggtslam` (3.11) | 服务端 | VGGT-SLAM 建图 + CLIP/SAM3 语义处理 |

两个环境通过 localhost TCP 通信，协议见 `mapping/protocol.py`。

```
┌──────────────────────────────────────────────────────────┐
│  Habitat (py3.9)                                         │
│  ┌────────────┐    TCP     ┌───────────────────────────┐ │
│  │ NavAgent   │◄──────────►│ MappingServer (py3.11)    │ │
│  │            │  喂帧/查询  │                           │ │
│  │ ┌────────┐ │            │ ┌───────────────────────┐ │ │
│  │ │VLM 战略│ │            │ │ VGGT-SLAM (OnlineSolver)│ │
│  │ │决策层  │ │            │ │   - 关键帧筛选          │ │
│  │ └────────┘ │            │ │   - VGGT-1B 前向推理    │ │
│  │            │            │ │   - SALAD 回环检测       │ │
│  │ ┌────────┐ │            │ │   - GTSAM SL(4) 因子图  │ │
│  │ │确定性   │ │            │ └───────────────────────┘ │ │
│  │ │导航执行 │ │            │ ┌───────────────────────┐ │ │
│  │ │- 重力对齐│ │            │ │ 语义层                 │ │
│  │ │- 占据栅格│ │            │ │   - CLIP 关键帧记忆    │ │
│  │ │- A* 规划 │ │            │ │   - SAM3 实例定位      │ │
│  │ │- 路径跟随│ │            │ └───────────────────────┘ │ │
│  │ └────────┘ │            └───────────────────────────┘ │
│  └────────────┘                                          │
└──────────────────────────────────────────────────────────┘
```

## 导航流程（多目标状态机）

### 1. EXPLORE 阶段

- 随机探索环境（默认 70% 前进、30% 转向），每步将 RGB 帧喂给 VGGT-SLAM 服务端建图
- 利用 RGB 变化检测碰撞/卡住（前后帧下采样后的像素差 < 阈值 → 判定为碰撞），碰撞后自动转向恢复
- 每 `NAV_QUERY_INTERVAL`（默认 20）步查询一次目标：
  - CLIP 检索 top-K 关键帧
  - 对 top-K 帧做 SAM3 文本提示分割
  - mask 内点云质心作为 3D 目标点候选

### 2. 候选选择（VLM 事件 2）

- SAM3 候选先按分数排序，过滤黑名单和分数不达标的
- VLM 审阅前 K 个候选的 red-mask 证据图（双栏合成 JPEG：左侧全局上下文 + 右侧红色半透明 mask 近景裁剪），结合当前视野和导航状态
- VLM 决定：`navigate`（选定一个候选，拒绝明确误检的候选）或 `explore`（给探索方向提示）

### 3. NAV 阶段

- A* 在"面包屑"占据栅格上规划路径（栅格沿相机轨迹构建，天然保证可行走且避障）
- `PathFollower` 输出离散动作（前进 0.25m / 左转 30° / 右转 30°）
- 最新关键帧位姿为锚点，航位推算跟踪当前位置
- 定期重建栅格并重规划（地图随探索增长，回环也会改写历史位姿）

### 4. 到达验证（VLM 事件 3）

- 距目标 < `NAV_REACH_M`（默认 0.8m）时触发验证流程：
  - **前置门槛**：SAM3 对当前帧做实时分割，分数 ≥ `NAV_VERIFY_MIN`（默认 0.25）
  - **VLM 复核**：对比当前视野与之前选中的历史 red-mask 证据图
- 三个可能结果：
  - `report_found`：发出 `TARGET_FOUND` 信号
  - `scan`：360° 原地旋转扫描（最多 12 步），每步重新确认
  - `reject`：目标点加入黑名单，退回探索

### 5. 多目标模式

支持四种任务模式：

| 模式 | 行为 | 终止条件 |
|---|---|---|
| `single` / `any` | 找到一个匹配实例即完成 | `TARGET_FOUND` |
| `many` | 报告指定数量的不同实例 | 数量满足 |
| `all` | 报告所有可发现的匹配实例 | VLM 判断搜索空间耗尽 + 步数晚期 |

黑名单机制（`_bad_points`）避免重复报告同一位置的目标。

---

## VLM 集成设计

VLM 是战略监督层，不是逐步运动策略。仅在 4 个事件被调用：

| 事件 | 输入（图像） | 输入（文本） | 输出 | 执行方式 |
|---|---|---|---|---|
| **指令解析** | 无 | 原始指令 + 任务模式 | `grounding_query` + `target_description` | `grounding_query` 给 CLIP/SAM3；`target_description` 给后续 VLM 验证 |
| **候选选择** | 当前 RGB + 历史 red-mask 证据图（≤4 张） | 指令 + target + 导航状态 JSON + 候选元数据 | `navigate(candidate_id)` 或 `explore(hint)` | navigate → A* 规划；explore → 执行 1~3 步方向提示 |
| **到达验证** | 当前 RGB + 选中候选的 red-mask 图 | 指令 + target + 导航状态 JSON | `report_found` / `scan` / `reject` | report_found → TARGET_FOUND；reject → 拉黑退回；scan → 360° 旋转 |
| **终止决策** | 当前 RGB | 指令 + target + 导航状态 JSON | `finish` / `explore` | finish → 不可逆结束 episode（仅 all 模式 + 晚期 + 有前置门槛） |

### 信息给 VLM 的设计原则

以下信息**刻意不发给 VLM**：

- 点云、原始轨迹
- 真值位姿 / 深度
- 语义 ID、目标 3D 坐标

VLM 只需要 RGB 图像 + 自然语言 + 紧凑的状态摘要，与人类视觉导航的信息条件一致。3D 几何、地图、路径规划由确定性模块处理。

### grounding_query vs target_description

| 维度 | `grounding_query` | `target_description` |
|---|---|---|
| **用途** | 给 CLIP/SAM3 做图像检索与分割 | 给 VLM 做候选验证与到达复核 |
| **内容** | 对象名词 + 内在视觉属性（颜色、材质、形状） | 完整匹配条件，保留关系从句等所有约束 |
| **去掉** | 导航动词、量词、关系词（如 "near the table"） | 无 |
| **示例** | `"red fabric chair with wooden legs"` | `"red fabric chair with fabric upholstery and wooden legs, near the sofa"` |

设计原理来自 FOUND-IT 的两级检索：CLIP/SAM3 擅长在图中找短名词短语对应的东西，复杂空间关系和复合条件则由 VLM 查看实际证据图进行语义推理。

---

## 关键技术要点

### 面包屑导航

占据栅格沿相机轨迹构建，而非依赖地板/障碍高度分层。语义层定位的目标必然在机器人曾到达过的地方附近（从历史关键帧检出），因此沿走过的路线规划既不需要高度分层（在稀疏点云+尺度误差下极不可靠），也天然避障。栅格分辨率用关键帧间距自适应。

### 在线尺度标定

VGGT-SLAM 输出的是单目相对尺度（Sim(3) 意义下一致）。利用已知前进步长（MOVE_FORWARD = 0.25m）在线回归"地图单位 → 米"的尺度因子：

- 样本：相邻关键帧的地图位移与其间前进动作数
- 估计：滑动窗口内的中位数，MAD 剔除异常值
- 门控：用当前参考尺度拒绝异常段（撞墙、回环跳变）

### 显存优化

`OnlineSolver` 将子图帧张量保存到 CPU（上游常驻 GPU），回环需要时临时搬回。8GB 级 GPU 即可运行，避免子图累积导致的 OOM。

### VLM 兜底机制

所有 VLM 调用失败（API 不可用、解析失败、超时）自动回退确定性策略：
- 候选选择：取最高 SAM 分数的候选
- 到达验证：仅依赖 SAM3 分数
- 终止决策：仅依赖步数和连续无命中计数

### 动作撤销

利用 RGB 变化（前后帧下采样像素差的均值）检测 MOVE_FORWARD 是否实际生效。碰撞未移动时，对应的动作从尺度标定样本中排除（标记为 -1），航位推算也回滚该步位移。

---

## 运行方式

```bash
# 1. 启动 VGGT-SLAM 服务端（vggtslam 环境）
conda activate vggtslam
python -m mapping.server --port 5555

# 2. 运行导航 agent（habitat 环境）
conda activate habitat
python run_eval.py \
  --config hm3d_config.yaml \
  --agent agents.nav_agent:NavAgent \
  --goal-type description \
  --dataset-dir dataset_semantic \
  --scene-root /path/to/hm3d/val \
  --limit 1 --episode-limit 1 --max-steps 300
```

### 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NAV_VLM_ENABLED` | auto | 启用 VLM 战略层 |
| `NAV_VLM_API_URL` | - | VLM API 地址 |
| `NAV_VLM_MODEL` | - | VLM 模型名 |
| `VGGT_SLAM_HOST` | 127.0.0.1 | SLAM 服务地址 |
| `VGGT_SLAM_PORT` | 5555 | SLAM 服务端口 |
| `NAV_QUERY_INTERVAL` | 20 | 目标查询间隔（步） |
| `NAV_REPLAN_INTERVAL` | 20 | 重规划间隔（步） |
| `NAV_REACH_M` | 0.8 | 到达判定距离（米） |
| `NAV_MIN_SAM` | 0.5 | SAM3 候选最低分数 |
| `NAV_VLM_CANDIDATE_CONF` | 0.35 | VLM 候选选择最低置信度 |
| `NAV_VLM_VERIFY_CONF` | 0.50 | VLM 到达验证最低置信度 |
| `NAV_VLM_FINISH_CONF` | 0.60 | VLM 终止决策最低置信度 |
