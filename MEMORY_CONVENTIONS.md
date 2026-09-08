# 记忆系统命名与组织规范（草案 v2 简化版，待审）

> 目的：v2 记忆改造（原始滑动窗口等）动手前，统一记忆组件的命名与暴露
> 方式。审过后并入 README 架构章节，改名与滑动窗口实现按本文档执行。
>
> 冻结原则：**凡进日志、被评测/分析脚本消费的键一律不改**，改名只发生
> 在 VLM 可见面（工具名、prompt、world_state 键）与代码内部标识符。

## 1. 体系：三层记忆 + 一个视图

```
                 ┌─────────────────────────────┐
                 │  world_state（确定性视图）    │  每次决策由代码对三层记忆
                 │  + BEV 图 + 预计算几何       │  有界渲染，非独立记忆
                 └──────────────┬──────────────┘
        ┌───────────────────────┼───────────────────────┐
  空间记忆（code）         语义记忆（code）          工作记忆（vlm 可写）
  点云 / 占据栅格          caption_store           agent_notes
  关键帧库                 instance_store          action_log
  尺度标定                 （Obs/Instance/Claim）   decision_window（待加）
        │                       │                       │
  不进 JSON：BEV 图、      实例表 top-K 摘要进       notes 全文、最近 3 条
  行内 path_cost/dist；   JSON；caption 只能       动作进 JSON；
  点云/栅格永不进         search_frames 检索       window 作独立节入
                                                   prompt，不进 JSON
```

- **空间记忆**：VGGT-SLAM 在线维护的几何状态。VLM 不直接读，经 BEV 图
  和预计算几何量（path_cost / dist）间接暴露。
- **语义记忆**：`caption_store`（可检索语料库）+ `instance_store`（实
  体集合：身份解析、去重/合并/报告事务）。实例文本经 `update_instance`
  由 VLM 修订，身份与几何由代码管。
- **工作记忆**：唯一 VLM 可写的层。`agent_notes`（VLM 覆写，长期结论）、
  `decision_window`（代码自动追加 VLM 最近 N 轮原始输出，意图连续性）、
  `action_log`（代码记录的客观动作流水，分页可查）。
- **world_state 是视图不是记忆**：每步由代码对三层记忆**有界渲染**、整
  体重建——进多少由摘要策略决定（实例 top-K、动作最近 3 条），点云与
  caption 全文永不进 JSON。无论对话历史如何压缩，任务账本与几何永远
  正确——这是 harness 记忆正确性的根基。

写入者规则一句话：**除 agent_notes 外全部由代码写入；decision_window
的追加由代码执行、内容来自 VLM 原始输出。**

## 2. 命名约定

### 2.1 存储后缀（只有 4 个）

| 后缀 | 判定 | 例子 |
|---|---|---|
| `*_store` | 集合类组件：检索语料库或实体集合（是否做身份解析由组件自身文档说明） | `caption_store`、`instance_store` |
| `*_log` | append-only，只增不改 | `_action_log`、`_rejected_spot_log` |
| `*_window` | 有界滚动窗口，存原始内容 | `_decision_window`（待加） |
| `*_queue` | 待处理条目，有状态机 | `_proposal_queue`、`_revisit_queue` |

实例实体集合命名为 `instance_store`（Observation / Instance /
ReportClaim 三类仍用类名表达内部结构）；论文显示名保留"实例记忆"。
其他内部字段自然命名，不强求后缀。

### 2.2 写入者前缀

VLM 可写的组件用 `agent_` 前缀：`notes` → `agent_notes`（world_state
键与内部字段 `_notes` → `_agent_notes` 同步改）。工具名保持
`update_notes`——从调用方 VLM 视角 "notes" 无歧义，不重复前缀。

### 2.3 工具动词表

| 类别 | 动词 | 现有工具 |
|---|---|---|
| 查看（只读，附图） | `view_*` | `view_frame`、`view_instance` |
| 检索（只读，返回行） | `search_*` | `search_frames`、`search_instances` |
| 查状态（只读，无图） | `get_*` | `get_instance`、`get_agent_status`、`get_action_history` |
| 写记忆 | `update_*` / `merge_*` | `update_instance`、`update_notes`、`merge_instances` |
| 实例化管线 | 阶段动词：`propose→review→instantiate→resolve→commit` | 现有工具已符合 |

只读工具描述中标注 read-only；写工具成功后刷新 world_state（现有行为）。

改名清单（工具）：`set_notes` → `update_notes`；
`som_pick` → `pick_segment`（待定，见 §5）。

### 2.4 world_state 键

不嵌套重构。新键带分组前缀（`instances_*` / `frontier_*` / `proposal_*`）；
现有键只改 `notes` → `agent_notes`（改前 grep 确认无分析脚本依赖）。
`rejected_spots` / `revisit_targets` 等键不动。

## 3. 工具内部状态（不属于记忆体系）

以下组件是执行/感知管线的内部状态，**不是记忆**，不进记忆章节、不进
论文叙事；即使不进论文，命名也一律按 §2.1 后缀执行（改名清单见 §4）：

`_proposal_queue`（审核队列）、`_rejected_spot_log`（被拒像素记录）、
`_revisit_queue`（几何失败重看）、frontier 失败/恢复状态、尺度标定候选。
它们经 world_state 摘要（计数/标记）向 VLM 暴露结论，不暴露过程。

## 4. 执行顺序（审过后）

1. 本规范并入 README 记忆章节；
2. 改名（`set_notes`→`update_notes`、`notes`→`agent_notes`、
   `som_pick`→`pick_segment`）+ prompt 同步——与滑动窗口同批改；
3. 内部字段按后缀表改名（纯代码标识符，不动日志键）：
   `memory` → `instance_store`、`_proposals` → `_proposal_queue`、
   `_rejected_spots` → `_rejected_spot_log`、`_revisit_targets` →
   `_revisit_queue`（world_state 键 `rejected_spots` / `revisit_targets`
   不动）；
4. 实现 `decision_window`；
5. 单测 + smoke episode。

## 5. 已定案项

1. `som_pick` → `pick_segment`（确认执行）。
2. decision_window 条目 = **动态输入 + 全部输出**，最近 5 轮：
   `{step, event, tool_calls: [{name, args}], tool_results: [原文，每条
   ≤1500 字符], reply: <最终 action JSON 原文>, outcome: <执行结果回填>}`。
   明确排除：静态 system prompt（只发一次）、world_state 旧副本（当前份
   已随决策下发）、图片（以占位符 `[image attached: ...]` 标注）、窗口
   节自身（防递归嵌套）。
