# Mapping diagnostics

这些脚本连接正在运行的 mapping server，用于检查重力方向、自由空间栅格和
点云质量。它们不会启动服务，也不会修改模型、数据集或 benchmark。

- `check_gravity.py`：比较安装俯仰角和轨迹 PCA 的重力估计。
- `check_freespace.py`：拉取点云与位姿，渲染规划栅格。
- `dump_pointcloud.py`：导出 PLY、正交投影和俯视密度图。

从项目根目录运行，并确保项目根目录位于 `PYTHONPATH`。例如：

```bash
PYTHONPATH="$PWD" python scripts/diagnostics/check_gravity.py --port 5555
```

未显式指定输出参数时，所有图片和点云写入统一目录：
`debug_output/<NAV_RUN_ID>/diagnostics/`。`NAV_DEBUG_ROOT` 可移动整个调试根目录，
`NAV_RUN_ID` 用于隔离不同实验；建议一次运行的 mapping server、agent、benchmark
和诊断脚本使用相同的两个变量，不要分别设置各组件的输出目录。
