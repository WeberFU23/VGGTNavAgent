"""增量式 VGGT-SLAM 建图模块（服务端 / 客户端分离）。

- ``mapping.server``：在 vggtslam (python 3.11) conda 环境中运行，
  包装 VGGT-SLAM 的在线建图流程（关键帧筛选、子图 VGGT 重建、
  SALAD 回环、GTSAM SL(4) 因子图优化）。
- ``mapping.client``：在 habitat (python 3.9) 环境中运行，
  供 agent 每步喂 RGB 帧、查询当前位姿与全局点云。

两个环境通过 localhost TCP 通信，协议见 ``mapping.protocol``。
"""
