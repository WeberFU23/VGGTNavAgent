"""本地测试环境缺少 benchmark 包时提供最小 benchmark_api 存根。

远端 harness 自带真实的 benchmark_api；这里只在 ImportError 时注入
sys.modules 存根，不会遮蔽真实模块，仅让 agents.nav_agent 的 import 链
可以在本地收集。
"""

import enum
import sys
import types

try:
    import benchmark_api  # noqa: F401
except ImportError:
    _stub = types.ModuleType("benchmark_api")

    class Action(enum.IntEnum):
        FINISH = 0
        STOP = 0
        MOVE_FORWARD = 1
        TURN_LEFT = 2
        TURN_RIGHT = 3
        LOOK_UP = 4
        LOOK_DOWN = 5
        TARGET_FOUND = 6
        SUBTASK_STOP = 6
        LEGACY_FINISH = 9

    _stub.Action = Action
    sys.modules["benchmark_api"] = _stub
