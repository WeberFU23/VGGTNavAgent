"""Dependency-free checks for centralized runtime output paths."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_paths import PROJECT_ROOT, debug_root, env_debug_path, run_debug_path


def _with_env(values, fn):
    previous = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        fn()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_defaults_share_one_project_debug_root():
    def check():
        expected = PROJECT_ROOT / "debug_output" / "current"
        assert Path(debug_root()) == PROJECT_ROOT / "debug_output"
        assert Path(run_debug_path("agent")) == expected / "agent"
        assert Path(run_debug_path("mapping", "diagnostics")) == \
            expected / "mapping" / "diagnostics"

    _with_env({"NAV_DEBUG_ROOT": None, "NAV_RUN_ID": None}, check)


def test_relative_legacy_override_stays_inside_run_root():
    def check():
        resolved = Path(env_debug_path("MAPPING_DEBUG_DIR", "unused"))
        expected = PROJECT_ROOT / "debug_output" / "episode-7" / "agent-old"
        assert resolved == expected

    _with_env({
        "NAV_DEBUG_ROOT": None,
        "NAV_RUN_ID": "episode/7",
        "MAPPING_DEBUG_DIR": "agent-old",
    }, check)


if __name__ == "__main__":
    test_defaults_share_one_project_debug_root()
    test_relative_legacy_override_stays_inside_run_root()
    print("runtime path tests passed")
