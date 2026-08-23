"""Canonical locations for generated diagnostics and evaluation artifacts.

Source code, datasets and model weights must never be mixed with runtime
outputs.  All built-in defaults therefore live below one project-local root::

    debug_output/<run-id>/

Set ``NAV_DEBUG_ROOT`` to move that root and ``NAV_RUN_ID`` to isolate a run.
Legacy per-component path variables remain supported.  Relative overrides are
resolved inside the selected run directory; absolute overrides are preserved
for compatibility with existing server deployments.
"""

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _safe_component(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned[:120] or "current"


def debug_root():
    """Return the absolute root shared by every built-in debug producer."""
    configured = str(os.environ.get("NAV_DEBUG_ROOT", "")).strip()
    root = Path(configured).expanduser() if configured else \
        PROJECT_ROOT / "debug_output"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return str(root.resolve())


def run_debug_root():
    """Return the root for the current logical run."""
    run_id = _safe_component(os.environ.get("NAV_RUN_ID", "current"))
    return str((Path(debug_root()) / run_id).resolve())


def run_debug_path(*parts):
    """Build an absolute path below the current run directory."""
    return str(Path(run_debug_root()).joinpath(*map(str, parts)).resolve())


def env_debug_path(env_name, default_path):
    """Resolve a legacy output override without scattering relative paths.

    Absolute paths are intentionally kept for remote deployment scripts that
    place each experiment under a dedicated server run directory.  A relative
    override is interpreted relative to ``debug_output/<run-id>``.
    """
    configured = str(os.environ.get(env_name, "")).strip()
    if not configured:
        return str(Path(default_path).expanduser().resolve())
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(run_debug_root()) / path
    return str(path.resolve())
