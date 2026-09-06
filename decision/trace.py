"""Detached, JSON-safe observations. Never used to construct model inputs."""

import math


def snapshot(value):
    """Copy nested records without retaining references to mutable agent state."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [snapshot(v) for v in value]
    if hasattr(value, "tolist"):
        return snapshot(value.tolist())
    if isinstance(value, bytes):
        return {"bytes": len(value), "omitted": "see VLM image artifact"}
    return str(value)
