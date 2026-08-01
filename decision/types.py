"""Dependency-free data contracts shared by NavAgent and the VLM client."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TargetSpec:
    grounding_query: str
    target_description: str
    confidence: float = 0.0


@dataclass(frozen=True)
class StrategicDecision:
    decision: str
    candidate_id: Optional[str] = None
    rejected_candidate_ids: List[str] = field(default_factory=list)
    exploration_hint: str = "none"
    confidence: float = 0.0
    reason: str = ""
