from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class ReleaseOpsAction(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ReleaseOpsObservation(BaseModel):
    observation: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    reward: float = 0.0
    done: bool = False
    terminal_reason: Optional[str] = None

class SystemState(BaseModel):
    phase: str
    phase_index: int
    hours_to_deadline: int
    review_budget_remaining: int
    evidence_actions_remaining: int
    release_service: str
    release_stage: str
    rules: List[str]
    proposals: List[Dict[str, Any]]
    worker_stats: List[Dict[str, Any]]

    # Internal state tracking
    active_proposals: List[str]
    resolved_proposals: Dict[str, Dict[str, Any]]
    known_violations: List[str]
    is_terminal: bool = False
    terminal_reason: Optional[str] = None
