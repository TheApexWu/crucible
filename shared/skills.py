"""
CRUCIBLE Skill Distillation Models

Contracts for converting strategy analytics into customer-facing advisory skills.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SkillTrigger(BaseModel):
    """Condition metadata describing when a skill should be considered."""
    name: str
    condition: str
    thresholds: dict[str, float] = Field(default_factory=dict)


class SkillPolicy(BaseModel):
    """Policy controls associated with a skill."""
    allowed_moves: list[str] = Field(default_factory=list)
    forbidden_moves: list[str] = Field(default_factory=list)
    response_style: str = "clear, empathetic, policy-compliant"
    risk_level: Literal["low", "medium", "high"] = "low"


class SkillCard(BaseModel):
    """Human-reviewable skill card plus prompt modules for advisory deployment."""
    id: str
    title: str
    intent: str
    customer_benefit: str
    trigger: SkillTrigger
    policy: SkillPolicy
    when_to_use: list[str] = Field(default_factory=list)
    when_not_to_use: list[str] = Field(default_factory=list)
    prompt_modules: dict[str, str] = Field(default_factory=dict)
    kpis: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class SkillAuditMetadata(BaseModel):
    """Lineage and governance metadata for each distilled bundle."""
    source_run_hash: str
    source_metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    lint_checks: list[str] = Field(default_factory=list)
    reviewer_decision: Literal["pending", "approved", "rejected"] = "pending"


class DistilledSkillBundle(BaseModel):
    """Output bundle from deterministic distillation."""
    run_id: str
    domain: Literal["support_cx"] = "support_cx"
    source_strategy_version: str
    skills: list[SkillCard] = Field(default_factory=list)
    policy_json: dict[str, Any] = Field(default_factory=dict)
    gating_checks: list[str] = Field(default_factory=list)
    approved: bool = False
    audit: SkillAuditMetadata


class SkillEvaluation(BaseModel):
    """Evaluation artifact for a skill bundle."""
    run_id: str
    scenario_count: int
    scores: dict[str, float] = Field(default_factory=dict)
    scenario_examples: list[dict[str, Any]] = Field(default_factory=list)
    summary: list[str] = Field(default_factory=list)
