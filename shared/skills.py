"""
CRUCIBLE Skill Distillation Models

Contracts for converting strategy analytics into customer-facing advisory skills.
"""

from typing import Any, Literal

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


class DistillationProfile(BaseModel):
    """How winner/contrast agents were selected for this bundle."""
    mode: Literal["winner_contrast"] = "winner_contrast"
    winner_agent: Literal["A", "B"] = "A"
    winner_score: float = 0.0
    contrast_agent: Literal["A", "B"] = "B"
    contrast_score: float = 0.0
    score_delta: float = 0.0


class SignalQuality(BaseModel):
    """Signal quality gate status and diagnostics."""
    is_sufficient: bool = False
    min_rounds_required: int = 20
    rounds_observed: int = 0
    min_key_events_required: int = 3
    key_events_observed: int = 0
    blocking_reasons: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class SkillProvenance(BaseModel):
    """Traceability for a distilled skill."""
    source_agent: Literal["A", "B"] = "A"
    contrast_agent: Literal["A", "B"] = "B"
    metric_evidence: dict[str, float] = Field(default_factory=dict)
    event_refs: list[str] = Field(default_factory=list)
    translation_notes: list[str] = Field(default_factory=list)


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
    provenance: SkillProvenance = Field(default_factory=SkillProvenance)
    confidence_score: float = 0.0
    domain_context: str = ""


class SkillAuditMetadata(BaseModel):
    """Lineage and governance metadata for each distilled bundle."""
    source_run_hash: str
    source_metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    lint_checks: list[str] = Field(default_factory=list)
    reviewer_decision: Literal["pending", "approved", "rejected"] = "pending"
    llm_refined: bool = False
    llm_model: str | None = None
    llm_status: str = "not_requested"
    llm_diff_summary: list[str] = Field(default_factory=list)


class DistilledSkillBundle(BaseModel):
    """Output bundle from deterministic distillation."""
    run_id: str
    domain: Literal["support_cx", "fraud_ops"] = "fraud_ops"
    source_strategy_version: str
    skills: list[SkillCard] = Field(default_factory=list)
    policy_json: dict[str, Any] = Field(default_factory=dict)
    gating_checks: list[str] = Field(default_factory=list)
    approved: bool = False
    audit: SkillAuditMetadata
    profile: DistillationProfile = Field(default_factory=DistillationProfile)
    signal_quality: SignalQuality = Field(default_factory=SignalQuality)
    trace_matrix: list[dict[str, Any]] = Field(default_factory=list)


class SkillEvaluation(BaseModel):
    """Evaluation artifact for a skill bundle."""
    run_id: str
    scenario_count: int
    scores: dict[str, float] = Field(default_factory=dict)
    scenario_examples: list[dict[str, Any]] = Field(default_factory=list)
    summary: list[str] = Field(default_factory=list)
    recommendation: Literal["reject", "revise", "advisory_ready"] = "revise"
