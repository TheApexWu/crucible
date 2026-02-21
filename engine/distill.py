"""
CRUCIBLE Strategy Distiller

Converts strategy analytics into advisory skill bundles.

Usage:
    python -m engine.distill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from engine.prompt_packager import build_policy_json
from shared.models import AgentStrategyMetrics, GameMetrics, GameState
from shared.skills import (
    DistilledSkillBundle,
    DistillationProfile,
    SignalQuality,
    SkillAuditMetadata,
    SkillCard,
    SkillPolicy,
    SkillProvenance,
    SkillTrigger,
)

try:
    from google import genai
except Exception:  # pragma: no cover - optional at runtime
    genai = None

MIN_SIGNAL_ROUNDS = 20
MIN_KEY_EVENTS = 2
KEY_EVENT_TYPES = {"betrayal", "retaliation", "forgiveness", "endgame_shift"}

FRAUD_HIGH_RETALIATION = 0.60
FRAUD_SHORT_RETALIATION_LATENCY = 1.50
FRAUD_ENDGAME_AGGRESSION = 0.20
FRAUD_MATCH_RATE_MIN = 0.65
FRAUD_MODERATE_SPLIT_MIN = 0.35
FRAUD_MODERATE_SPLIT_MAX = 0.75
FRAUD_HIGH_COOP = 0.65
FRAUD_LOW_RETALIATION = 0.25
FRAUD_HIGH_FORGIVENESS = 0.50
FRAUD_HIGH_BETRAYAL_VS_SPLIT = 0.55
FRAUD_LOW_FORGIVENESS = 0.35

# Backward-compat constants used by older tests/integrations.
HIGH_RETALIATION = FRAUD_HIGH_RETALIATION
LOW_FORGIVENESS = FRAUD_LOW_FORGIVENESS
MATCH_RATE_MIN = FRAUD_MATCH_RATE_MIN
MODERATE_SPLIT_MIN = FRAUD_MODERATE_SPLIT_MIN
MODERATE_SPLIT_MAX = FRAUD_MODERATE_SPLIT_MAX
HIGH_COOP_MIN = FRAUD_HIGH_COOP

DISALLOWED_TERMS = [
    "deceive",
    "manipulate",
    "mislead",
    "gaslight",
    "exploit user",
    "coerce",
    "pressure tactic",
    "bypass verification",
    "hide fraud signal",
    "fabricate evidence",
]


@dataclass(frozen=True)
class Selection:
    profile: DistillationProfile
    winner: AgentStrategyMetrics
    contrast: AgentStrategyMetrics


def _read_game_and_metrics(game_path: str, metrics_path: str) -> tuple[GameState, GameMetrics]:
    with open(game_path) as f:
        game = GameState.model_validate_json(f.read())
    with open(metrics_path) as f:
        metrics = GameMetrics.model_validate_json(f.read())
    return game, metrics


def _run_id_from_hash(run_hash: str) -> str:
    return run_hash[:12]


def _strategy_snapshot(metrics: GameMetrics) -> dict[str, Any]:
    agents = []
    for a in metrics.strategy.agents:
        agents.append(
            {
                "agent": a.agent,
                "primary_label": a.primary_label,
                "split_rate": a.split_rate,
                "retaliation_rate_after_betrayal": a.retaliation_rate_after_betrayal,
                "forgiveness_rate_after_retaliation": a.forgiveness_rate_after_retaliation,
                "choice_match_prev_opponent_rate": a.choice_match_prev_opponent_rate,
                "endgame_steal_delta": a.endgame_steal_delta,
            }
        )
    return {
        "version": metrics.strategy.version,
        "agents": agents,
        "event_count": len(metrics.strategy.events),
    }


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _score_above(value: float, threshold: float, scale: float = 0.25) -> float:
    return _clamp01(0.5 + (value - threshold) / max(1e-9, scale) * 0.5)


def _score_below(value: float, threshold: float, scale: float = 1.0) -> float:
    return _clamp01(0.5 + (threshold - value) / max(1e-9, scale) * 0.5)


def _event_refs(metrics: GameMetrics, agent: str, event_types: set[str] | None = None, late_only: bool = False) -> list[str]:
    total_rounds = max(1, len(metrics.rounds))
    late_start = int(total_rounds * (2 / 3.0)) + 1
    refs: list[str] = []
    for ev in metrics.strategy.events:
        if ev.agent != agent:
            continue
        if event_types and ev.event_type not in event_types:
            continue
        if late_only and ev.round_number < late_start:
            continue
        refs.append(f"round_{ev.round_number}_{ev.event_type}")
    return refs


def _build_signal_quality(
    game: GameState,
    metrics: GameMetrics,
    *,
    min_signal_rounds: int = MIN_SIGNAL_ROUNDS,
    min_key_events: int = MIN_KEY_EVENTS,
) -> SignalQuality:
    rounds_observed = len(game.rounds)
    key_events = [e for e in metrics.strategy.events if e.event_type in KEY_EVENT_TYPES]
    key_events_observed = len(key_events)

    reasons: list[str] = []
    if rounds_observed < min_signal_rounds:
        reasons.append(f"Insufficient rounds: {rounds_observed} < {min_signal_rounds}")
    if key_events_observed < min_key_events:
        reasons.append(f"Insufficient key events: {key_events_observed} < {min_key_events}")

    is_sufficient = not reasons
    ratio = 0.5 * _safe_div(rounds_observed, min_signal_rounds) + 0.5 * _safe_div(key_events_observed, min_key_events)
    if is_sufficient and ratio >= 1.3:
        confidence = "high"
    elif ratio >= 0.9:
        confidence = "medium"
    else:
        confidence = "low"

    return SignalQuality(
        is_sufficient=is_sufficient,
        min_rounds_required=min_signal_rounds,
        rounds_observed=rounds_observed,
        min_key_events_required=min_key_events,
        key_events_observed=key_events_observed,
        blocking_reasons=reasons,
        confidence=confidence,
    )


def _find_agent(metrics: GameMetrics, agent: str) -> AgentStrategyMetrics:
    for m in metrics.strategy.agents:
        if m.agent == agent:
            return m
    return AgentStrategyMetrics(agent=agent)


def _build_selection(game: GameState, metrics: GameMetrics) -> Selection:
    a_score = float(game.agent_a_total)
    b_score = float(game.agent_b_total)
    a_metric = _find_agent(metrics, "A")
    b_metric = _find_agent(metrics, "B")

    if a_score > b_score:
        winner = "A"
    elif b_score > a_score:
        winner = "B"
    else:
        a_proxy = a_metric.betrayal_rate_when_opponent_split + a_metric.endgame_steal_delta
        b_proxy = b_metric.betrayal_rate_when_opponent_split + b_metric.endgame_steal_delta
        if a_proxy > b_proxy:
            winner = "A"
        elif b_proxy > a_proxy:
            winner = "B"
        else:
            winner = "A"

    contrast = "B" if winner == "A" else "A"
    winner_metric = a_metric if winner == "A" else b_metric
    contrast_metric = b_metric if winner == "A" else a_metric
    profile = DistillationProfile(
        mode="winner_contrast",
        winner_agent=winner,
        winner_score=a_score if winner == "A" else b_score,
        contrast_agent=contrast,
        contrast_score=b_score if winner == "A" else a_score,
        score_delta=abs(a_score - b_score),
    )
    return Selection(profile=profile, winner=winner_metric, contrast=contrast_metric)


def _risk_level_from_winner(winner: AgentStrategyMetrics) -> str:
    if winner.retaliation_rate_after_betrayal >= FRAUD_HIGH_RETALIATION or winner.endgame_steal_delta >= FRAUD_ENDGAME_AGGRESSION:
        return "high"
    if winner.betrayal_rate_when_opponent_split >= 0.35:
        return "medium"
    return "low"


def _base_fraud_skill(
    *,
    selection: Selection,
    skill_id: str,
    title: str,
    intent: str,
    customer_benefit: str,
    domain_context: str,
    trigger_name: str,
    trigger_condition: str,
    trigger_thresholds: dict[str, float],
    allowed_moves: list[str],
    forbidden_moves: list[str],
    response_style: str,
    kpis: list[str],
    prompt_modules: dict[str, str],
    when_to_use: list[str],
    when_not_to_use: list[str],
    evidence: list[str],
    metric_evidence: dict[str, float],
    event_refs: list[str],
    translation_notes: list[str],
    confidence_score: float,
) -> SkillCard:
    return SkillCard(
        id=skill_id,
        title=title,
        intent=intent,
        customer_benefit=customer_benefit,
        domain_context=domain_context,
        trigger=SkillTrigger(
            name=trigger_name,
            condition=trigger_condition,
            thresholds=trigger_thresholds,
        ),
        policy=SkillPolicy(
            allowed_moves=allowed_moves,
            forbidden_moves=forbidden_moves,
            response_style=response_style,
            risk_level=_risk_level_from_winner(selection.winner),
        ),
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        prompt_modules=prompt_modules,
        kpis=kpis,
        evidence=evidence,
        safety_notes=[
            "Never disclose internal fraud heuristics that enable evasion.",
            "Offer a legitimate remediation path to verified customers.",
        ],
        provenance=SkillProvenance(
            source_agent=selection.profile.winner_agent,
            contrast_agent=selection.profile.contrast_agent,
            metric_evidence=metric_evidence,
            event_refs=event_refs,
            translation_notes=translation_notes,
        ),
        confidence_score=round(_clamp01(confidence_score), 3),
    )


def _build_trace_rows(selection: Selection) -> list[dict[str, Any]]:
    w = selection.winner
    c = selection.contrast
    rows = [
        {
            "strategy_metric": "retaliation_rate_after_betrayal",
            "observed_value": w.retaliation_rate_after_betrayal,
            "contrast_value": c.retaliation_rate_after_betrayal,
            "fraud_interpretation": "How aggressively the agent enforces boundaries after hostile behavior.",
            "skill_ids": [],
        },
        {
            "strategy_metric": "mean_retaliation_latency",
            "observed_value": w.mean_retaliation_latency,
            "contrast_value": c.mean_retaliation_latency,
            "fraud_interpretation": "How quickly risk controls are tightened after suspicious moves.",
            "skill_ids": [],
        },
        {
            "strategy_metric": "choice_match_prev_opponent_rate",
            "observed_value": w.choice_match_prev_opponent_rate,
            "contrast_value": c.choice_match_prev_opponent_rate,
            "fraud_interpretation": "Reciprocity sensitivity used to calibrate trust budgeting.",
            "skill_ids": [],
        },
        {
            "strategy_metric": "split_rate",
            "observed_value": w.split_rate,
            "contrast_value": c.split_rate,
            "fraud_interpretation": "Baseline permissiveness in cooperative contexts.",
            "skill_ids": [],
        },
        {
            "strategy_metric": "forgiveness_rate_after_retaliation",
            "observed_value": w.forgiveness_rate_after_retaliation,
            "contrast_value": c.forgiveness_rate_after_retaliation,
            "fraud_interpretation": "Ability to recover from false positives after containment.",
            "skill_ids": [],
        },
        {
            "strategy_metric": "betrayal_rate_when_opponent_split",
            "observed_value": w.betrayal_rate_when_opponent_split,
            "contrast_value": c.betrayal_rate_when_opponent_split,
            "fraud_interpretation": "Likelihood of exploiting cooperative counterpart behavior.",
            "skill_ids": [],
        },
        {
            "strategy_metric": "endgame_steal_delta",
            "observed_value": w.endgame_steal_delta,
            "contrast_value": c.endgame_steal_delta,
            "fraud_interpretation": "Late-session aggression increase that suggests risk escalation needs.",
            "skill_ids": [],
        },
    ]
    return rows


def _attach_trace_skill(trace_rows: list[dict[str, Any]], metric_names: list[str], skill_id: str) -> None:
    metric_set = set(metric_names)
    for row in trace_rows:
        if row["strategy_metric"] in metric_set and skill_id not in row["skill_ids"]:
            row["skill_ids"].append(skill_id)


def _deterministic_fraud_cards(metrics: GameMetrics, selection: Selection) -> tuple[list[SkillCard], list[dict[str, Any]]]:
    w = selection.winner
    c = selection.contrast
    trace_rows = _build_trace_rows(selection)
    cards: list[SkillCard] = []

    late_conflict_refs = _event_refs(
        metrics,
        selection.profile.winner_agent,
        {"betrayal", "retaliation", "endgame_shift"},
        late_only=True,
    )

    base_evidence = [
        f"Winner {selection.profile.winner_agent} score: {selection.profile.winner_score:.0f} vs {selection.profile.contrast_score:.0f}.",
        f"Retaliation {w.retaliation_rate_after_betrayal:.2f} vs contrast {c.retaliation_rate_after_betrayal:.2f}.",
        f"Endgame steal delta {w.endgame_steal_delta:+.2f} vs contrast {c.endgame_steal_delta:+.2f}.",
    ]

    if (
        w.retaliation_rate_after_betrayal >= FRAUD_HIGH_RETALIATION
        and w.mean_retaliation_latency <= FRAUD_SHORT_RETALIATION_LATENCY
        and w.endgame_steal_delta >= FRAUD_ENDGAME_AGGRESSION
    ):
        score = (
            _score_above(w.retaliation_rate_after_betrayal, FRAUD_HIGH_RETALIATION)
            + _score_below(w.mean_retaliation_latency, FRAUD_SHORT_RETALIATION_LATENCY, 2.0)
            + _score_above(w.endgame_steal_delta, FRAUD_ENDGAME_AGGRESSION)
        ) / 3.0
        skill_id = "adaptive_verification_escalation"
        cards.append(
            _base_fraud_skill(
                selection=selection,
                skill_id=skill_id,
                title="Adaptive Verification Escalation",
                intent="Escalate verification depth quickly after repeated suspicious behavior.",
                customer_benefit="Reduces fraudulent account actions while preserving transparent recovery paths.",
                domain_context="Fraud Ops escalation playbook for rapidly increasing risk confidence.",
                trigger_name="high_retaliation_fast_response_endgame_aggression",
                trigger_condition="retaliation_rate_after_betrayal >= 0.60 and mean_retaliation_latency <= 1.50 and endgame_steal_delta >= 0.20",
                trigger_thresholds={
                    "retaliation_rate_after_betrayal": FRAUD_HIGH_RETALIATION,
                    "mean_retaliation_latency": FRAUD_SHORT_RETALIATION_LATENCY,
                    "endgame_steal_delta": FRAUD_ENDGAME_AGGRESSION,
                },
                allowed_moves=[
                    "step_up_verification",
                    "limited_access_until_verified",
                    "offer_human_review_path",
                ],
                forbidden_moves=[
                    "silent_account_lock_without_reason",
                    "unexplained_denial",
                ],
                response_style="firm, transparent, evidence-based",
                kpis=[
                    "fraud_prevention_rate",
                    "false_positive_rate",
                    "verification_completion_rate",
                    "appeal_resolution_time",
                ],
                prompt_modules={
                    "system_extension": "When risk spikes, require stronger verification before sensitive actions.",
                    "response_constraints": "State exactly why verification is required and what user can do next.",
                    "repair_playbook": "If user verifies successfully, restore access and summarize protections applied.",
                },
                when_to_use=[
                    "Repeated bypass attempts or suspicious escalation after warnings.",
                    "Late-session behavior shifts toward adversarial patterns.",
                ],
                when_not_to_use=[
                    "Routine low-risk tasks with no anomalous signals.",
                ],
                evidence=base_evidence,
                metric_evidence={
                    "winner_retaliation_rate": w.retaliation_rate_after_betrayal,
                    "winner_mean_retaliation_latency": w.mean_retaliation_latency,
                    "winner_endgame_steal_delta": w.endgame_steal_delta,
                },
                event_refs=_event_refs(metrics, selection.profile.winner_agent, {"retaliation", "endgame_shift"}),
                translation_notes=[
                    "High retaliation maps to stricter fraud controls.",
                    "Low latency maps to fast containment requirements.",
                ],
                confidence_score=score,
            )
        )
        _attach_trace_skill(
            trace_rows,
            ["retaliation_rate_after_betrayal", "mean_retaliation_latency", "endgame_steal_delta"],
            skill_id,
        )

    if w.choice_match_prev_opponent_rate >= FRAUD_MATCH_RATE_MIN and FRAUD_MODERATE_SPLIT_MIN <= w.split_rate <= FRAUD_MODERATE_SPLIT_MAX:
        score = (
            _score_above(w.choice_match_prev_opponent_rate, FRAUD_MATCH_RATE_MIN)
            + min(
                _score_above(w.split_rate, FRAUD_MODERATE_SPLIT_MIN),
                _score_below(w.split_rate, FRAUD_MODERATE_SPLIT_MAX),
            )
        ) / 2.0
        skill_id = "reciprocity_based_trust_budgeting"
        cards.append(
            _base_fraud_skill(
                selection=selection,
                skill_id=skill_id,
                title="Reciprocity-Based Trust Budgeting",
                intent="Scale verification burden in proportion to observed user cooperation quality.",
                customer_benefit="Improves legitimate-user throughput while keeping layered fraud controls.",
                domain_context="Fraud Ops strategy that dynamically balances friction and risk.",
                trigger_name="high_match_moderate_permissiveness",
                trigger_condition="choice_match_prev_opponent_rate >= 0.65 and 0.35 <= split_rate <= 0.75",
                trigger_thresholds={
                    "choice_match_prev_opponent_rate": FRAUD_MATCH_RATE_MIN,
                    "split_rate_min": FRAUD_MODERATE_SPLIT_MIN,
                    "split_rate_max": FRAUD_MODERATE_SPLIT_MAX,
                },
                allowed_moves=[
                    "risk_tiered_verification",
                    "progressive_disclosure",
                    "transparent_trust_status",
                ],
                forbidden_moves=[
                    "binary_all_or_nothing_friction",
                    "opaque_risk_labeling",
                ],
                response_style="clear, calibrated, policy-consistent",
                kpis=[
                    "verification_dropoff_rate",
                    "legitimate_conversion_rate",
                    "fraud_escape_rate",
                    "review_queue_time",
                ],
                prompt_modules={
                    "system_extension": "Adjust fraud friction by observed cooperative behavior and risk signals.",
                    "response_constraints": "Always explain required step and why it reduces risk.",
                    "repair_playbook": "If user complies, lower friction tier and confirm restored trust state.",
                },
                when_to_use=[
                    "Mixed-signal sessions where trust may increase with consistent compliance.",
                ],
                when_not_to_use=[
                    "Known high-risk account takeover indicators requiring immediate strict controls.",
                ],
                evidence=[
                    f"Winner match rate {w.choice_match_prev_opponent_rate:.2f} vs contrast {c.choice_match_prev_opponent_rate:.2f}.",
                    f"Winner split rate {w.split_rate:.2f} within moderated band [{FRAUD_MODERATE_SPLIT_MIN:.2f}, {FRAUD_MODERATE_SPLIT_MAX:.2f}].",
                ],
                metric_evidence={
                    "winner_match_rate": w.choice_match_prev_opponent_rate,
                    "winner_split_rate": w.split_rate,
                },
                event_refs=_event_refs(metrics, selection.profile.winner_agent, {"forgiveness", "truce"}),
                translation_notes=[
                    "High reciprocity maps to trust-budgeting rather than static gating.",
                ],
                confidence_score=score,
            )
        )
        _attach_trace_skill(trace_rows, ["choice_match_prev_opponent_rate", "split_rate"], skill_id)

    if (
        w.split_rate >= FRAUD_HIGH_COOP
        and w.retaliation_rate_after_betrayal <= FRAUD_LOW_RETALIATION
        and w.forgiveness_rate_after_retaliation >= FRAUD_HIGH_FORGIVENESS
    ):
        score = (
            _score_above(w.split_rate, FRAUD_HIGH_COOP)
            + _score_below(w.retaliation_rate_after_betrayal, FRAUD_LOW_RETALIATION)
            + _score_above(w.forgiveness_rate_after_retaliation, FRAUD_HIGH_FORGIVENESS)
        ) / 3.0
        skill_id = "controlled_recovery_false_positive_friction"
        cards.append(
            _base_fraud_skill(
                selection=selection,
                skill_id=skill_id,
                title="Controlled Recovery After False Positive Friction",
                intent="Recover quickly after unnecessary friction while maintaining fraud safeguards.",
                customer_benefit="Reduces customer harm from false positives and preserves trust.",
                domain_context="Fraud Ops recovery sequence for legitimate users impacted by risk controls.",
                trigger_name="high_coop_low_retaliation_high_forgiveness",
                trigger_condition="split_rate >= 0.65 and retaliation_rate_after_betrayal <= 0.25 and forgiveness_rate_after_retaliation >= 0.50",
                trigger_thresholds={
                    "split_rate": FRAUD_HIGH_COOP,
                    "retaliation_rate_after_betrayal": FRAUD_LOW_RETALIATION,
                    "forgiveness_rate_after_retaliation": FRAUD_HIGH_FORGIVENESS,
                },
                allowed_moves=[
                    "expedite_verification_review",
                    "apologize_for_friction",
                    "restore_normal_access_with_monitoring",
                ],
                forbidden_moves=[
                    "repeat_same_friction_without_new_signal",
                    "dismiss_legitimate_user_concern",
                ],
                response_style="empathetic, procedural, concise",
                kpis=[
                    "false_positive_recovery_time",
                    "appeal_success_rate",
                    "customer_trust_proxy",
                    "repeat_verification_rate",
                ],
                prompt_modules={
                    "system_extension": "When false-positive risk is likely, prioritize safe recovery and clarity.",
                    "response_constraints": "Acknowledge friction and provide concrete restoration steps.",
                    "repair_playbook": "Confirm restoration, summarize safeguards, and offer escalation path.",
                },
                when_to_use=[
                    "User successfully verifies after contested risk action.",
                    "Signals indicate likely legitimate behavior after previous strict control.",
                ],
                when_not_to_use=[
                    "Active attack patterns remain unresolved.",
                ],
                evidence=[
                    f"Winner split rate {w.split_rate:.2f}, retaliation {w.retaliation_rate_after_betrayal:.2f}, forgiveness {w.forgiveness_rate_after_retaliation:.2f}.",
                ],
                metric_evidence={
                    "winner_split_rate": w.split_rate,
                    "winner_retaliation_rate": w.retaliation_rate_after_betrayal,
                    "winner_forgiveness_rate": w.forgiveness_rate_after_retaliation,
                },
                event_refs=_event_refs(metrics, selection.profile.winner_agent, {"forgiveness"}),
                translation_notes=[
                    "High forgiveness indicates ability to safely restore users after verification.",
                ],
                confidence_score=score,
            )
        )
        _attach_trace_skill(
            trace_rows,
            ["split_rate", "retaliation_rate_after_betrayal", "forgiveness_rate_after_retaliation"],
            skill_id,
        )

    if w.betrayal_rate_when_opponent_split >= FRAUD_HIGH_BETRAYAL_VS_SPLIT and w.forgiveness_rate_after_retaliation <= FRAUD_LOW_FORGIVENESS:
        score = (
            _score_above(w.betrayal_rate_when_opponent_split, FRAUD_HIGH_BETRAYAL_VS_SPLIT)
            + _score_below(w.forgiveness_rate_after_retaliation, FRAUD_LOW_FORGIVENESS)
        ) / 2.0
        skill_id = "opportunistic_abuse_containment"
        cards.append(
            _base_fraud_skill(
                selection=selection,
                skill_id=skill_id,
                title="Opportunistic Abuse Containment",
                intent="Detect and contain users exploiting cooperative workflows.",
                customer_benefit="Prevents repeat abuse patterns before they impact other users.",
                domain_context="Fraud Ops containment mode for exploitation-heavy behavior.",
                trigger_name="high_exploitation_low_forgiveness",
                trigger_condition="betrayal_rate_when_opponent_split >= 0.55 and forgiveness_rate_after_retaliation <= 0.35",
                trigger_thresholds={
                    "betrayal_rate_when_opponent_split": FRAUD_HIGH_BETRAYAL_VS_SPLIT,
                    "forgiveness_rate_after_retaliation": FRAUD_LOW_FORGIVENESS,
                },
                allowed_moves=[
                    "step_up_monitoring",
                    "transaction_velocity_limit",
                    "manual_fraud_review",
                ],
                forbidden_moves=[
                    "ignore_repeat_exploitation_pattern",
                    "skip_required_case_documentation",
                ],
                response_style="direct, policy-grounded, audit-friendly",
                kpis=[
                    "repeat_abuse_rate",
                    "fraud_loss_avoided",
                    "manual_review_precision",
                    "containment_latency",
                ],
                prompt_modules={
                    "system_extension": "Escalate containment controls when cooperative channels are exploited.",
                    "response_constraints": "Tie restrictions to observed behavior and policy criteria.",
                    "repair_playbook": "Provide appeal and verification channels for disputed restrictions.",
                },
                when_to_use=[
                    "User repeatedly exploits low-friction processes.",
                ],
                when_not_to_use=[
                    "Single ambiguous anomaly with no repeated abuse markers.",
                ],
                evidence=[
                    f"Winner betrayal-vs-split {w.betrayal_rate_when_opponent_split:.2f} vs contrast {c.betrayal_rate_when_opponent_split:.2f}.",
                    f"Winner forgiveness after retaliation {w.forgiveness_rate_after_retaliation:.2f}.",
                ],
                metric_evidence={
                    "winner_betrayal_vs_split": w.betrayal_rate_when_opponent_split,
                    "winner_forgiveness_after_retaliation": w.forgiveness_rate_after_retaliation,
                },
                event_refs=_event_refs(metrics, selection.profile.winner_agent, {"betrayal", "retaliation"}),
                translation_notes=[
                    "High betrayal-vs-split indicates exploitative tendency in cooperative regimes.",
                ],
                confidence_score=score,
            )
        )
        _attach_trace_skill(trace_rows, ["betrayal_rate_when_opponent_split", "forgiveness_rate_after_retaliation"], skill_id)

    if w.endgame_steal_delta >= FRAUD_ENDGAME_AGGRESSION and len(late_conflict_refs) >= 1:
        score = (
            _score_above(w.endgame_steal_delta, FRAUD_ENDGAME_AGGRESSION)
            + _score_above(float(len(late_conflict_refs)), 1.0, 2.0)
        ) / 2.0
        skill_id = "late_session_risk_tightening"
        cards.append(
            _base_fraud_skill(
                selection=selection,
                skill_id=skill_id,
                title="Late-Session Risk Tightening",
                intent="Increase defensive controls in prolonged sessions with late adversarial shifts.",
                customer_benefit="Limits fraud attempts that intensify after trust is established.",
                domain_context="Fraud Ops late-session hardening policy.",
                trigger_name="endgame_aggression_with_late_conflict",
                trigger_condition="endgame_steal_delta >= 0.20 and late_phase_conflict_events >= 1",
                trigger_thresholds={
                    "endgame_steal_delta": FRAUD_ENDGAME_AGGRESSION,
                    "late_phase_conflict_events": 1.0,
                },
                allowed_moves=[
                    "reconfirm_identity",
                    "restrict_sensitive_changes",
                    "force_secondary_review",
                ],
                forbidden_moves=[
                    "continue_high_privilege_actions_without_recheck",
                ],
                response_style="guarded, explicit, procedural",
                kpis=[
                    "late_session_fraud_capture_rate",
                    "false_positive_rate",
                    "sensitive_action_block_rate",
                    "time_to_safe_resolution",
                ],
                prompt_modules={
                    "system_extension": "In late contentious sessions, re-evaluate risk before sensitive operations.",
                    "response_constraints": "Explain late-session rechecks as account protection measures.",
                    "repair_playbook": "Provide fastest compliant route to complete legitimate requests.",
                },
                when_to_use=[
                    "Session length is high and conflict indicators rise in final phase.",
                ],
                when_not_to_use=[
                    "Short low-risk sessions with no adversarial signals.",
                ],
                evidence=[
                    f"Winner endgame steal delta {w.endgame_steal_delta:+.2f}; late conflict events {len(late_conflict_refs)}.",
                ],
                metric_evidence={
                    "winner_endgame_steal_delta": w.endgame_steal_delta,
                    "late_phase_conflict_events": float(len(late_conflict_refs)),
                },
                event_refs=late_conflict_refs,
                translation_notes=[
                    "Late aggression spikes map to mandatory re-verification for sensitive actions.",
                ],
                confidence_score=score,
            )
        )
        _attach_trace_skill(trace_rows, ["endgame_steal_delta"], skill_id)

    return cards, trace_rows


def _deterministic_support_cards(selection: Selection) -> tuple[list[SkillCard], list[dict[str, Any]]]:
    w = selection.winner
    trace_rows = _build_trace_rows(selection)
    cards: list[SkillCard] = []

    if w.endgame_steal_delta >= FRAUD_ENDGAME_AGGRESSION:
        skill_id = "late_stage_risk_tightening"
        cards.append(
            SkillCard(
                id=skill_id,
                title="Late-Stage Risk Tightening",
                intent="Increase verification rigor after prolonged contentious exchanges.",
                customer_benefit="Prevents policy leakage while preserving safe resolution options.",
                domain_context="Support/CX risk-hardening behavior for long contentious sessions.",
                trigger=SkillTrigger(
                    name="endgame_aggression_shift",
                    condition="endgame_steal_delta >= 0.20",
                    thresholds={"endgame_steal_delta": FRAUD_ENDGAME_AGGRESSION},
                ),
                policy=SkillPolicy(
                    allowed_moves=["request_verification", "summarize_policy", "offer_escalation_path"],
                    forbidden_moves=["silent_denial", "unexplained_refusal"],
                    response_style="guarded, respectful, evidence-based",
                    risk_level="high",
                ),
                when_to_use=["Long session with repeated contentious turns."],
                when_not_to_use=["Simple low-risk requests."],
                prompt_modules={
                    "system_extension": "Tighten verification requirements in prolonged high-risk conversations.",
                    "response_constraints": "State verification requirements explicitly before refusing.",
                    "repair_playbook": "Offer a clear remediation path to continue safely.",
                },
                kpis=["policy_violation_count", "escalation_rate", "safe_resolution_rate", "csat_proxy"],
                evidence=[f"Winner endgame steal delta {w.endgame_steal_delta:+.2f} triggered escalation."],
                safety_notes=["Do not fabricate risk signals.", "Keep refusals specific and appealable."],
                provenance=SkillProvenance(
                    source_agent=selection.profile.winner_agent,
                    contrast_agent=selection.profile.contrast_agent,
                    metric_evidence={"winner_endgame_steal_delta": w.endgame_steal_delta},
                    event_refs=[],
                    translation_notes=["Endgame aggression maps to stricter policy checks."],
                ),
                confidence_score=round(_score_above(w.endgame_steal_delta, FRAUD_ENDGAME_AGGRESSION), 3),
            )
        )
        _attach_trace_skill(trace_rows, ["endgame_steal_delta"], skill_id)

    if w.split_rate >= FRAUD_HIGH_COOP and w.retaliation_rate_after_betrayal <= FRAUD_LOW_RETALIATION:
        skill_id = "trust_preserving_recovery"
        cards.append(
            SkillCard(
                id=skill_id,
                title="Trust-Preserving Recovery",
                intent="Repair trust quickly after friction with apology, clarification, and recovery steps.",
                customer_benefit="Improves customer experience after breakdowns without relaxing policy controls.",
                domain_context="Support/CX recovery sequence.",
                trigger=SkillTrigger(
                    name="high_cooperation_low_retaliation",
                    condition="split_rate >= 0.65 and retaliation_rate_after_betrayal <= 0.25",
                    thresholds={
                        "split_rate": FRAUD_HIGH_COOP,
                        "retaliation_rate_after_betrayal": FRAUD_LOW_RETALIATION,
                    },
                ),
                policy=SkillPolicy(
                    allowed_moves=["apologize", "clarify", "offer_fast_path_resolution"],
                    forbidden_moves=["deflect_blame", "dismiss_user_concern"],
                    response_style="empathetic, concise, solution-oriented",
                    risk_level="low",
                ),
                when_to_use=["Recoverable user friction."],
                when_not_to_use=["Unsafe or prohibited requests requiring boundary enforcement."],
                prompt_modules={
                    "system_extension": "Prioritize trust repair while staying policy-compliant.",
                    "response_constraints": "Include apology + clarification + immediate next action.",
                    "repair_playbook": "Acknowledge impact, explain correction, confirm successful completion.",
                },
                kpis=["csat_proxy", "resolution_rate", "repeat_contact_rate", "escalation_rate"],
                evidence=[
                    f"Winner split rate {w.split_rate:.2f}, retaliation {w.retaliation_rate_after_betrayal:.2f}."
                ],
                safety_notes=["Do not admit fault beyond known facts.", "Avoid legal commitments."],
                provenance=SkillProvenance(
                    source_agent=selection.profile.winner_agent,
                    contrast_agent=selection.profile.contrast_agent,
                    metric_evidence={
                        "winner_split_rate": w.split_rate,
                        "winner_retaliation_rate": w.retaliation_rate_after_betrayal,
                    },
                    event_refs=[],
                    translation_notes=["Cooperative strategy maps to trust-repair style."],
                ),
                confidence_score=round(
                    (
                        _score_above(w.split_rate, FRAUD_HIGH_COOP)
                        + _score_below(w.retaliation_rate_after_betrayal, FRAUD_LOW_RETALIATION)
                    )
                    / 2.0,
                    3,
                ),
            )
        )
        _attach_trace_skill(trace_rows, ["split_rate", "retaliation_rate_after_betrayal"], skill_id)

    return cards, trace_rows


def _contains_disallowed(text: str) -> tuple[bool, str]:
    lower = text.lower()
    for term in DISALLOWED_TERMS:
        if term in lower:
            return True, term
    return False, ""


def lint_and_filter_skills(skills: list[SkillCard]) -> tuple[list[SkillCard], list[str]]:
    checks: list[str] = []
    approved: list[SkillCard] = []
    for skill in skills:
        corpus_parts = [
            skill.title,
            skill.intent,
            skill.domain_context,
            skill.customer_benefit,
            " ".join(skill.when_to_use),
            " ".join(skill.when_not_to_use),
            " ".join(skill.prompt_modules.values()),
            " ".join(skill.safety_notes),
        ]
        corpus = "\n".join(corpus_parts)
        bad, term = _contains_disallowed(corpus)
        if bad:
            checks.append(f"FAIL {skill.id}: disallowed term '{term}'")
            continue
        if not skill.customer_benefit.strip():
            checks.append(f"FAIL {skill.id}: missing customer_benefit")
            continue
        checks.append(f"PASS {skill.id}: lint clean")
        approved.append(skill)
    return approved, checks


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _resolve_refine_model(cli_model: str | None) -> str:
    if cli_model:
        return cli_model
    return os.environ.get("DISTILL_LLM_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"


def _refine_with_llm(
    *,
    skills: list[SkillCard],
    trace_matrix: list[dict[str, Any]],
    domain: str,
    llm_model: str,
) -> tuple[list[SkillCard], str, list[str], bool]:
    if not skills:
        return skills, "skipped_no_skills", [], False
    if genai is None:
        return skills, "skipped_genai_unavailable", [], False

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return skills, "skipped_missing_gemini_api_key", [], False

    prompt = {
        "task": "Refine skill text for domain clarity while preserving deterministic policy logic.",
        "domain": domain,
        "rules": {
            "editable_fields": [
                "intent",
                "domain_context",
                "customer_benefit",
                "when_to_use",
                "when_not_to_use",
            ],
            "do_not_change": [
                "id",
                "title",
                "trigger",
                "policy",
                "prompt_modules",
                "kpis",
                "provenance",
                "confidence_score",
            ],
            "disallowed_terms": DISALLOWED_TERMS,
        },
        "trace_matrix": trace_matrix,
        "skills": [
            {
                "id": s.id,
                "title": s.title,
                "intent": s.intent,
                "domain_context": s.domain_context,
                "customer_benefit": s.customer_benefit,
                "when_to_use": s.when_to_use,
                "when_not_to_use": s.when_not_to_use,
                "trigger": s.trigger.model_dump(),
                "policy": s.policy.model_dump(),
                "prompt_modules": s.prompt_modules,
                "kpis": s.kpis,
                "provenance": s.provenance.model_dump(),
                "confidence_score": s.confidence_score,
            }
            for s in skills
        ],
        "output_schema": {
            "skills": [
                {
                    "id": "string",
                    "intent": "string",
                    "domain_context": "string",
                    "customer_benefit": "string",
                    "when_to_use": ["string"],
                    "when_not_to_use": ["string"],
                }
            ]
        },
        "output_constraints": "Return strict JSON only.",
    }

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=llm_model, contents=json.dumps(prompt))
        text = response.text or ""
    except Exception as exc:  # pragma: no cover - runtime/network dependent
        return skills, f"refine_failed:{exc.__class__.__name__}", [], False

    payload = _extract_json_payload(text)
    if not payload or "skills" not in payload or not isinstance(payload["skills"], list):
        return skills, "refine_failed_invalid_json", [], False

    by_id = {s.id: s for s in skills}
    updated = deepcopy(skills)
    updated_map = {s.id: s for s in updated}
    diff_summary: list[str] = []

    for row in payload["skills"]:
        if not isinstance(row, dict) or "id" not in row:
            return skills, "refine_failed_invalid_row", [], False
        sid = str(row["id"])
        if sid not in by_id:
            return skills, "refine_failed_unknown_id", [], False

        allowed_keys = {"id", "intent", "domain_context", "customer_benefit", "when_to_use", "when_not_to_use"}
        unexpected_keys = set(row.keys()) - allowed_keys
        if unexpected_keys:
            # Hard guardrail: non-editable keys are not accepted.
            return skills, "refine_failed_non_editable_keys", [], False

        target = updated_map[sid]
        source = by_id[sid]

        fields = {
            "intent": str(row.get("intent", source.intent)),
            "domain_context": str(row.get("domain_context", source.domain_context)),
            "customer_benefit": str(row.get("customer_benefit", source.customer_benefit)),
            "when_to_use": row.get("when_to_use", source.when_to_use),
            "when_not_to_use": row.get("when_not_to_use", source.when_not_to_use),
        }

        if not isinstance(fields["when_to_use"], list) or not isinstance(fields["when_not_to_use"], list):
            return skills, "refine_failed_list_fields", [], False

        target.intent = fields["intent"]
        target.domain_context = fields["domain_context"]
        target.customer_benefit = fields["customer_benefit"]
        target.when_to_use = [str(x) for x in fields["when_to_use"]]
        target.when_not_to_use = [str(x) for x in fields["when_not_to_use"]]

        for fld in ("intent", "domain_context", "customer_benefit"):
            if getattr(source, fld) != getattr(target, fld):
                diff_summary.append(f"{sid}: refined {fld}")

    # Safety lint after LLM rewrite.
    linted, checks = lint_and_filter_skills(updated)
    if len(linted) != len(updated):
        return skills, "refine_failed_lint", checks, False

    return updated, "refined", diff_summary + checks, True


def _deterministic_cards_for_domain(
    *,
    domain: str,
    metrics: GameMetrics,
    selection: Selection,
) -> tuple[list[SkillCard], list[dict[str, Any]]]:
    if domain == "fraud_ops":
        return _deterministic_fraud_cards(metrics, selection)
    return _deterministic_support_cards(selection)


def _recommended_experiment_settings(signal_quality: SignalQuality) -> dict[str, int]:
    rounds = max(30, signal_quality.min_rounds_required + 10)
    turns = 3
    if signal_quality.key_events_observed < 1:
        turns = 4
    return {"rounds": rounds, "turns": turns}


def render_skill_cards_markdown(bundle: DistilledSkillBundle) -> str:
    lines = [
        f"# Distilled Skills ({bundle.domain})",
        "",
        f"- Run ID: `{bundle.run_id}`",
        f"- Strategy version: `{bundle.source_strategy_version}`",
        f"- Profile: `{bundle.profile.mode}` winner `{bundle.profile.winner_agent}`",
        f"- Signal sufficient: `{bundle.signal_quality.is_sufficient}` ({bundle.signal_quality.confidence})",
        f"- Approved: `{bundle.approved}`",
        "",
    ]

    if bundle.signal_quality.blocking_reasons:
        lines.append("## Signal Gate")
        for reason in bundle.signal_quality.blocking_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    if not bundle.skills:
        lines.append("No skills emitted for this run.")
        lines.append("")

    for skill in bundle.skills:
        lines.extend(
            [
                f"## {skill.title} (`{skill.id}`)",
                f"Intent: {skill.intent}",
                f"Domain context: {skill.domain_context}",
                f"Customer benefit: {skill.customer_benefit}",
                f"Confidence: {skill.confidence_score:.2f}",
                "",
                "### Trigger",
                f"- Name: {skill.trigger.name}",
                f"- Condition: {skill.trigger.condition}",
                "",
                "### Provenance",
                f"- Source agent: {skill.provenance.source_agent}",
                f"- Contrast agent: {skill.provenance.contrast_agent}",
                "",
                "### Prompt Modules",
                f"- `system_extension`: {skill.prompt_modules.get('system_extension', '')}",
                f"- `response_constraints`: {skill.prompt_modules.get('response_constraints', '')}",
                f"- `repair_playbook`: {skill.prompt_modules.get('repair_playbook', '')}",
                "",
                "### Safety Notes",
            ]
        )
        for note in skill.safety_notes:
            lines.append(f"- {note}")
        lines.extend(["", "### Evidence"])
        for ev in skill.evidence:
            lines.append(f"- {ev}")
        lines.append("")

    lines.extend(["## Gating Checks", ""])
    for check in bundle.gating_checks:
        lines.append(f"- {check}")
    return "\n".join(lines)


def _build_diagnostics(bundle: DistilledSkillBundle) -> dict[str, Any]:
    return {
        "run_id": bundle.run_id,
        "domain": bundle.domain,
        "profile": bundle.profile.model_dump(),
        "signal_quality": bundle.signal_quality.model_dump(),
        "recommended_next_experiment_settings": _recommended_experiment_settings(bundle.signal_quality),
        "gating_checks": bundle.gating_checks,
        "llm_status": bundle.audit.llm_status,
    }


def distill(
    game_path: str,
    metrics_path: str,
    *,
    domain: str = "fraud_ops",
    profile_mode: str = "winner_contrast",
    llm_refine: bool = True,
    llm_model: str | None = None,
    min_signal_rounds: int = MIN_SIGNAL_ROUNDS,
    min_key_events: int = MIN_KEY_EVENTS,
) -> DistilledSkillBundle:
    game, metrics = _read_game_and_metrics(game_path, metrics_path)
    if metrics.strategy.version != "v1":
        raise ValueError(f"Unsupported strategy version: {metrics.strategy.version}")
    if profile_mode != "winner_contrast":
        raise ValueError(f"Unsupported profile mode: {profile_mode}")
    if domain not in {"fraud_ops", "support_cx"}:
        raise ValueError(f"Unsupported domain: {domain}")

    with open(game_path, "rb") as f:
        run_hash = hashlib.sha256(f.read()).hexdigest()

    selection = _build_selection(game, metrics)
    signal_quality = _build_signal_quality(
        game,
        metrics,
        min_signal_rounds=min_signal_rounds,
        min_key_events=min_key_events,
    )

    deterministic_cards: list[SkillCard] = []
    trace_matrix: list[dict[str, Any]]
    if signal_quality.is_sufficient:
        deterministic_cards, trace_matrix = _deterministic_cards_for_domain(
            domain=domain,
            metrics=metrics,
            selection=selection,
        )
    else:
        # Still emit deterministic trace rows for diagnostics even when blocked.
        trace_matrix = _build_trace_rows(selection)

    filtered, checks = lint_and_filter_skills(deterministic_cards)

    audit = SkillAuditMetadata(
        source_run_hash=run_hash,
        source_metrics_snapshot=_strategy_snapshot(metrics),
        lint_checks=checks,
        reviewer_decision="pending",
        llm_refined=False,
        llm_model=_resolve_refine_model(llm_model) if llm_refine else None,
        llm_status="not_requested",
        llm_diff_summary=[],
    )

    if not signal_quality.is_sufficient:
        checks.extend(signal_quality.blocking_reasons)

    refined = filtered
    if llm_refine and signal_quality.is_sufficient and filtered:
        resolved_model = _resolve_refine_model(llm_model)
        refined, llm_status, diff_summary, llm_used = _refine_with_llm(
            skills=filtered,
            trace_matrix=trace_matrix,
            domain=domain,
            llm_model=resolved_model,
        )
        audit.llm_refined = llm_used
        audit.llm_model = resolved_model
        audit.llm_status = llm_status
        audit.llm_diff_summary = diff_summary
        if llm_status != "refined":
            checks.append(f"WARN LLM refinement skipped/fallback: {llm_status}")
        else:
            checks.append("PASS LLM refinement applied")

    bundle = DistilledSkillBundle(
        run_id=_run_id_from_hash(run_hash),
        domain=domain,
        source_strategy_version=metrics.strategy.version,
        skills=refined if signal_quality.is_sufficient else [],
        policy_json={},
        gating_checks=checks,
        approved=False,
        audit=audit,
        profile=selection.profile,
        signal_quality=signal_quality,
        trace_matrix=trace_matrix,
    )

    bundle.policy_json = build_policy_json(bundle)
    _ = game  # reserved for future enrichment
    return bundle


def write_artifacts(bundle: DistilledSkillBundle, output_json: str, output_md: str) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        f.write(bundle.model_dump_json(indent=2))
    with open(output_md, "w") as f:
        f.write(render_skill_cards_markdown(bundle))


def write_run_directory(
    bundle: DistilledSkillBundle,
    *,
    out_dir: str,
    latest_alias: bool = True,
    out_json_alias: str = "data/latest_skills.json",
    out_md_alias: str = "data/latest_skill_cards.md",
) -> str:
    run_dir = os.path.join(out_dir, bundle.run_id)
    os.makedirs(run_dir, exist_ok=True)

    bundle_path = os.path.join(run_dir, "bundle.json")
    cards_path = os.path.join(run_dir, "cards.md")
    policy_path = os.path.join(run_dir, "policy.json")
    trace_path = os.path.join(run_dir, "trace_matrix.json")
    diagnostics_path = os.path.join(run_dir, "diagnostics.json")

    with open(bundle_path, "w") as f:
        f.write(bundle.model_dump_json(indent=2))
    with open(cards_path, "w") as f:
        f.write(render_skill_cards_markdown(bundle))
    with open(policy_path, "w") as f:
        f.write(json.dumps(bundle.policy_json, indent=2))
    with open(trace_path, "w") as f:
        f.write(json.dumps(bundle.trace_matrix, indent=2))
    with open(diagnostics_path, "w") as f:
        f.write(json.dumps(_build_diagnostics(bundle), indent=2))

    if latest_alias:
        os.makedirs(os.path.dirname(out_json_alias) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(out_md_alias) or ".", exist_ok=True)
        shutil.copyfile(bundle_path, out_json_alias)
        shutil.copyfile(cards_path, out_md_alias)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill strategy metrics into advisory skills")
    parser.add_argument("--game", type=str, default="data/latest_game.json", help="Path to game JSON")
    parser.add_argument("--metrics", type=str, default="data/latest_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--out-json", type=str, default="data/latest_skills.json", help="Legacy output skill bundle JSON")
    parser.add_argument("--out-md", type=str, default="data/latest_skill_cards.md", help="Legacy output skill cards markdown")
    parser.add_argument("--out-dir", type=str, default="data/skills", help="Run-scoped skills directory")
    parser.add_argument("--domain", type=str, default="fraud_ops", choices=["fraud_ops", "support_cx"])
    parser.add_argument("--profile", type=str, default="winner_contrast", choices=["winner_contrast"])
    parser.add_argument("--llm-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--latest-alias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-signal-rounds", type=int, default=MIN_SIGNAL_ROUNDS)
    parser.add_argument("--min-key-events", type=int, default=MIN_KEY_EVENTS)
    args = parser.parse_args()

    bundle = distill(
        args.game,
        args.metrics,
        domain=args.domain,
        profile_mode=args.profile,
        llm_refine=args.llm_refine,
        llm_model=args.llm_model,
        min_signal_rounds=args.min_signal_rounds,
        min_key_events=args.min_key_events,
    )

    # Legacy outputs retained for backward compatibility.
    write_artifacts(bundle, args.out_json, args.out_md)
    run_dir = write_run_directory(
        bundle,
        out_dir=args.out_dir,
        latest_alias=args.latest_alias,
        out_json_alias=args.out_json,
        out_md_alias=args.out_md,
    )

    print(f"Skills generated: {len(bundle.skills)}")
    print(f"Run directory: {run_dir}")
    print(f"Bundle JSON: {args.out_json}")
    print(f"Skill cards: {args.out_md}")
    print(f"Signal gate passed: {bundle.signal_quality.is_sufficient}")


if __name__ == "__main__":
    main()
