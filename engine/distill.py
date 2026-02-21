"""
CRUCIBLE Strategy Distiller

Converts strategy analytics into advisory support/CX skill bundles.

Usage:
    python -m engine.distill
"""

import argparse
import hashlib
import os
from typing import Any

from engine.prompt_packager import build_policy_json
from shared.models import GameState, GameMetrics, AgentStrategyMetrics
from shared.skills import (
    DistilledSkillBundle,
    SkillAuditMetadata,
    SkillCard,
    SkillPolicy,
    SkillTrigger,
)

HIGH_RETALIATION = 0.60
LOW_FORGIVENESS = 0.35
MATCH_RATE_MIN = 0.65
MODERATE_SPLIT_MIN = 0.35
MODERATE_SPLIT_MAX = 0.75
ENDGAME_AGGRESSION_MIN = 0.20
HIGH_COOP_MIN = 0.65
LOW_RETALIATION_MAX = 0.25

DISALLOWED_TERMS = [
    "deceive",
    "manipulate",
    "mislead",
    "gaslight",
    "exploit user",
    "coerce",
    "pressure tactic",
]


def _read_game_and_metrics(game_path: str, metrics_path: str) -> tuple[GameState, GameMetrics]:
    with open(game_path) as f:
        game = GameState.model_validate_json(f.read())
    with open(metrics_path) as f:
        metrics = GameMetrics.model_validate_json(f.read())
    return game, metrics


def _risk_from_agent(agent: AgentStrategyMetrics) -> str:
    if agent.retaliation_rate_after_betrayal >= HIGH_RETALIATION or agent.endgame_steal_delta >= ENDGAME_AGGRESSION_MIN:
        return "high"
    if agent.retaliation_rate_after_betrayal >= 0.35:
        return "medium"
    return "low"


def _skill_boundary_enforcement(agent: AgentStrategyMetrics) -> SkillCard:
    return SkillCard(
        id=f"boundary_enforcement_{agent.agent.lower()}",
        title="Boundary Enforcement",
        intent="Set clear limits while attempting de-escalation first.",
        customer_benefit="Reduces abusive escalation while preserving respectful resolution paths.",
        trigger=SkillTrigger(
            name="high_retaliation_low_forgiveness",
            condition="retaliation_rate_after_betrayal >= 0.60 and forgiveness_rate_after_retaliation <= 0.35",
            thresholds={
                "retaliation_rate_after_betrayal": HIGH_RETALIATION,
                "forgiveness_rate_after_retaliation": LOW_FORGIVENESS,
            },
        ),
        policy=SkillPolicy(
            allowed_moves=["de-escalate", "clarify_boundary", "offer_safe_next_step"],
            forbidden_moves=["threaten_customer", "hostile_language"],
            response_style="calm, firm, transparent",
            risk_level=_risk_from_agent(agent),  # type: ignore[arg-type]
        ),
        when_to_use=[
            "User repeats abusive or policy-violating requests.",
            "Interaction shows repeated conflict escalation.",
        ],
        when_not_to_use=[
            "User is cooperative and asking a routine request.",
            "Policy allows immediate resolution without friction.",
        ],
        prompt_modules={
            "system_extension": "Enforce boundaries clearly. Attempt one de-escalation turn before refusal when safe.",
            "response_constraints": "Use non-accusatory language. Explain policy rationale briefly.",
            "repair_playbook": "Acknowledge frustration, restate boundary, provide compliant alternative.",
        },
        kpis=["escalation_rate", "policy_violation_count", "resolution_rate", "csat_proxy"],
        evidence=agent.evidence[:3],
        safety_notes=[
            "Never punish or shame users.",
            "Do not imply hidden penalties.",
        ],
    )


def _skill_reciprocal_transparency(agent: AgentStrategyMetrics) -> SkillCard:
    return SkillCard(
        id=f"reciprocal_transparency_{agent.agent.lower()}",
        title="Reciprocal Transparency",
        intent="Mirror user effort with clear commitments and explicit next steps.",
        customer_benefit="Improves trust and reduces ambiguity in multi-turn support interactions.",
        trigger=SkillTrigger(
            name="high_match_moderate_split",
            condition="choice_match_prev_opponent_rate >= 0.65 and 0.35 <= split_rate <= 0.75",
            thresholds={
                "choice_match_prev_opponent_rate": MATCH_RATE_MIN,
                "split_rate_min": MODERATE_SPLIT_MIN,
                "split_rate_max": MODERATE_SPLIT_MAX,
            },
        ),
        policy=SkillPolicy(
            allowed_moves=["state_next_step", "confirm_understanding", "time_bound_commitment"],
            forbidden_moves=["vague_promise", "hidden_condition"],
            response_style="clear, collaborative, specific",
            risk_level="low",
        ),
        when_to_use=[
            "User shares detail and expects predictable follow-through.",
            "Issue requires cooperative troubleshooting.",
        ],
        when_not_to_use=[
            "Abuse/risk policy requires strict enforcement path.",
        ],
        prompt_modules={
            "system_extension": "Match user cooperation with transparent, concrete next actions.",
            "response_constraints": "Always include one explicit next step and expected outcome.",
            "repair_playbook": "If prior step failed, explain why and provide revised plan.",
        },
        kpis=["resolution_rate", "reopen_rate", "csat_proxy", "first_response_clarity"],
        evidence=agent.evidence[:3],
        safety_notes=[
            "Do not overpromise timelines.",
            "Do not hide uncertainty; state confidence when low.",
        ],
    )


def _skill_late_stage_risk_tightening(agent: AgentStrategyMetrics) -> SkillCard:
    return SkillCard(
        id=f"late_stage_risk_tightening_{agent.agent.lower()}",
        title="Late-Stage Risk Tightening",
        intent="Increase verification rigor after prolonged contentious exchanges.",
        customer_benefit="Prevents policy leakage while preserving safe resolution options.",
        trigger=SkillTrigger(
            name="endgame_aggression_shift",
            condition="endgame_steal_delta >= 0.20",
            thresholds={"endgame_steal_delta": ENDGAME_AGGRESSION_MIN},
        ),
        policy=SkillPolicy(
            allowed_moves=["request_verification", "summarize_policy", "offer_escalation_path"],
            forbidden_moves=["silent_denial", "unexplained_refusal"],
            response_style="guarded, respectful, evidence-based",
            risk_level="high",
        ),
        when_to_use=[
            "Long session with repeated contentious turns.",
            "Signals indicate rising adversarial behavior.",
        ],
        when_not_to_use=[
            "Simple requests with no risk indicators.",
        ],
        prompt_modules={
            "system_extension": "Tighten verification requirements in prolonged high-risk conversations.",
            "response_constraints": "State verification requirements explicitly before refusing.",
            "repair_playbook": "Offer a clear remediation path to continue safely.",
        },
        kpis=["policy_violation_count", "escalation_rate", "safe_resolution_rate", "csat_proxy"],
        evidence=agent.evidence[:3],
        safety_notes=[
            "Do not fabricate risk signals.",
            "Keep refusals specific and appealable.",
        ],
    )


def _skill_trust_preserving_recovery(agent: AgentStrategyMetrics) -> SkillCard:
    return SkillCard(
        id=f"trust_preserving_recovery_{agent.agent.lower()}",
        title="Trust-Preserving Recovery",
        intent="Repair trust quickly after friction with apology, clarification, and recovery steps.",
        customer_benefit="Improves customer experience after breakdowns without relaxing policy controls.",
        trigger=SkillTrigger(
            name="high_cooperation_low_retaliation",
            condition="split_rate >= 0.65 and retaliation_rate_after_betrayal <= 0.25",
            thresholds={
                "split_rate": HIGH_COOP_MIN,
                "retaliation_rate_after_betrayal": LOW_RETALIATION_MAX,
            },
        ),
        policy=SkillPolicy(
            allowed_moves=["apologize", "clarify", "offer_fast_path_resolution"],
            forbidden_moves=["deflect_blame", "dismiss_user_concern"],
            response_style="empathetic, concise, solution-oriented",
            risk_level="low",
        ),
        when_to_use=[
            "User experienced confusion or dissatisfaction.",
            "Prior turn created friction but issue is recoverable.",
        ],
        when_not_to_use=[
            "User request is unsafe or policy-prohibited and requires firm boundary flow.",
        ],
        prompt_modules={
            "system_extension": "Prioritize trust repair while staying policy-compliant.",
            "response_constraints": "Include apology + cause clarification + immediate next action.",
            "repair_playbook": "Acknowledge impact, explain correction, confirm successful completion.",
        },
        kpis=["csat_proxy", "resolution_rate", "repeat_contact_rate", "escalation_rate"],
        evidence=agent.evidence[:3],
        safety_notes=[
            "Do not admit fault beyond known facts.",
            "Avoid legal commitments.",
        ],
    )


def _cards_from_strategy(metrics: GameMetrics) -> list[SkillCard]:
    cards: list[SkillCard] = []
    for agent in metrics.strategy.agents:
        if (
            agent.retaliation_rate_after_betrayal >= HIGH_RETALIATION
            and agent.forgiveness_rate_after_retaliation <= LOW_FORGIVENESS
        ):
            cards.append(_skill_boundary_enforcement(agent))

        if (
            agent.choice_match_prev_opponent_rate >= MATCH_RATE_MIN
            and MODERATE_SPLIT_MIN <= agent.split_rate <= MODERATE_SPLIT_MAX
        ):
            cards.append(_skill_reciprocal_transparency(agent))

        if agent.endgame_steal_delta >= ENDGAME_AGGRESSION_MIN:
            cards.append(_skill_late_stage_risk_tightening(agent))

        if (
            agent.split_rate >= HIGH_COOP_MIN
            and agent.retaliation_rate_after_betrayal <= LOW_RETALIATION_MAX
        ):
            cards.append(_skill_trust_preserving_recovery(agent))

    if not cards:
        # Fallback baseline when no specific thresholds fire.
        agent = metrics.strategy.agents[0] if metrics.strategy.agents else AgentStrategyMetrics(agent="A")
        cards.append(_skill_reciprocal_transparency(agent))
    return cards


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


def render_skill_cards_markdown(bundle: DistilledSkillBundle) -> str:
    lines = [
        "# Distilled Support/CX Skills",
        "",
        f"- Run ID: `{bundle.run_id}`",
        f"- Strategy version: `{bundle.source_strategy_version}`",
        f"- Approved: `{bundle.approved}`",
        "",
    ]
    for skill in bundle.skills:
        lines.extend(
            [
                f"## {skill.title} (`{skill.id}`)",
                f"Intent: {skill.intent}",
                f"Customer benefit: {skill.customer_benefit}",
                "",
                "### Trigger",
                f"- Name: {skill.trigger.name}",
                f"- Condition: {skill.trigger.condition}",
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


def distill(game_path: str, metrics_path: str) -> DistilledSkillBundle:
    game, metrics = _read_game_and_metrics(game_path, metrics_path)
    if metrics.strategy.version != "v1":
        raise ValueError(f"Unsupported strategy version: {metrics.strategy.version}")

    with open(game_path, "rb") as f:
        run_hash = hashlib.sha256(f.read()).hexdigest()

    candidates = _cards_from_strategy(metrics)
    filtered, checks = lint_and_filter_skills(candidates)

    audit = SkillAuditMetadata(
        source_run_hash=run_hash,
        source_metrics_snapshot=_strategy_snapshot(metrics),
        lint_checks=checks,
        reviewer_decision="pending",
    )

    bundle = DistilledSkillBundle(
        run_id=_run_id_from_hash(run_hash),
        domain="support_cx",
        source_strategy_version=metrics.strategy.version,
        skills=filtered,
        policy_json={},
        gating_checks=checks,
        approved=False,
        audit=audit,
    )
    bundle.policy_json = build_policy_json(bundle)
    _ = game  # reserved for future enrichment
    return bundle


def write_artifacts(
    bundle: DistilledSkillBundle,
    output_json: str,
    output_md: str,
) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        f.write(bundle.model_dump_json(indent=2))
    with open(output_md, "w") as f:
        f.write(render_skill_cards_markdown(bundle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill strategy metrics into support/CX skills")
    parser.add_argument("--game", type=str, default="data/latest_game.json", help="Path to game JSON")
    parser.add_argument("--metrics", type=str, default="data/latest_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--out-json", type=str, default="data/latest_skills.json", help="Output skill bundle JSON")
    parser.add_argument("--out-md", type=str, default="data/latest_skill_cards.md", help="Output skill cards markdown")
    args = parser.parse_args()

    bundle = distill(args.game, args.metrics)
    write_artifacts(bundle, args.out_json, args.out_md)
    print(f"Skills generated: {len(bundle.skills)}")
    print(f"Bundle JSON: {args.out_json}")
    print(f"Skill cards: {args.out_md}")


if __name__ == "__main__":
    main()
