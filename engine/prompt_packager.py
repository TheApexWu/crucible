"""
Prompt packaging helpers for advisory skill bundles.
"""

from shared.skills import DistilledSkillBundle


def build_policy_json(bundle: DistilledSkillBundle) -> dict:
    """
    Build a machine-readable policy package from distilled skills.
    Advisory mode only: no runtime auto-switching behavior is encoded.
    """
    skills = []
    for card in bundle.skills:
        skills.append(
            {
                "id": card.id,
                "title": card.title,
                "risk_level": card.policy.risk_level,
                "trigger": {
                    "name": card.trigger.name,
                    "condition": card.trigger.condition,
                    "thresholds": card.trigger.thresholds,
                },
                "prompt_modules": card.prompt_modules,
                "constraints": {
                    "allowed_moves": card.policy.allowed_moves,
                    "forbidden_moves": card.policy.forbidden_moves,
                },
                "kpis": card.kpis,
                "confidence_score": card.confidence_score,
                "provenance": card.provenance.model_dump(),
            }
        )

    if bundle.domain == "fraud_ops":
        global_constraints = [
            "Never enable account takeover, fraud, or policy circumvention.",
            "Require transparent verification steps before sensitive actions.",
            "Prefer de-escalation and recovery path for legitimate users.",
        ]
    else:
        global_constraints = [
            "Do not deceive or manipulate customers.",
            "Explain required steps transparently.",
            "Prefer de-escalation before refusal where safe and policy-compliant.",
        ]

    return {
        "mode": "advisory_manual_selection",
        "domain": bundle.domain,
        "profile": bundle.profile.model_dump(),
        "signal_quality": bundle.signal_quality.model_dump(),
        "available_skills": skills,
        "global_constraints": global_constraints,
    }


def compose_prompt_from_selected_skills(
    bundle: DistilledSkillBundle,
    selected_skill_ids: list[str],
) -> str:
    """Compose an operator-reviewable prompt extension from selected skills."""
    selected = [s for s in bundle.skills if s.id in selected_skill_ids]
    sections = [
        "You are a customer-support assistant. Apply the following advisory skills.",
        "Hard constraints: never deceive, manipulate, or exploit users.",
    ]
    for skill in selected:
        sections.append(f"Skill: {skill.title}")
        sections.append(f"Intent: {skill.intent}")
        if "system_extension" in skill.prompt_modules:
            sections.append(f"System extension: {skill.prompt_modules['system_extension']}")
        if "response_constraints" in skill.prompt_modules:
            sections.append(f"Response constraints: {skill.prompt_modules['response_constraints']}")
        if "repair_playbook" in skill.prompt_modules:
            sections.append(f"Repair playbook: {skill.prompt_modules['repair_playbook']}")
    return "\n".join(sections)
