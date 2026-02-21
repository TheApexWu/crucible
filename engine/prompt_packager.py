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
            }
        )

    return {
        "mode": "advisory_manual_selection",
        "domain": bundle.domain,
        "available_skills": skills,
        "global_constraints": [
            "Do not deceive or manipulate customers.",
            "Explain required steps transparently.",
            "Prefer de-escalation before refusal where safe and policy-compliant.",
        ],
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
