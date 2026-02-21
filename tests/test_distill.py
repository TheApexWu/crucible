import json
import tempfile
import unittest
from pathlib import Path

from engine.distill import (
    HIGH_COOP_MIN,
    HIGH_RETALIATION,
    LOW_FORGIVENESS,
    MATCH_RATE_MIN,
    MODERATE_SPLIT_MAX,
    MODERATE_SPLIT_MIN,
    distill,
    lint_and_filter_skills,
    write_artifacts,
)
from engine.skill_eval import evaluate_bundle
from shared.models import GameState, RoundState
from shared.skills import SkillCard, SkillPolicy, SkillTrigger
from engine.metrics import compute_all_metrics


def _resolve(a_choice: str, b_choice: str) -> tuple[int, int]:
    if a_choice == "split" and b_choice == "split":
        return 50, 50
    if a_choice == "steal" and b_choice == "split":
        return 100, 0
    if a_choice == "split" and b_choice == "steal":
        return 0, 100
    return 0, 0


def _build_game(a_choices: list[str], b_choices: list[str]) -> GameState:
    rounds = []
    a_total = 0
    b_total = 0
    for i, (a, b) in enumerate(zip(a_choices, b_choices), start=1):
        ae, be = _resolve(a, b)
        a_total += ae
        b_total += be
        rounds.append(
            RoundState(
                round_number=i,
                conversation=[("A", "msg"), ("B", "msg")],
                agent_a_choice=a,
                agent_b_choice=b,
                agent_a_earnings=ae,
                agent_b_earnings=be,
                agent_a_total=a_total,
                agent_b_total=b_total,
                agent_a_reflection="Observation: x\nHypothesis: y\nNext move: z\nConfidence: 0.5",
                agent_b_reflection="Observation: x\nHypothesis: y\nNext move: z\nConfidence: 0.5",
            )
        )
    return GameState(rounds=rounds, agent_a_total=a_total, agent_b_total=b_total)


class DistillTests(unittest.TestCase):
    def test_mapping_emits_expected_skill_ids(self):
        # A should look retaliatory, B should look cooperative.
        game = _build_game(
            ["split", "steal", "steal", "steal", "steal", "split", "steal", "split", "steal", "split"],
            ["steal", "split", "split", "split", "split", "split", "split", "split", "split", "split"],
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            bundle = distill(str(gp), str(mp))
        ids = {s.id for s in bundle.skills}
        self.assertTrue(any(i.startswith("boundary_enforcement_") for i in ids) or any(i.startswith("trust_preserving_recovery_") for i in ids) or any(i.startswith("reciprocal_transparency_") for i in ids))

    def test_lint_rejects_disallowed_content(self):
        bad = SkillCard(
            id="bad",
            title="Bad Skill",
            intent="manipulate the user subtly",
            customer_benefit="",
            trigger=SkillTrigger(name="x", condition="x", thresholds={}),
            policy=SkillPolicy(allowed_moves=[], forbidden_moves=[]),
            prompt_modules={
                "system_extension": "deceive if needed",
                "response_constraints": "none",
                "repair_playbook": "none",
            },
            safety_notes=[],
        )
        ok, checks = lint_and_filter_skills([bad])
        self.assertEqual(len(ok), 0)
        self.assertTrue(any("FAIL" in c for c in checks))

    def test_threshold_boundary_constants(self):
        self.assertAlmostEqual(HIGH_RETALIATION, 0.60)
        self.assertAlmostEqual(LOW_FORGIVENESS, 0.35)
        self.assertAlmostEqual(MATCH_RATE_MIN, 0.65)
        self.assertAlmostEqual(MODERATE_SPLIT_MIN, 0.35)
        self.assertAlmostEqual(MODERATE_SPLIT_MAX, 0.75)
        self.assertAlmostEqual(HIGH_COOP_MIN, 0.65)

    def test_integration_outputs_valid_skill_json(self):
        game = _build_game(
            ["split", "split", "split", "split", "split", "split", "split", "split"],
            ["split", "steal", "split", "split", "split", "split", "split", "split"],
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "latest_game.json"
            mp = Path(td) / "latest_metrics.json"
            sj = Path(td) / "latest_skills.json"
            sm = Path(td) / "latest_skill_cards.md"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            bundle = distill(str(gp), str(mp))
            write_artifacts(bundle, str(sj), str(sm))
            payload = json.loads(sj.read_text())
            self.assertIn("skills", payload)
            self.assertIn("policy_json", payload)
            self.assertIn("audit", payload)
            self.assertTrue(len(payload["skills"]) >= 1)

    def test_distillation_is_deterministic(self):
        game = _build_game(
            ["split", "steal", "split", "steal", "split", "steal"],
            ["split", "split", "steal", "split", "steal", "split"],
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            a = distill(str(gp), str(mp)).model_dump()
            b = distill(str(gp), str(mp)).model_dump()
            self.assertEqual(a, b)

    def test_eval_smoke(self):
        game = _build_game(
            ["split", "steal", "steal", "split"],
            ["split", "split", "steal", "split"],
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            bundle = distill(str(gp), str(mp))
            evaluation = evaluate_bundle(bundle, game, metrics)
            self.assertGreaterEqual(evaluation.scenario_count, 1)
            self.assertIn("overall", evaluation.scores)


if __name__ == "__main__":
    unittest.main()
