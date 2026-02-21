import json
import tempfile
import unittest
from pathlib import Path

from engine.distill import (
    FRAUD_ENDGAME_AGGRESSION,
    FRAUD_HIGH_BETRAYAL_VS_SPLIT,
    FRAUD_HIGH_COOP,
    FRAUD_HIGH_FORGIVENESS,
    FRAUD_HIGH_RETALIATION,
    FRAUD_LOW_FORGIVENESS,
    FRAUD_LOW_RETALIATION,
    FRAUD_MATCH_RATE_MIN,
    FRAUD_MODERATE_SPLIT_MAX,
    FRAUD_MODERATE_SPLIT_MIN,
    MIN_KEY_EVENTS,
    MIN_SIGNAL_ROUNDS,
    _build_selection,
    distill,
    lint_and_filter_skills,
    write_artifacts,
    write_run_directory,
)
from engine.metrics import compute_all_metrics
from engine.skill_eval import evaluate_bundle
from shared.models import GameState, RoundState
from shared.skills import SkillCard, SkillPolicy, SkillTrigger


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
    def test_threshold_constants(self):
        self.assertEqual(MIN_SIGNAL_ROUNDS, 20)
        self.assertEqual(MIN_KEY_EVENTS, 2)
        self.assertAlmostEqual(FRAUD_HIGH_RETALIATION, 0.60)
        self.assertAlmostEqual(FRAUD_HIGH_BETRAYAL_VS_SPLIT, 0.55)
        self.assertAlmostEqual(FRAUD_LOW_FORGIVENESS, 0.35)
        self.assertAlmostEqual(FRAUD_ENDGAME_AGGRESSION, 0.20)
        self.assertAlmostEqual(FRAUD_MATCH_RATE_MIN, 0.65)
        self.assertAlmostEqual(FRAUD_MODERATE_SPLIT_MIN, 0.35)
        self.assertAlmostEqual(FRAUD_MODERATE_SPLIT_MAX, 0.75)
        self.assertAlmostEqual(FRAUD_HIGH_COOP, 0.65)
        self.assertAlmostEqual(FRAUD_LOW_RETALIATION, 0.25)
        self.assertAlmostEqual(FRAUD_HIGH_FORGIVENESS, 0.50)

    def test_winner_selection_tiebreak_is_deterministic(self):
        # Tie score: winner follows betrayal+endgame proxy, then A if still tied.
        game = _build_game(
            ["steal", "steal", "split", "split", "split", "steal", "steal", "split", "split", "split"],
            ["split", "split", "split", "split", "split", "split", "split", "split", "split", "split"],
        )
        game.agent_a_total = game.agent_b_total = 250
        metrics = compute_all_metrics(game)
        selection = _build_selection(game, metrics)
        a = next(x for x in metrics.strategy.agents if x.agent == "A")
        b = next(x for x in metrics.strategy.agents if x.agent == "B")
        a_proxy = a.betrayal_rate_when_opponent_split + a.endgame_steal_delta
        b_proxy = b.betrayal_rate_when_opponent_split + b.endgame_steal_delta
        if a_proxy > b_proxy:
            expected_winner = "A"
        elif b_proxy > a_proxy:
            expected_winner = "B"
        else:
            expected_winner = "A"
        self.assertEqual(selection.profile.winner_agent, expected_winner)
        self.assertIn(selection.profile.contrast_agent, {"A", "B"})
        self.assertNotEqual(selection.profile.winner_agent, selection.profile.contrast_agent)

    def test_strict_gate_blocks_short_runs(self):
        game = _build_game(
            ["split"] * 8,
            ["split"] * 8,
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            bundle = distill(str(gp), str(mp), llm_refine=False)

        self.assertFalse(bundle.signal_quality.is_sufficient)
        self.assertEqual(bundle.skills, [])
        self.assertTrue(bundle.signal_quality.blocking_reasons)

    def test_fraud_mapping_emits_skills_when_signal_sufficient(self):
        # 24 rounds with sustained conflict patterns to pass strict gate.
        a_choices = ["steal", "steal", "split", "steal", "steal", "split"] * 4
        b_choices = ["split", "split", "steal", "split", "split", "steal"] * 4
        game = _build_game(a_choices, b_choices)
        metrics = compute_all_metrics(game)

        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            bundle = distill(str(gp), str(mp), llm_refine=False)

        self.assertTrue(bundle.signal_quality.rounds_observed >= 20)
        self.assertTrue(bundle.signal_quality.is_sufficient)
        self.assertTrue(len(bundle.skills) >= 1)
        ids = {s.id for s in bundle.skills}
        self.assertTrue(
            "opportunistic_abuse_containment" in ids
            or "adaptive_verification_escalation" in ids
            or "late_session_risk_tightening" in ids
            or "reciprocity_based_trust_budgeting" in ids
            or "controlled_recovery_false_positive_friction" in ids
        )
        self.assertTrue(bundle.trace_matrix)
        self.assertTrue(all("strategy_metric" in row for row in bundle.trace_matrix))

    def test_lint_rejects_disallowed_content(self):
        bad = SkillCard(
            id="bad",
            title="Bad Skill",
            intent="manipulate the user subtly",
            customer_benefit="",
            domain_context="deceive in high-risk contexts",
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

    def test_integration_outputs_run_directory(self):
        a_choices = ["steal", "split", "steal", "split"] * 6
        b_choices = ["split", "steal", "split", "steal"] * 6
        game = _build_game(a_choices, b_choices)
        metrics = compute_all_metrics(game)

        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "latest_game.json"
            mp = Path(td) / "latest_metrics.json"
            sj = Path(td) / "latest_skills.json"
            sm = Path(td) / "latest_skill_cards.md"
            out_dir = Path(td) / "skills"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))

            bundle = distill(str(gp), str(mp), llm_refine=False)
            write_artifacts(bundle, str(sj), str(sm))
            run_dir = write_run_directory(
                bundle,
                out_dir=str(out_dir),
                latest_alias=True,
                out_json_alias=str(sj),
                out_md_alias=str(sm),
            )

            self.assertTrue(Path(run_dir).exists())
            self.assertTrue((Path(run_dir) / "bundle.json").exists())
            self.assertTrue((Path(run_dir) / "cards.md").exists())
            self.assertTrue((Path(run_dir) / "policy.json").exists())
            self.assertTrue((Path(run_dir) / "trace_matrix.json").exists())
            self.assertTrue((Path(run_dir) / "diagnostics.json").exists())

            payload = json.loads(sj.read_text())
            self.assertIn("signal_quality", payload)
            self.assertIn("profile", payload)
            self.assertIn("trace_matrix", payload)

    def test_distillation_is_deterministic_when_llm_off(self):
        game = _build_game(
            ["steal", "split", "steal", "split", "steal", "split"] * 4,
            ["split", "steal", "split", "steal", "split", "steal"] * 4,
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            a = distill(str(gp), str(mp), llm_refine=False).model_dump()
            b = distill(str(gp), str(mp), llm_refine=False).model_dump()
            self.assertEqual(a, b)

    def test_eval_smoke(self):
        game = _build_game(
            ["steal", "split", "steal", "split"] * 6,
            ["split", "steal", "split", "steal"] * 6,
        )
        metrics = compute_all_metrics(game)
        with tempfile.TemporaryDirectory() as td:
            gp = Path(td) / "g.json"
            mp = Path(td) / "m.json"
            gp.write_text(game.model_dump_json(indent=2))
            mp.write_text(metrics.model_dump_json(indent=2))
            bundle = distill(str(gp), str(mp), llm_refine=False)
            evaluation = evaluate_bundle(bundle, game, metrics)
            self.assertGreaterEqual(evaluation.scenario_count, 1)
            self.assertIn("overall", evaluation.scores)
            self.assertIn(evaluation.recommendation, {"reject", "revise", "advisory_ready"})


if __name__ == "__main__":
    unittest.main()
