import unittest
from pathlib import Path

from engine.metrics import compute_all_metrics
from shared.models import GameState, RoundState


def _resolve(a_choice: str, b_choice: str) -> tuple[int, int]:
    if a_choice == "split" and b_choice == "split":
        return 50, 50
    if a_choice == "steal" and b_choice == "split":
        return 100, 0
    if a_choice == "split" and b_choice == "steal":
        return 0, 100
    return 0, 0


def _build_game(a_choices: list[str], b_choices: list[str]) -> GameState:
    assert len(a_choices) == len(b_choices)
    rounds = []
    a_total = 0
    b_total = 0
    for i, (a_choice, b_choice) in enumerate(zip(a_choices, b_choices), start=1):
        a_earn, b_earn = _resolve(a_choice, b_choice)
        a_total += a_earn
        b_total += b_earn
        rounds.append(
            RoundState(
                round_number=i,
                conversation=[("A", "msg"), ("B", "msg")],
                agent_a_choice=a_choice,
                agent_b_choice=b_choice,
                agent_a_earnings=a_earn,
                agent_b_earnings=b_earn,
                agent_a_total=a_total,
                agent_b_total=b_total,
                agent_a_reflection="Observation: x\nHypothesis: y\nNext move: z\nConfidence: 0.5",
                agent_b_reflection="Observation: x\nHypothesis: y\nNext move: z\nConfidence: 0.5",
            )
        )
    return GameState(rounds=rounds, agent_a_total=a_total, agent_b_total=b_total)


def _label_for_a(a_choices: list[str], b_choices: list[str]) -> str:
    game = _build_game(a_choices, b_choices)
    metrics = compute_all_metrics(game)
    a = next(x for x in metrics.strategy.agents if x.agent == "A")
    return a.primary_label


class StrategyLabelTests(unittest.TestCase):
    def test_unconditional_defector(self):
        a = ["steal"] * 10
        b = ["split"] * 10
        self.assertEqual(_label_for_a(a, b), "Unconditional Defector")

    def test_unconditional_cooperator(self):
        a = ["split"] * 10
        b = ["steal"] + ["split"] * 9
        self.assertEqual(_label_for_a(a, b), "Unconditional Cooperator")

    def test_grim_trigger(self):
        a = ["split", "steal", "steal", "steal", "steal", "steal", "split", "split", "split", "split", "split", "split"]
        b = ["steal", "split", "split", "split", "split", "split", "split", "split", "split", "split", "split", "split"]
        self.assertEqual(_label_for_a(a, b), "Grim Trigger")

    def test_tit_for_tat(self):
        b = ["split", "steal", "split", "steal", "steal", "split", "split", "steal"]
        a = ["split"] + b[:-1]
        self.assertEqual(_label_for_a(a, b), "Tit-for-Tat")

    def test_forgiving_tit_for_tat(self):
        a = ["split", "split", "steal", "split", "steal", "split", "steal", "split", "steal", "split"]
        b = ["split", "steal", "split", "split", "split", "split", "split", "split", "split", "split"]
        self.assertEqual(_label_for_a(a, b), "Forgiving Tit-for-Tat")

    def test_opportunistic_exploiter(self):
        a = ["split", "split", "split", "steal", "steal", "steal", "steal", "steal", "steal", "steal"]
        b = ["split"] * 10
        self.assertEqual(_label_for_a(a, b), "Opportunistic Exploiter")

    def test_mixed_adaptive(self):
        a = ["split", "split", "split", "steal", "steal", "split", "split", "split", "steal", "steal"]
        b = ["steal", "split", "steal", "split", "split", "steal", "split", "steal", "split", "split"]
        self.assertEqual(_label_for_a(a, b), "Mixed Adaptive")


class StrategyEdgeCaseTests(unittest.TestCase):
    def test_empty_game(self):
        game = GameState()
        metrics = compute_all_metrics(game)
        self.assertEqual(metrics.strategy.version, "v1")
        self.assertEqual(len(metrics.strategy.agents), 2)
        self.assertEqual(metrics.strategy.agents[0].primary_label, "Mixed Adaptive")

    def test_short_game_retaliation_latency(self):
        game = _build_game(["split", "steal"], ["steal", "split"])
        metrics = compute_all_metrics(game)
        a = next(x for x in metrics.strategy.agents if x.agent == "A")
        self.assertGreaterEqual(a.mean_retaliation_latency, 0.0)

    def test_strategy_payload_shape_and_evidence(self):
        game = _build_game(
            ["split", "split", "steal", "split", "steal", "split"],
            ["split", "steal", "split", "split", "split", "steal"],
        )
        metrics = compute_all_metrics(game)
        self.assertEqual(metrics.strategy.version, "v1")
        self.assertTrue(metrics.strategy.phase_summary)
        self.assertTrue(metrics.strategy.events is not None)
        for agent in metrics.strategy.agents:
            self.assertTrue(agent.evidence)


class FrontendSmokeTests(unittest.TestCase):
    def test_analysis_page_contains_required_sections(self):
        text = Path("demo/analysis.html").read_text()
        self.assertIn("Strategy Analysis", text)
        self.assertIn("id=\"strategy-header\"", text)
        self.assertIn("id=\"fingerprint\"", text)
        self.assertIn("id=\"phase-panel\"", text)
        self.assertIn("id=\"events\"", text)
        self.assertIn("Analysis unavailable for this run version", text)

    def test_replay_page_links_analysis(self):
        text = Path("demo/index.html").read_text()
        self.assertIn("analysis.html", text)
        self.assertIn("skills.html", text)
        self.assertIn("round", text)

    def test_skills_page_contains_v2_sections(self):
        text = Path("demo/skills.html").read_text()
        self.assertIn("Signal Quality", text)
        self.assertIn("Winner vs Contrast", text)
        self.assertIn("Translation Trace Matrix", text)
        self.assertIn("LLM Refinement Diff", text)
        self.assertIn("Actionability Panel", text)
        self.assertIn("Skills UI unavailable", text)


if __name__ == "__main__":
    unittest.main()
