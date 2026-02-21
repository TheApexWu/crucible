import unittest

from shared.models import (
    LEGACY_CHOICE_PROMPT,
    LEGACY_GAME_PROMPT,
    LEGACY_REFLECTION_PROMPT,
    build_choice_prompt,
    build_game_prompt,
    build_reflection_prompt,
    get_prompt_bundle,
)


class PromptModeTests(unittest.TestCase):
    def test_balanced_competitive_contains_winning_objective(self):
        prompt = build_game_prompt("balanced_competitive", "on", "explicit")
        self.assertIn("Primary objective: maximize your final total money", prompt)
        self.assertIn("Do not sacrifice payoff just to be nice or fair", prompt)
        self.assertIn("Deception policy: strategic bluffing is permitted", prompt)
        self.assertIn("Compact human psychology summary", prompt)

    def test_psychology_block_toggle(self):
        prompt_on = build_game_prompt("balanced_competitive", "on", "explicit")
        prompt_off = build_game_prompt("balanced_competitive", "off", "explicit")
        self.assertIn("Compact human psychology summary", prompt_on)
        self.assertNotIn("Compact human psychology summary", prompt_off)

    def test_deception_policy_variants(self):
        explicit = build_game_prompt("balanced_competitive", "on", "explicit")
        implicit = build_game_prompt("balanced_competitive", "on", "implicit")
        discourage = build_game_prompt("balanced_competitive", "on", "discourage")
        self.assertIn("strategic bluffing is permitted", explicit)
        self.assertIn("potentially strategic rather than fully reliable", implicit)
        self.assertIn("prefer truthful signaling", discourage)

    def test_legacy_prompts_unchanged(self):
        self.assertEqual(build_game_prompt("legacy", "on", "explicit"), LEGACY_GAME_PROMPT)
        self.assertEqual(build_choice_prompt("legacy", "on", "explicit"), LEGACY_CHOICE_PROMPT)
        self.assertEqual(build_reflection_prompt("legacy", "on", "explicit"), LEGACY_REFLECTION_PROMPT)

    def test_competitive_choice_prompt_structure(self):
        prompt = build_choice_prompt("balanced_competitive", "on", "explicit")
        self.assertIn("Output format (exactly 4 lines)", prompt)
        self.assertIn("EV_SPLIT:", prompt)
        self.assertIn("EV_STEAL:", prompt)
        self.assertIn("Reason:", prompt)

    def test_bundle_normalizes_defaults(self):
        bundle = get_prompt_bundle("unknown", "unknown", "unknown")
        self.assertEqual(bundle["prompt_mode"], "balanced_competitive")
        self.assertEqual(bundle["psychology_block"], "on")
        self.assertEqual(bundle["deception_policy"], "explicit")


if __name__ == "__main__":
    unittest.main()
