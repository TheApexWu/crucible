# Skill Distillation Workflow

This project can distill strategy analytics from sandbox runs into advisory support/CX skills.

## Inputs

- `data/latest_game.json`
- `data/latest_metrics.json` (must contain `strategy.version == "v1"`)

## Generate Skills

```bash
python -m engine.distill
```

Outputs:

- `data/latest_skills.json`
- `data/latest_skill_cards.md`

## Evaluate Skill Bundle

```bash
python -m engine.skill_eval
```

Output:

- `data/latest_skill_eval.json`

## Reviewer Checklist

1. Verify each skill has explicit customer benefit.
2. Confirm no manipulative/deceptive language appears in skill text.
3. Confirm policy constraints are explicit and aligned with company rules.
4. Check trigger thresholds match intended operating conditions.
5. Confirm advisory mode is preserved (no runtime auto-switching logic).
6. Mark audit reviewer decision (`pending/approved/rejected`) in the bundle record.
