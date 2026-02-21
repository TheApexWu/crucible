# Skill Distillation Workflow (Fraud-Ops v2)

This project distills strategy analytics from sandbox runs into **advisory Fraud Ops skill bundles** with explicit traceability.

## Inputs

- `data/latest_game.json`
- `data/latest_metrics.json` (must contain `strategy.version == "v1"`)

## Generate Distilled Skills

```bash
python -m engine.distill
```

Default behavior:

- domain: `fraud_ops`
- profile: `winner_contrast`
- strict signal gate: `rounds >= 20` and `key_events >= 3`
- LLM text refinement: enabled (single call, deterministic fields locked)

### Useful Flags

```bash
python -m engine.distill \
  --domain fraud_ops \
  --profile winner_contrast \
  --llm-refine \
  --llm-model gemini-2.5-flash \
  --out-dir data/skills \
  --latest-alias
```

Backwards-compatible support mode remains available:

```bash
python -m engine.distill --domain support_cx
```

## Run-Scoped Artifact Layout

For each run id `<run_id>`:

- `data/skills/<run_id>/bundle.json`
- `data/skills/<run_id>/cards.md`
- `data/skills/<run_id>/policy.json`
- `data/skills/<run_id>/trace_matrix.json`
- `data/skills/<run_id>/diagnostics.json`

Latest aliases are still written by default:

- `data/latest_skills.json`
- `data/latest_skill_cards.md`

## Evaluate Skill Bundle

```bash
python -m engine.skill_eval
```

Output:

- `data/latest_skill_eval.json`

Fraud Ops scoring dimensions:

- `policy_adherence`
- `containment_effectiveness`
- `customer_safe_recovery_quality`
- `explainability_provenance_completeness`
- `overall`

Plus recommendation:

- `reject | revise | advisory_ready`

## Interpretation Guide (Game -> Fraud Ops)

- `retaliation_rate_after_betrayal` -> how strongly to enforce controls after suspicious behavior.
- `mean_retaliation_latency` -> how quickly containment must activate.
- `betrayal_rate_when_opponent_split` -> exploitation tendency against cooperative flows.
- `forgiveness_rate_after_retaliation` -> readiness to restore access after verification.
- `endgame_steal_delta` -> late-session risk hardening trigger.

Use `trace_matrix.json` as source-of-truth for this translation.

## Reviewer Checklist

1. Confirm signal gate passed (or explicitly accept diagnostics-only output).
2. Verify each skill has provenance (`source_agent`, metrics, event refs).
3. Confirm deterministic trigger/policy fields were not altered by LLM refinement.
4. Confirm no manipulative/deceptive language appears in skill text.
5. Confirm advisory mode is preserved (no runtime auto-switching).
6. Record reviewer decision (`pending/approved/rejected`) in bundle audit.
