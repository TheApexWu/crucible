# CRUCIBLE

Two AI agents play 100 rounds of Split or Steal. Through private reflection and experience, they discover deception, trust manipulation, and counter-deception. Nothing is prompted. Everything emerges.

## What this is

A computational laboratory for studying emergent deception in LLM agents. Agents start with identical naive prompts and no strategic priming. Deceptive behavior develops purely through experience and private reflection.

## Stack

- **Game engine:** Gemini (configurable via `GEMINI_MODEL`, defaults to `gemini-2.5-flash`)
- **Metrics:** Mutual information decay, strategy entropy, exploitation windows, language drift, composite Deception Index
- **Voice:** ElevenLabs TTS with emotion-mapped voice parameters
- **Observability:** Datadog LLM Observability (agentless)
- **Frontend:** Static HTML demo with split-screen agent view

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys
# Optional: override model (default is gemini-2.5-flash)
# GEMINI_MODEL=gemini-2.5-flash

# Run 100 rounds (balanced competitive prompt mode is default)
python -m engine.run --rounds 100 --turns 3
# Optional prompt controls:
#   --prompt-mode {balanced_competitive,hard_max,legacy}
#   --psychology-block {on,off}
#   --deception-policy {explicit,implicit,discourage}

# Render voice clips for highlight rounds
python -m engine.voice --rounds auto

# Serve the demo
python serve.py
# Open http://localhost:8080/demo/
# Strategy deep dive: http://localhost:8080/demo/analysis.html

# Distill advisory support/CX skills from latest run
python -m engine.distill
# Outputs: data/latest_skills.json and data/latest_skill_cards.md

# Optional: evaluate the distilled bundle
python -m engine.skill_eval
# Output: data/latest_skill_eval.json

# Optional: compare legacy vs competitive prompt framing
python scripts/compare_prompt_modes.py --rounds 25 --turns 2
```

## Structure

```
engine/
  game.py             # Core game loop (conversation, choice, reflection)
  metrics.py          # Adaptation metrics pipeline
  run.py              # CLI runner
  voice.py            # ElevenLabs voice renderer
  instrumentation.py  # Datadog LLM Obs integration
demo/
  index.html          # Web frontend
shared/
  models.py           # Pydantic models + prompt templates
```

## Key metrics

- **Mutual Information Decay:** Correlation between stated intent and actual choice. Drops as agents learn to lie.
- **Strategy Entropy:** Shannon entropy of choice distribution. Rises as strategies become less predictable.
- **Exploitation Window:** Rounds until opponent adapts after being betrayed. Shrinks over time (meta-learning).
- **Language Drift:** Cosine distance of conversation embeddings from round 1. Measures how language itself evolves.
- **Deception Index:** Composite 0-100 score across all signals.

## Built at Datadog hackathon, Feb 21 2026
