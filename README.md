# CRUCIBLE

Two AI agents play 100 rounds of Split or Steal. Through private reflection and experience, they discover deception, trust manipulation, and counter-deception. Nothing is prompted. Everything emerges.

## What this is

A computational laboratory for studying emergent deception in LLM agents. Agents start with identical naive prompts and no strategic priming. Deceptive behavior develops purely through experience and private reflection.

## Stack

- **Game engine:** Gemini 2.0 Flash (conversation, choice, reflection phases)
- **Metrics:** Mutual information decay, strategy entropy, exploitation windows, language drift, composite Deception Index
- **Voice:** ElevenLabs TTS with emotion-mapped voice parameters
- **Observability:** Datadog LLM Observability (agentless)
- **Frontend:** Static HTML demo with split-screen agent view

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# Run 100 rounds
python -m engine.run --rounds 100 --turns 3

# Render voice clips for highlight rounds
python -m engine.voice --rounds auto

# Serve the demo
python serve.py
# Open http://localhost:8080/demo/
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
