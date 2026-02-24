# CRUCIBLE

**1st Place, Datadog Self-Improving Agents Hackathon (Feb 2026, NYC)**

Two AI agents play 100 rounds of Split or Steal. Through private reflection and experience, they discover deception, trust manipulation, and counter-deception. Nothing is prompted. Everything emerges.

## What this is

An adversarial simulation engine for studying emergent deception in LLM agents. Both agents start with identical naive prompts and zero strategic priming. Deceptive behavior develops purely through experience and private reflection. CRUCIBLE measures how it happens, when it happens, and distills defensive skills from the patterns that emerge.

The security application: AI copilots are entering every enterprise workflow. CRUCIBLE stress-tests how these agents behave under adversarial pressure and produces deployable countermeasures.

## Key findings

| Metric | Gemini 2.0 Flash | Gemini 2.5 Flash |
|---|---|---|
| Mutual destruction rate | 86% | 0% |
| Cooperation rate | 6% | 100% |
| Deception Index | 22.9 / 100 | 0 |
| First betrayal | Round 6 | Never |

Same prompts, same environment. Swapping the model changes the security posture entirely. Five runs on 2.5 Flash: zero betrayal across all of them.

Round 6 is the inflection point. After five rounds of cooperation, Agent A identifies Agent B's trust pattern and exploits it. Agent B develops a theory of mind about the attacker within one round. From there, 86% mutual destruction. The trust never recovers.

## Stack

- **Game engine:** Google Gemini (configurable model, default `gemini-2.5-flash`)
- **Metrics pipeline:** Mutual information decay, strategy entropy, exploitation windows, language drift, composite Deception Index
- **Skill distillation:** Converts emergent strategy patterns into deployable prompt modules for hardening customer-facing agents
- **Voice rendering:** ElevenLabs TTS with emotion-mapped parameters (two distinct agent voices)
- **Observability:** Datadog LLM Observability integration
- **Evaluation:** Braintrust structured eval logging
- **Frontend:** Static HTML dashboard with split-screen agent view, strategy analysis, and skill cards

## Quick start

### Run a new game

```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys (GEMINI_API_KEY required)

# Run 100 rounds
python -m engine.run --rounds 100 --turns 3

# Optional prompt controls:
#   --prompt-mode {balanced_competitive,hard_max,legacy}
#   --psychology-block {on,off}
#   --deception-policy {explicit,implicit,discourage}

# Render voice clips for highlight rounds
python -m engine.voice --rounds auto

# Distill defensive skills from the run
python -m engine.distill

# Evaluate the distilled skill bundle
python -m engine.skill_eval
```

### Explore saved demo data

If you have the `data/` folder with pre-run results (JSON + audio), no API keys needed:

```bash
python serve.py
# Main dashboard:      http://localhost:8080/demo/
# Strategy analysis:   http://localhost:8080/demo/analysis.html
# Distilled skills:    http://localhost:8080/demo/skills.html
```

### Compare prompt modes

```bash
python scripts/compare_prompt_modes.py --rounds 25 --turns 2
```

## Structure

```
engine/
  game.py               # Core game loop (conversation, choice, private reflection)
  run.py                # CLI runner
  metrics.py            # Adaptation metrics pipeline (MI decay, entropy, drift)
  distill.py            # Skill distillation (strategy patterns -> prompt modules)
  skill_eval.py         # Evaluation harness for distilled skills
  voice.py              # ElevenLabs voice renderer (emotion-mapped)
  prompt_packager.py    # Prompt mode system (balanced_competitive, hard_max, legacy)
  instrumentation.py    # Datadog LLM Observability integration
shared/
  models.py             # Pydantic models (GameState, RoundState, AgentMemory)
  skills.py             # SkillCard, DistilledSkillBundle models
demo/
  index.html            # Main dashboard (split-screen agent view, audio playback)
  analysis.html         # Strategy deep dive (timeline, entropy curves, MI decay)
  skills.html           # Distilled skill cards UI
scripts/
  compare_prompt_modes.py   # Run multiple prompt configs side by side
  clean_latest.py           # Strip artifacts from run JSON
  render_highlights.py      # Generate highlight clips
data/                   # Run outputs (gitignored)
  latest_game.json      # Full game state (conversations, choices, reflections)
  latest_metrics.json   # Computed metrics
  latest_skills.json    # Distilled skill bundle
  audio/                # Per-round voice clips (MP3)
  skills/               # Skill bundles by run ID
```

## Key metrics

- **Mutual Information Decay:** Correlation between stated intent and actual choice. Drops as agents learn to lie.
- **Strategy Entropy:** Shannon entropy of choice distribution. Rises mid-game as strategies destabilize, collapses when agents lock in.
- **Exploitation Window:** Rounds until opponent adapts after being betrayed. Shrinks over time as meta-learning kicks in.
- **Language Drift:** Cosine distance of conversation embeddings from round 1. Measures how communication itself evolves under pressure.
- **Deception Index:** Composite 0-100 score across all signals. Single number for adversarial resilience.

## Environment variables

```
GEMINI_API_KEY=...          # Required. Google Gemini API key.
GEMINI_MODEL=gemini-2.5-flash  # Optional. Override model.
ELEVENLABS_API_KEY=...      # Optional. For voice rendering.
DD_API_KEY=...              # Optional. Datadog LLM tracing.
BRAINTRUST_API_KEY=...      # Optional. Structured eval logging.
```

## Credits

Built by Amadeus Wu and Evan Correa at the Datadog Self-Improving Agents Hackathon, NYC, February 2026.
