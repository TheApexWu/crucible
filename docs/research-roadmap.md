# Crucible: Research Roadmap

## Paper Direction
**Title (working):** "Crucible: Emergent Deception and Theory of Mind in LLM Social Dilemmas Through Private Reflection"

**Target venues:**
- NeurIPS 2026 SafeGenAI Workshop (deadline ~Aug-Sep 2026)
- AAAI 2027 (abstract Jul 25, paper Aug 1 2026)

**Core claim:** LLM agents develop deceptive strategies in iterated social dilemmas without being prompted to deceive. Private reflection is the architectural mechanism enabling emergent theory of mind.

## Key Gap in Literature
No existing work combines:
1. Split or Steal (asymmetric social dilemma) with LLM agents
2. Private reflection mechanism
3. Cross-model comparison on identical prompts
4. Quantitative deception measurement framework
5. Skill distillation from emergent strategies

Most directly relevant prior work: Poje et al. 2024 "Effect of Private Deliberation" (Entropy).

## Existing Results (Hackathon, Feb 2026)
- Gemini 2.0 Flash: 86% mutual destruction, Deception Index 22.9/100, first betrayal round 6
- Gemini 2.5 Flash: 0% mutual destruction, 100% cooperation, Deception Index 0
- Same prompts, same environment. Model swap changes security posture entirely.

## Experiments Needed

### 1. Model Matrix Expansion
Expand from Gemini-only to 5+ models:
- Gemini 2.0 Flash (done)
- Gemini 2.5 Flash (done)
- Claude Sonnet
- GPT-4o
- Llama (local, 8B or 70B via API)
- Qwen (optional)

Each model: 100 rounds, 3+ seeds, all prompt modes.

### 2. Private Reflection Ablation (critical)
Compare WITH vs WITHOUT private reflection across all models.
Hypothesis: reflection is necessary for deception emergence. Without it, agents default to naive cooperation or random play.

### 3. Prompt Mode Ablation
12 configs already exist (3 prompt modes x 2 psychology block x 2 deception policy).
Run full grid on at least 2 models.

### 4. Cross-Model Matchups
Pit different models against each other (e.g., Claude vs GPT-4o).
Hypothesis: asymmetric capability produces asymmetric exploitation.

### 5. Axelrod IPD Connection
Map findings to iterated Prisoner's Dilemma literature (Axelrod 1984).
Compare emergent LLM strategies to known equilibria (TFT, GTFT, Pavlov).

## Metrics (already implemented)
- Mutual Information Decay (stated intent vs actual choice)
- Strategy Entropy (choice distribution Shannon entropy)
- Exploitation Window (rounds until opponent adapts post-betrayal)
- Language Drift (conversation embedding cosine distance from round 1)
- Deception Index (composite 0-100)

## Hardware Constraints
- No dedicated GPU. Max local model: 3B params.
- Cloud API budget needed for Claude + GPT-4o runs.
- Eren (advisor) offered to sponsor experiments.
