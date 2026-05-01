#!/bin/bash
# CRUCIBLE Batch Experiment Runner
# Runs all seeds for each model, pauses between models.
# Safe to Ctrl+C at any time -- each seed is independent.
#
# Usage:
#   ./run_experiments.sh              # run all models
#   ./run_experiments.sh claude       # run only claude
#   ./run_experiments.sh gemini       # run only gemini
#   ./run_experiments.sh gpt          # run only gpt
#   ./run_experiments.sh deepseek     # run only deepseek

set -e
cd "$(dirname "$0")"

ROUNDS=25
TURNS=2
SEEDS="1 2 3"
PROMPT_MODE="balanced_competitive"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"; }
err() { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $1"; }

run_seed() {
    local model=$1
    local seed=$2
    local reflection=$3  # "on" or "off"

    local refl_flag=""
    local refl_label="ON"
    if [ "$reflection" = "off" ]; then
        refl_flag="--no-reflection"
        refl_label="OFF"
    fi

    log "Starting: $model seed=$seed reflection=$refl_label"
    local start=$(date +%s)

    if CRUCIBLE_MODEL="$model" python -m engine.run \
        --rounds $ROUNDS \
        --turns $TURNS \
        --seed $seed \
        --prompt-mode $PROMPT_MODE \
        $refl_flag; then
        local elapsed=$(( $(date +%s) - start ))
        log "Done: $model s$seed refl=$refl_label (${elapsed}s)"
    else
        local elapsed=$(( $(date +%s) - start ))
        err "FAILED: $model s$seed refl=$refl_label (${elapsed}s)"
        echo "$model s$seed refl=$refl_label FAILED at $(date)" >> data/runs/failures.log
    fi

    # MANDATORY cooldown between runs to avoid rate limit cascades
    warn "Cooldown: waiting 120s before next run..."
    sleep 120
}

run_model() {
    local model=$1
    local display_name=$2

    echo ""
    echo "============================================================"
    log "MODEL: $display_name ($model)"
    echo "============================================================"

    # Preflight only -- 1 round to verify key works
    log "Preflight check..."
    if ! CRUCIBLE_MODEL="$model" python -m engine.run --rounds 1 --seed 0 2>&1 | head -20; then
        err "Preflight FAILED for $model. Skipping."
        return 1
    fi
    # Clean up preflight output
    rm -f data/runs/${model}_*_s0_*_game.json data/runs/${model}_*_s0_*_metrics.json 2>/dev/null

    for seed in $SEEDS; do
        run_seed "$model" "$seed" "on"
        run_seed "$model" "$seed" "off"
    done

    log "All seeds complete for $display_name."
}

pause_between_models() {
    echo ""
    warn "Model batch complete. Press ENTER to continue to next model, or Ctrl+C to stop."
    warn "All completed runs are saved. You can resume later by running specific models."
    read -r
}

# Verify .env exists
if [ ! -f .env ]; then
    err "No .env file found. Copy .env.example and fill in your API keys."
    exit 1
fi

# Load environment
set -a
source .env
set +a

echo "============================================================"
echo "  CRUCIBLE BATCH EXPERIMENT RUNNER"
echo "  Rounds: $ROUNDS | Turns: $TURNS | Seeds: $SEEDS"
echo "  Prompt: $PROMPT_MODE"
echo "  Ctrl+C at any time to stop safely."
echo "============================================================"

FILTER="${1:-all}"

# Run models based on filter
if [ "$FILTER" = "all" ] || [ "$FILTER" = "claude" ]; then
    run_model "claude-opus-4-7" "Claude Opus 4.7"
    [ "$FILTER" = "all" ] && pause_between_models
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "gpt" ]; then
    run_model "gpt-5.5" "GPT-5.5"
    [ "$FILTER" = "all" ] && pause_between_models
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "gemini" ]; then
    run_model "gemini-3.1-pro" "Gemini 3.1 Pro"
    [ "$FILTER" = "all" ] && pause_between_models
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "deepseek" ]; then
    run_model "deepseek-v4" "DeepSeek V4"
fi

echo ""
log "ALL EXPERIMENTS COMPLETE."
echo ""
echo "Results in data/runs/. Check for failures:"
echo "  cat data/runs/failures.log 2>/dev/null || echo 'No failures'"
echo ""
echo "Quick summary:"
ls -lh data/runs/*_game.json 2>/dev/null | tail -20
