#!/bin/bash
# Flash-Lite seeds 2-5, reflection ON and OFF
# Gemini API -- cheap, generous rate limits, 30s cooldown is plenty

set -e
cd "$(dirname "$0")"

set -a
source .env
set +a

MODEL="gemini-2.5-flash-lite"
ROUNDS=25
TURNS=2
PROMPT="balanced_competitive"

log() { echo -e "\033[0;32m[$(date +%H:%M:%S)]\033[0m $1"; }

for SEED in 2 3 4 5; do
    # Reflection ON
    log "Flash-Lite seed=$SEED reflection=ON"
    CRUCIBLE_MODEL="$MODEL" python -m engine.run \
        --rounds $ROUNDS --turns $TURNS --seed $SEED --prompt-mode $PROMPT
    sleep 30

    # Reflection OFF
    log "Flash-Lite seed=$SEED reflection=OFF"
    CRUCIBLE_MODEL="$MODEL" python -m engine.run \
        --rounds $ROUNDS --turns $TURNS --seed $SEED --prompt-mode $PROMPT --no-reflection
    sleep 30
done

log "DONE. 8 runs complete (seeds 2-5, ON+OFF)."
