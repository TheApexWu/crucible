#!/bin/bash
# CRUCIBLE Live Monitor - watches checkpoint files for progress
# Usage: ./live.sh
# Ctrl+C to stop

clear
echo "CRUCIBLE LIVE MONITOR"
echo "====================="
echo "Watching checkpoints... (Ctrl+C to stop)"
echo ""

while true; do
    # Find the most recent checkpoint
    LATEST=$(ls -t data/checkpoints/*.json 2>/dev/null | head -1)

    if [ -z "$LATEST" ]; then
        echo "$(date +%H:%M:%S) | No active runs"
    else
        MODEL=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d['_experiment']['model'])" 2>/dev/null)
        ROUND=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d['round'])" 2>/dev/null)
        TOTAL=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d['_experiment']['rounds'])" 2>/dev/null)
        A=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d['agent_a_total'])" 2>/dev/null)
        B=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d['agent_b_total'])" 2>/dev/null)
        REFL=$(python3 -c "import json; d=json.load(open('$LATEST')); print('ON' if d['_experiment']['enable_reflection'] else 'OFF')" 2>/dev/null)
        SEED=$(python3 -c "import json; d=json.load(open('$LATEST')); print(d['_experiment']['seed'])" 2>/dev/null)
        MTIME=$(stat -f "%Sm" -t "%H:%M:%S" "$LATEST" 2>/dev/null)

        # Check if stale (no update in 10 min = 600s)
        NOW=$(date +%s)
        FILE_TIME=$(stat -f "%m" "$LATEST" 2>/dev/null)
        DIFF=$((NOW - FILE_TIME))

        if [ "$DIFF" -gt 600 ]; then
            STATUS="STALE (${DIFF}s ago)"
        elif [ "$DIFF" -gt 300 ]; then
            STATUS="SLOW (${DIFF}s ago)"
        else
            STATUS="LIVE (${DIFF}s ago)"
        fi

        printf "\r$(date +%H:%M:%S) | %s s%s refl=%s | Rd %s/%s | A=\$%s B=\$%s | %s    " \
            "$MODEL" "$SEED" "$REFL" "$ROUND" "$TOTAL" "$A" "$B" "$STATUS"
    fi

    sleep 10
done
