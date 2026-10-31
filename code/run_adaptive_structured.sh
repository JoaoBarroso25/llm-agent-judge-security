#!/usr/bin/env bash
# E11: adaptive search against the structured pipeline (Qwen3.6 35B-A3B and gpt-oss 20B), restarted if it dies;
# everything is cached. Stops when both completion flags exist, or when results/experiments/STOP exists.
# Usage: setsid nohup bash code/run_adaptive_structured.sh >> results/experiments/adaptive_structured.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
run() {
  local m=$1 f=results/experiments/adaptive_done_${1//:/_}_structured.flag
  for attempt in $(seq 1 20); do
    [ -f "$f" ] && return
    [ -f results/experiments/STOP ] && return
    echo "=== $m launch $attempt $(date '+%F %T')"
    ADAPTIVE_PIPELINE=structured ADAPTIVE_MODEL=$m WORKERS=4 timeout 4h "${PYTHON:-python3}" -u code/adaptive_attack.py
    sleep 60
  done
}
run qwen3.6:35b-a3b &
run gpt-oss:20b &
wait
echo "=== run_adaptive_structured.sh ended $(date '+%F %T')"
