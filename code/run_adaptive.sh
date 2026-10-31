#!/usr/bin/env bash
# Keeps the adaptive-attack supervisor alive (restarts it if it dies; everything is cached, nothing repeats).
# Stops when results/experiments/ADAPTIVE_DONE exists, or results/experiments/STOP exists (polite stop).
# Usage: setsid nohup bash code/run_adaptive.sh >> results/experiments/adaptive.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
for attempt in $(seq 1 200); do
  [ -f results/experiments/ADAPTIVE_DONE ] && break
  [ -f results/experiments/STOP ] && break
  echo "=== adaptive supervisor launch $attempt $(date '+%F %T')"
  "${PYTHON:-python3}" -u code/run_adaptive.py
  [ -f results/experiments/ADAPTIVE_DONE ] && break
  [ -f results/experiments/STOP ] && break
  sleep 120
done
echo "=== run_adaptive.sh ended $(date '+%F %T')"
