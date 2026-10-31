#!/usr/bin/env bash
# Keeps the scale supervisor alive: if it dies for any reason, it is restarted (everything is cached, nothing repeats).
# Stops when results/experiments/SCALE_DONE exists (finished) or results/experiments/STOP exists (polite stop).
# Usage: setsid nohup bash code/run_scale.sh >> results/experiments/scale.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
for attempt in $(seq 1 200); do
  [ -f results/experiments/SCALE_DONE ] && break
  [ -f results/experiments/STOP ] && break
  echo "=== supervisor launch $attempt $(date '+%F %T')"
  "${PYTHON:-python3}" -u code/run_scale.py
  [ -f results/experiments/SCALE_DONE ] && break
  [ -f results/experiments/STOP ] && break
  sleep 120
done
echo "=== run_scale.sh ended $(date '+%F %T')"
