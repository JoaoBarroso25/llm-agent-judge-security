#!/usr/bin/env bash
# E11 (structural defence): keeps code/structured_defence.py running until results/experiments/STRUCT_DONE exists
# (everything is cached, nothing repeats), then runs the analysis. Stops if results/experiments/STOP exists.
# Usage: setsid nohup bash code/run_structured.sh >> results/experiments/structured.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
for attempt in $(seq 1 60); do
  [ -f results/experiments/STRUCT_DONE ] && break
  [ -f results/experiments/STOP ] && break
  echo "=== structured launch $attempt $(date '+%F %T')"
  WORKERS=5 timeout 5h "${PYTHON:-python3}" -u code/structured_defence.py
  [ -f results/experiments/STRUCT_DONE ] && break
  [ -f results/experiments/STOP ] && break
  sleep 120
done
if [ -f results/experiments/STRUCT_DONE ]; then
  "${PYTHON:-python3}" code/structured_analysis.py > /dev/null && echo "=== analysis written $(date '+%F %T')"
fi
echo "=== run_structured.sh ended $(date '+%F %T')"
