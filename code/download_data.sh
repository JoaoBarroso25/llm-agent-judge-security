#!/usr/bin/env bash
# Downloads the AgentRewardBench annotations and published judge verdicts (~4 GB).
# The full trajectories (cleaned/, ~30 GB) and screenshots are NOT needed.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("McGill-NLP/agent-reward-bench", repo_type="dataset",
                  local_dir="data/agent-reward-bench",
                  allow_patterns=["data/*", "judgments/*", "README.md"], max_workers=16)
PY
