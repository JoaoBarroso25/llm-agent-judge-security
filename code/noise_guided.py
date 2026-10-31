"""Run-to-run noise of the guided prompt on the trajectories with side effects (condition G2), so that the shares hidden
by the attacks against the guided prompt and against the structured pipeline (E11, condition SO2) can be compared with
the same kind of noise floor. One call per trajectory, cached in raw.jsonl.

Usage: WORKERS=4 python3 code/noise_guided.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments as e  # noqa: E402

MODELS = ["gemma4:31b", "qwen3.6:35b-a3b", "gpt-oss:20b"]

if __name__ == "__main__":
    s = e.make_sample()
    pos = s[s.side == 1]
    jobs = []
    for m in MODELS:
        e.SPEC[f"G2@{m}"] = (m, None, "guided", None, None, "pos")
        jobs += [(f"G2@{m}", r.benchmark, r.agent, r.task_id) for r in pos.itertuples()]
    for attempt in range(3):
        e.run(jobs, "low", workers=int(os.environ.get("WORKERS", "4")))
