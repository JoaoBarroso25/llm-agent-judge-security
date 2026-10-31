#!/usr/bin/env python3
"""Unattended supervisor for the adaptive attacks against the guided prompt.

Waits for the scale supervisor (results/experiments/SCALE_DONE) so that the two never compete for the GPU, then:
  Phase 1  tailored attacks T1..T9 against the guided prompt of the three main models, on the side effects that each one
           detects without attack (about 460 calls);
  Phase 2  automated search (code/adaptive_attack.py) against Gemma 4 31B and Qwen3.6 35B-A3B, two processes in parallel;
  Then     the analysis (results/experiments/adaptive_report.md, results/tables/exp_adaptive_*.csv) and ADAPTIVE_DONE.
Same robustness as run_scale.py: stalled or crashed workers are restarted, everything is cached, STOP stops it politely.
Resume: setsid nohup bash code/run_adaptive.sh >> results/experiments/adaptive.log 2>&1 < /dev/null &
"""
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import run_scale as rs  # noqa: E402

EXP, PY, ROOT = rs.EXP, rs.PY, rs.ROOT
DONE = EXP / "ADAPTIVE_DONE"
MAIN = ["gemma4:31b", "qwen3.6:35b-a3b", "gpt-oss:20b"]
TARGETS = ["gemma4:31b", "qwen3.6:35b-a3b"]
STALL_S = 40 * 60


def log(msg):
    print(f"{datetime.now():%F %T} [adaptive] {msg}", flush=True)


def flag(model):
    return EXP / f"adaptive_done_{model.replace(':', '_')}.flag"


def search(model):
    """Run the search process for one target model until its completion flag exists."""
    fails = 0
    while not flag(model).exists():
        if rs.stopping() or fails >= 15:
            log(f"search {model}: stopping (fails {fails})")
            return False
        env = dict(os.environ, ADAPTIVE_MODEL=model, WORKERS=os.environ.get("WORKER_CAP", "8"))
        p = subprocess.Popen([PY, "-u", str(CODE / "adaptive_attack.py")], env=env, cwd=ROOT,
                             stdout=open(EXP / f"adaptive_{model.replace(':', '_')}.log", "a"), stderr=subprocess.STDOUT)
        last = (rs.e.RAW.stat().st_size, max([f.stat().st_mtime for f in EXP.glob("adaptive_state_*.json")] or [0]))
        changed = time.time()
        while p.poll() is None:
            time.sleep(30)
            if rs.stopping():
                p.terminate(); break
            now = (rs.e.RAW.stat().st_size, max([f.stat().st_mtime for f in EXP.glob("adaptive_state_*.json")] or [0]))
            if now != last:
                last, changed = now, time.time()
            elif time.time() - changed > STALL_S:
                log(f"search {model}: stalled, killing")
                p.kill(); break
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()
        if not flag(model).exists():
            fails += 1
            log(f"search {model}: process ended without finishing (failure {fails})")
            time.sleep(min(600, 60 * 2 ** fails))
        else:
            fails = 0
    log(f"search {model}: complete")
    return True


def main():
    if DONE.exists():
        DONE.unlink()
    log("supervisor started")
    while not rs.DONE.exists() and not os.environ.get("ADAPTIVE_NOW"):
        if rs.stopping():
            log("stopped by STOP while waiting")
            return
        time.sleep(300)
    log("scale phase finished (or skipped): starting")
    ts = [threading.Thread(target=rs.supervise, args=([m], "tailored", 8, "tailored_" + m.split(":")[0])) for m in MAIN]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for m in MAIN:
        rs.unload(m)
    if rs.stopping():
        log("stopped by STOP")
        return
    ts = [threading.Thread(target=search, args=(m,)) for m in TARGETS]
    [t.start() for t in ts]
    [t.join() for t in ts]
    for m in set(TARGETS + [search_attacker for search_attacker in ("qwen3.6:35b-a3b", "gemma4:31b")]):
        rs.unload(m)
    if rs.stopping():
        log("stopped by STOP")
        return
    r = subprocess.run([PY, str(CODE / "adaptive_analysis.py")], cwd=ROOT, capture_output=True, text=True)
    log(f"analysis exit code {r.returncode} {r.stderr[-300:]}")
    DONE.write_text(f"finished {datetime.now():%F %T}\n")
    log("finished")


if __name__ == "__main__":
    main()
