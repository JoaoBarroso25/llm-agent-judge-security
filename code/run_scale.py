#!/usr/bin/env python3
"""Unattended supervisor for the scale and reasoning tests on the University's Ollama server.

Phase 1  large public open-weight models, one at a time, on the attack subset (O, G, A2, J2, A4):
         qwen3.5:122b, nemotron-3-super:120b, llama4:latest. A model runs only if it fits entirely in the GPU.
Phase 2  reasoning on: qwen3.6:35b-a3b and gemma4:31b with thinking enabled (OH, GH, A2H).
Phase 3  qwen3:235b, only if it fits entirely in the GPU.
Then     the analysis (results/experiments/scale_report.md, results/tables/exp_scale*.csv) and a SCALE_DONE marker.
The thesis files are NOT touched: the results are integrated by hand afterwards.

Robustness: every valid answer is cached in results/experiments/raw.jsonl (a call is never repeated); a stalled or
crashed worker process is killed and restarted; the supervisor retries with backoff; raw.jsonl is copied every hour.
To stop politely (frees the GPU, loses nothing): touch results/experiments/STOP
To resume:  nohup python3 code/run_scale.py >> results/experiments/scale.log 2>&1 &
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))
import experiments as e  # noqa: E402

EXP = ROOT / "results" / "experiments"
LOG, STOP, DONE = EXP / "scale.log", EXP / "STOP", EXP / "SCALE_DONE"
PY = os.environ.get("SCALE_PYTHON", sys.executable)
BASE = e.OLLAMA_URL
KEY = e.OLLAMA_KEY_FILE.read_text().strip()
STALL_S = 25 * 60          # no new answer for this long: restart the worker process
lock = threading.Lock()
procs = []


def log(msg):
    line = f"{datetime.now():%F %T} {msg}"
    with lock:
        print(line, flush=True)


def api(path, body=None, timeout=60):
    req = urllib.request.Request(BASE + path, None if body is None else json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def loaded():
    try:
        return api("/api/ps")["models"]
    except Exception:
        return []


def unload(model):
    try:
        api("/api/generate", {"model": model, "keep_alive": 0})
    except Exception as ex:
        log(f"unload {model}: {str(ex)[:100]}")


def stopping():
    return STOP.exists()


def pending(models, phase):
    done, _ = e.load_raw()
    if phase == "tailored":
        conds = [f"T{i}G@{m}" for m in models for i in range(1, len(e.TAILORED_IDS) + 1)]
    else:
        conds = [f"{t}@{m}" for m in models for t in (e.THINK_ON if phase == "think" else e.SCALE_CONDS)]
    jobs = e.jobs_for(e.make_sample(), conds)
    return sum(1 for j in jobs if j not in done)


def fits(model):
    """Load the model with a real prompt and check that it sits entirely in the GPU. Returns (ok, message)."""
    s = e.make_sample()
    system, user = e.build_prompt(*e.jobs_for(s, ["O@gemma4:31b"])[0])
    body = {"model": model, "stream": False, "think": False, "options": {"num_ctx": 32768, "num_predict": 200},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(3):
        if stopping():
            return False, "stopped"
        try:
            t = time.time()
            r = api("/api/chat", body, timeout=3600)
            mine = [m for m in loaded() if m["name"] == model]
            if not mine:
                return False, "not listed in /api/ps after the call"
            m = mine[0]
            frac = m["size_vram"] / max(m["size"], 1)
            speed = r.get("eval_count", 0) / max(r.get("eval_duration", 1) / 1e9, 1e-9)
            msg = (f"{model}: {m['size']/1e9:.1f} GB, {100*frac:.0f}% in GPU, load {r.get('load_duration',0)/1e9:.0f}s, "
                   f"{speed:.1f} tok/s, first call {time.time()-t:.0f}s")
            return frac >= 0.97, msg
        except Exception as ex:
            log(f"fit check {model}, attempt {attempt+1}: {str(ex)[:150]}")
            time.sleep(120)
    return False, "could not load"


def backup_loop():
    d = EXP / "backups"
    d.mkdir(exist_ok=True)
    while not DONE.exists():
        time.sleep(3600)
        try:
            shutil.copy(e.RAW, d / f"raw_{datetime.now():%m%d_%H%M}.jsonl")
            for old in sorted(d.glob("raw_*.jsonl"))[:-8]:
                old.unlink()
        except Exception as ex:
            log(f"backup: {ex}")


def supervise(models, phase, workers, tag):
    workers = min(workers, int(os.environ.get("WORKER_CAP", "99")))  # lower the pace (noise) without losing anything
    """Run the worker process until nothing is pending, restarting it on stalls and crashes."""
    fails, attempt = 0, 0
    while True:
        left = pending(models, phase)
        if left == 0:
            log(f"[{tag}] complete")
            return True
        if stopping():
            return False
        if fails >= 12:
            log(f"[{tag}] giving up after {fails} attempts without progress ({left} pending)")
            return False
        attempt += 1
        log(f"[{tag}] attempt {attempt}: {left} pending, {workers} workers")
        env = dict(os.environ, SCALE_MODELS=",".join(models), SCALE_PHASE=phase, WORKERS=str(workers))
        p = subprocess.Popen([PY, "-u", str(CODE / "experiments.py"), "run_scale"], env=env, cwd=ROOT,
                             stdout=open(EXP / f"scale_{tag}.log", "a"), stderr=subprocess.STDOUT)
        procs.append(p)
        last_size, last_change = e.RAW.stat().st_size, time.time()
        while p.poll() is None:
            time.sleep(30)
            if stopping():
                p.terminate()
                break
            size = e.RAW.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.time()
            elif time.time() - last_change > STALL_S:
                log(f"[{tag}] stalled for {STALL_S//60} min: killing the worker")
                p.kill()
                break
        try:
            p.wait(timeout=60)
        except Exception:
            p.kill()
        now = pending(models, phase)
        fails = 0 if now < left else fails + 1
        log(f"[{tag}] worker ended, {now} pending (was {left}), consecutive failures {fails}")
        if now:
            time.sleep(min(900, 60 * 2 ** fails))


def run_large(model):
    ok, msg = fits(model)
    log(f"FIT {msg} -> {'RUN' if ok else 'SKIP'}")
    if ok:
        supervise([model], "large", 8, model.split(":")[0].replace("/", "_"))
    unload(model)
    time.sleep(10)


def thinking_supported(model):
    try:
        return "thinking" in (api("/api/show", {"name": model}).get("capabilities") or [])
    except Exception as ex:
        log(f"show {model}: {str(ex)[:100]}")
        return False


def main():
    EXP.mkdir(exist_ok=True)
    if STOP.exists():
        STOP.unlink()
    if DONE.exists():
        DONE.unlink()
    log("=== scale supervisor started")
    shutil.copy(e.RAW, EXP / f"raw_backup_scale_{datetime.now():%m%d_%H%M}.jsonl")
    threading.Thread(target=backup_loop, daemon=True).start()
    for m in e.LOCAL_LARGE:
        if stopping():
            break
        run_large(m)
    if not stopping():
        models = [m for m in e.THINK_MODELS if thinking_supported(m)]
        log(f"thinking phase on {models}")
        ts = [threading.Thread(target=supervise, args=([m], "think", 8, "think_" + m.split(":")[0])) for m in models]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for m in models:
            unload(m)
    if not stopping():
        for m in e.LOCAL_HUGE:
            run_large(m)
    if stopping():
        log("=== stopped by STOP; everything done so far is saved. Resume with: python3 code/run_scale.py")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        return
    r = subprocess.run([PY, str(CODE / "experiments_scale_analysis.py")], cwd=ROOT, capture_output=True, text=True)
    log("analysis exit code " + str(r.returncode) + " " + r.stderr[-300:])
    DONE.write_text(f"finished {datetime.now():%F %T}\n")
    log("=== finished")


if __name__ == "__main__":
    main()
