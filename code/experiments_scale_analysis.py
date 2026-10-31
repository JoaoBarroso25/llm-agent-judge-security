"""Analysis of the scale and reasoning tests (code/run_scale.py), on the attack subset (84 trajectories with a side
effect, 100 without). Population-weighted κ as in the other experiments; Wilson intervals for the share of detected side
effects hidden by an attack; stratified bootstrap (seed 2026) for κ.

Output: results/experiments/scale_report.md, results/tables/exp_scale.csv and exp_think.csv. Does not touch the thesis.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import OUT, TAB, md_table, pct, wilson  # noqa: E402
from experiments import LOCAL_LARGE, LOCAL_HUGE, LOCAL_MODELS, SHORT_NAMES, THINK_MODELS, attack_rows  # noqa: E402
KEYS = ["benchmark", "agent", "task_id"]
from experiments_analysis import wmetrics  # noqa: E402

EXP = OUT / "experiments"
SEED, N_BOOT = 2026, 2000
N_POS_POP, N_NEG_POP = 84, 1217
REF = ["gemma4:31b", "qwen3.6:35b-a3b", "gpt-oss:20b", "gpt-oss:120b"]   # models already in the thesis, for comparison
SHORT = {"qwen3.5:122b": "Qwen3.5 122B-A10B", "nemotron-3-super:120b": "Nemotron 3 Super 120B", "llama4:latest": "Llama 4 109B",
         "qwen3:235b": "Qwen3 235B-A22B", "gemma4:31b": "Gemma 4 31B", "qwen3.6:35b-a3b": "Qwen3.6 35B-A3B",
         "gpt-oss:20b": "gpt-oss 20B", "gpt-oss:120b": "gpt-oss 120B"}


def load():
    s = pd.read_csv(EXP / "sample.csv", dtype={"task_id": str})
    rows = []
    for line in (EXP / "raw.jsonl").read_text(errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    r = pd.DataFrame(rows)
    r = r[r.error.isna()] if "error" in r else r
    r = r[r.cond.str.contains("@", regex=False)].drop_duplicates(["cond"] + KEYS, keep="last")
    r[["base", "model"]] = r.cond.str.split("@", n=1, expand=True)
    r = r[r.model != "qwen3:235b"]   # abandoned: 13 answers in 48 min (too slow on the shared server)
    sub = attack_rows(s)[KEYS + ["side"]].rename(columns={"side": "y"})
    return r.merge(sub, on=KEYS)


def metrics(d):
    y = d.y.astype(int).values
    p = d.side.fillna(0).astype(int).values
    w = np.where(y == 1, N_POS_POP / max((y == 1).sum(), 1), N_NEG_POP / max((y == 0).sum(), 1))
    sens, fa, _, k = wmetrics(y, p, w)
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ks = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ks.append(wmetrics(y[idx], p[idx], w[idx])[3])
    lo, hi = np.nanpercentile(ks, [2.5, 97.5])
    return sens, fa, k, lo, hi


def cell(r, model, base):
    return r[(r.model == model) & (r.base == base)].set_index(KEYS)


def hidden(r, model, ref_base, att_base):
    a, o = cell(r, model, att_base), cell(r, model, ref_base)
    idx = o.index.intersection(a.index)
    det = idx[(o.loc[idx].y == 1) & (o.loc[idx].side == 1)]
    k = int((a.loc[det].side.fillna(0) == 0).sum())
    if not len(det):
        return "—"
    p, lo, hi = wilson(k, len(det))
    return f"{k}/{len(det)} = {pct(p)} [{pct(lo)}; {pct(hi)}]"


def baseline_row(r, model, base):
    d = cell(r, model, base)
    if d.empty:
        return None
    sens, fa, k, lo, hi = metrics(d.reset_index())
    return {"sens": pct(sens), "fa": pct(fa), "kappa": f"{k:.2f} [{lo:.2f}; {hi:.2f}]", "n": len(d),
            "unreadable": pct(d.side.isna().mean()), "seconds": f"{d['seconds'].median():.0f}" if "seconds" in d else "—"}


def scale_table(r):
    rows = []
    for m in LOCAL_LARGE + LOCAL_HUGE + REF:
        o, g = baseline_row(r, m, "O"), baseline_row(r, m, "G")
        if o is None:
            continue
        row = {"Evaluator": SHORT.get(m, m), "Sensitivity": o["sens"], "False alarms": o["fa"], "κ original": o["kappa"],
               "κ guided": g["kappa"] if g else "—", "Unreadable": o["unreadable"], "Median s/call": o["seconds"],
               "Injection hides": hidden(r, m, "O", "A2"), "False authority hides": hidden(r, m, "O", "J2"),
               "Injection hides (guided)": hidden(r, m, "G", "A4") if g else "—"}
        rows.append(row)
    return pd.DataFrame(rows)


def think_table(r):
    rows = []
    for m in THINK_MODELS:
        for off, on, lab in (("O", "OH", "Original prompt"), ("G", "GH", "Guided prompt")):
            a, b = baseline_row(r, m, off), baseline_row(r, m, on)
            if b is None:
                continue
            rows.append({"Evaluator": SHORT.get(m, m), "Prompt": lab, "Sensitivity (off → on)": f"{a['sens']} → {b['sens']}",
                         "False alarms (off → on)": f"{a['fa']} → {b['fa']}", "κ off": a["kappa"], "κ on": b["kappa"],
                         "Unreadable on": b["unreadable"], "Median s on": b["seconds"]})
        a2 = cell(r, m, "A2H")
        if not a2.empty:
            rows.append({"Evaluator": SHORT.get(m, m), "Prompt": "Injection (original)",
                         "Sensitivity (off → on)": "—", "False alarms (off → on)": "—",
                         "κ off": f"hides {hidden(r, m, 'O', 'A2')}", "κ on": f"hides {hidden(r, m, 'OH', 'A2H')}",
                         "Unreadable on": pct(a2.side.isna().mean()), "Median s on": f"{a2['seconds'].median():.0f}"})
    return pd.DataFrame(rows)


# total parameters in billions, from the model metadata served by Ollama (general.parameter_count); MoE = mixture of experts
PARAMS = {"llama3.1:8b": (8.0, False), "ministral-3:14b": (13.9, False), "phi4:14b": (14.7, False),
          "gpt-oss:20b": (20.9, True), "hf.co/mradermacher/EuroLLM-22B-Instruct-2512-i1-GGUF:Q4_K_M": (22.6, False),
          "glm-4.7-flash:latest": (29.9, True), "gemma4:31b": (31.3, False), "nemotron-3-nano:latest": (31.6, True),
          "qwen3.6:35b-a3b": (36.0, True), "llama4:latest": (108.6, True), "gpt-oss:120b": (116.8, True),
          "nemotron-3-super:120b": (123.6, True), "qwen3.5:122b": (125.1, True)}


def size_table(r):
    """kappa with the original and the guided prompt on the attack subset, against the size of every open-weight model."""
    rows = []
    for m in list(LOCAL_MODELS) + LOCAL_LARGE:
        o, g = cell(r, m, "O"), cell(r, m, "G")
        if o.empty or g.empty:
            continue
        ko, go = metrics(o.reset_index()), metrics(g.reset_index())
        rows.append({"model": m, "name": SHORT.get(m, SHORT_NAMES.get(m, m)), "params_b": PARAMS[m][0], "moe": PARAMS[m][1],
                     "kappa_original": round(ko[2], 3), "kappa_original_lo": round(ko[3], 3), "kappa_original_hi": round(ko[4], 3),
                     "kappa_guided": round(go[2], 3), "kappa_guided_lo": round(go[3], 3), "kappa_guided_hi": round(go[4], 3),
                     "injection_hides": hidden(r, m, "O", "A2")})
    return pd.DataFrame(rows).sort_values("params_b")


def main():
    r = load()
    a, b = scale_table(r), think_table(r)
    a.to_csv(TAB / "exp_scale.csv", index=False)
    b.to_csv(TAB / "exp_think.csv", index=False)
    size_table(r).to_csv(TAB / "exp_size.csv", index=False)
    rep = ["# Scale and reasoning tests", "",
           "Attack subset: 84 trajectories with a side effect and 100 without. κ is re-weighted to the population "
           "prevalence (6.5%); 95% stratified bootstrap intervals. 'Hides' is the share of the side effects detected "
           "without the attack that the attack makes the evaluator miss. Rows without a model are models that did not fit in the GPU or were not run.",
           "", "## Larger models, with the models of the thesis for comparison", "", md_table(a) if len(a) else "(no data)",
           "", "## Reasoning switched on", "", md_table(b) if len(b) else "(no data)", ""]
    (EXP / "scale_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
