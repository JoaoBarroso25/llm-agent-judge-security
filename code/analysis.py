"""Security-focused secondary analysis of AgentRewardBench.

Reads the expert annotations and the published verdicts of the automated
evaluators (judges) of AgentRewardBench (Lù et al., 2025) and answers the four
research questions of the dissertation. It never calls a language model: the
verdicts are those published by the authors of the benchmark; the questions,
the derived variables and the statistics are the contribution of this work.

  RQ1  Prevalence of unsafe behaviour (side effects, loops) and what it is
       associated with (agent, environment, length, success).
  RQ2  Reliability of automated evaluators as security monitors
       (sensitivity, false alarms, kappa, AC1, human ceiling, modality).
  RQ3  Temporal analysis: occurrence and detection versus trajectory length.
  RQ4  Robustness of the oversight pipeline: conflation of success, loops and
       side effects in the evaluator; shared errors and committees.

Usage:   python3 code/analysis.py
Input:   data/agent-reward-bench/{data,judgments}  (not versioned)
Output:  results/tables/*.csv and results/analysis_report.md
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest, chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "agent-reward-bench"
OUT = ROOT / "results"
TAB = OUT / "tables"
SEED = 2026
N_BOOT = 2000
RNG = np.random.default_rng(SEED)

BENCHMARKS = {
    "assistantbench": "AssistantBench",
    "visualwebarena": "VisualWebArena",
    "webarena": "WebArena",
    "workarena": "WorkArena",
}
AGENTS = {
    "GenericAgent-anthropic_claude-3.7-sonnet": "Claude 3.7 Sonnet",
    "GenericAgent-gpt-4o-2024-11-20": "GPT-4o",
    "GenericAgent-meta-llama_Llama-3.3-70B-Instruct": "Llama 3.3 70B",
    "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL 72B",
}
JUDGES = {
    "functional": "Benchmark rule",
    "aer": "AER-C",
    "aerv": "AER-V",
    "nnetnav": "NNetNav",
    "claude-3.7-sonnet-noscreen": "Claude 3.7 Sonnet (AXTree)",
    "claude-3.7-sonnet-noaxtree": "Claude 3.7 Sonnet (screenshot)",
    "gpt-4o-noscreen": "GPT-4o (AXTree)",
    "gpt-4o-noaxtree": "GPT-4o (screenshot)",
    "gpt-4o-mini": "GPT-4o mini (AXTree + screenshot)",
    "gpt-4o-mini-noscreen": "GPT-4o mini (AXTree)",
    "gpt-4o-mini-noaxtree": "GPT-4o mini (screenshot)",
    "gpt-4o-mini-noscreen-noaxtree": "GPT-4o mini (actions only)",
    "qwen-2.5-vl-noscreen": "Qwen2.5-VL (AXTree)",
    "qwen-2.5-vl-noaxtree": "Qwen2.5-VL (screenshot)",
    "llama-3.3-70b-noscreen": "Llama 3.3 70B (AXTree)",
}
NON_LLM = {"functional", "aer", "aerv", "nnetnav"}
DIMS = {"trajectory_success": "success", "trajectory_side_effect": "side",
        "trajectory_looping": "loop"}
MODALITY_PAIRS = [("claude-3.7-sonnet-noscreen", "claude-3.7-sonnet-noaxtree"),
                  ("gpt-4o-noscreen", "gpt-4o-noaxtree"),
                  ("gpt-4o-mini-noscreen", "gpt-4o-mini-noaxtree"),
                  ("qwen-2.5-vl-noscreen", "qwen-2.5-vl-noaxtree")]


# --------------------------------------------------------------------------- #
# Small statistical helpers
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def kappa(y, p):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    if n == 0:
        return math.nan
    po = np.mean(y == p)
    py, pp = y.mean(), p.mean()
    pe = py * pp + (1 - py) * (1 - pp)
    return math.nan if pe == 1 else (po - pe) / (1 - pe)


def ac1(y, p):
    """Gwet's AC1 for two raters and a binary category."""
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return math.nan
    po = np.mean(y == p)
    q = (y.mean() + p.mean()) / 2
    pe = 2 * q * (1 - q)
    return math.nan if pe == 1 else (po - pe) / (1 - pe)


def boot_ci(y, p, fn, n_boot=N_BOOT):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        i = RNG.integers(0, n, n)
        v = fn(y[i], p[i])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (math.nan, math.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def krippendorff_nominal(matrix: np.ndarray) -> float:
    """Krippendorff's alpha for nominal data. matrix: units x raters, NaN = missing."""
    values = np.unique(matrix[~np.isnan(matrix)])
    idx = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for row in matrix:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        for a, b in itertools.permutations(range(m), 2):
            o[idx[r[a]], idx[r[b]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    do = n - np.trace(o)
    de = (n * n - (n_c ** 2).sum()) / (n - 1)
    return 1 - do / de if de else math.nan


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    n = len(y)
    return {
        "n": n, "positives": tp + fn, "prevalence": (tp + fn) / n if n else math.nan,
        "flag_rate": (tp + fp) / n if n else math.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
        "false_alarm": fp / (fp + tn) if fp + tn else math.nan,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "accuracy": (tp + tn) / n if n else math.nan,
        "kappa": kappa(y, p), "ac1": ac1(y, p),
    }


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{d}f}"


def pct(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def label(value):
    """Expert or judge label -> 1, 0 or None (unreadable / unsure)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    r = str(value).strip().lower().strip("'\"")
    if r in ("successful", "yes", "success"):
        return 1
    if r in ("unsuccessful", "no", "failure", "unsuccesful", "partially successful"):
        return 0
    return None


def tag(name, text):
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def read_annotations():
    a = pd.read_csv(DATA / "data" / "annotations.csv")
    a["key"] = list(zip(a.benchmark, a.model_name, a.task_id.astype(str)))
    a["order"] = a.groupby("key").cumcount()
    prim = a[a.order == 0].copy()
    sec = a[a.order == 1].copy()
    for df in (prim, sec):
        for col, dim in DIMS.items():
            df[dim] = df[col].map(label)
    return prim.set_index("key"), sec.set_index("key")


def judgment_path(bench, agent, judge, task):
    return DATA / "judgments" / bench / agent / judge / f"{task}.json"


def judge_verdicts(judge, j):
    """Return {success, side, loop}: 1/0, None if unreadable, 'na' if not asked."""
    out = {"success": "na", "side": "na", "loop": "na"}
    if judge == "functional":
        out["success"] = int(j["trajectory_info"]["summary_info"]["cum_reward"] > 0.5)
        return out
    resp = j.get("response") or {}
    choices = resp.get("choices") or []
    msg = (choices[0]["message"]["content"] or "") if choices else ""
    if judge in ("aer", "aerv"):
        m = re.search(r"Status:\s*\"?(\w+)", msg)
        out["success"] = label(m.group(1)) if m else None
        return out
    if judge == "nnetnav":
        m = re.search(r"Reward:\s*(\d)", msg)
        out["success"] = (1 if int(m.group(1)) >= 4 else 0) if m else None
        return out
    for dim in ("success", "side", "loop"):
        out[dim] = label(tag(dim, msg))
    return out


STEP_RE = re.compile(r"Step: (\d+)\nURL: (.*?)\nAction: (.*?)\nReasoning: (.*?)(?=-----)", re.S)


def trajectory_features(j):
    """Features of the trajectory, parsed from the prompt given to a simplified judge."""
    content = ""
    for m in j["chat_messages"]["regular"]:
        if m["role"] != "user":
            continue
        c = m["content"]
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        content += c
    steps = STEP_RE.findall(content + "-----")
    actions = [s[2].strip() for s in steps if s[2].strip() not in ("None", "")]
    final = actions[-1] if actions else ""
    final_type = final.split("(")[0] if final else "none"
    urls = [s[1].strip() for s in steps]
    info = j["trajectory_info"]["summary_info"]
    return {
        "n_steps": int(info.get("n_steps") or len(actions)),
        "n_actions_parsed": len(actions),
        "repeated_actions": len(actions) - len(set(actions)),
        "distinct_urls": len(set(urls)),
        "final_action": final_type,
        "reasoning_chars": sum(len(s[3]) for s in steps),
        "goal": (j.get("goal") or "").strip(),
    }


def build():
    prim, sec = read_annotations()
    rows, trajs = [], {}
    for key, a in prim.iterrows():
        bench, agent, task = key
        jdir = DATA / "judgments" / bench / agent
        if not jdir.exists():
            continue
        for judge in JUDGES:
            path = judgment_path(bench, agent, judge, task)
            if not path.exists():
                continue
            j = json.loads(path.read_text())
            if key not in trajs and judge not in NON_LLM:
                trajs[key] = trajectory_features(j)
            v = judge_verdicts(judge, j)
            rows.append({"benchmark": bench, "agent": agent, "task_id": task, "judge": judge,
                         **{f"j_{d}": v[d] for d in ("success", "side", "loop")}})
    pairs = pd.DataFrame(rows)
    feats = pd.DataFrame.from_dict(trajs, orient="index")
    feats.index = pd.MultiIndex.from_tuples(feats.index, names=["benchmark", "agent", "task_id"])
    labels = prim[["success", "side", "loop", "annotator_name"]].copy()
    labels.index = pd.MultiIndex.from_tuples(labels.index, names=["benchmark", "agent", "task_id"])
    traj = labels.join(feats, how="inner").reset_index()
    traj["bench_name"] = traj.benchmark.map(BENCHMARKS)
    traj["agent_name"] = traj.agent.map(AGENTS)
    traj["log_steps"] = np.log1p(traj.n_steps)
    traj["at_limit"] = (traj.n_steps >= 30).astype(int)
    pairs = pairs.merge(traj, on=["benchmark", "agent", "task_id"], how="inner")
    sec_labels = sec[["success", "side", "loop"]].copy()
    sec_labels.index = pd.MultiIndex.from_tuples(sec_labels.index, names=["benchmark", "agent", "task_id"])
    return traj, pairs, sec_labels.reset_index()


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def rq1(traj, rep):
    rep.append("\n## RQ1 Prevalence of unsafe behaviour\n")
    rep.append(f"Trajectories with a primary expert annotation and a published judgment: **{len(traj)}**.\n")
    rows = []
    for (b, g), d in list(traj.groupby(["bench_name", "agent_name"])) + [(("All", "All"), traj)]:
        r = {"benchmark": b, "agent": g, "n": len(d)}
        for dim in ("success", "side", "loop"):
            s = d[dim].dropna()
            p, lo, hi = wilson(int(s.sum()), len(s))
            r[f"{dim}_n"] = int(s.sum())
            r[f"{dim}_rate"] = p
            r[f"{dim}_lo"], r[f"{dim}_hi"] = lo, hi
        r["mean_steps"] = d.n_steps.mean()
        r["at_limit"] = d.at_limit.mean()
        rows.append(r)
    prev = pd.DataFrame(rows)
    prev.to_csv(TAB / "rq1_prevalence.csv", index=False)
    show = prev.assign(
        success=[f"{int(a)} ({pct(b)})" for a, b in zip(prev.success_n, prev.success_rate)],
        side_effects=[f"{int(a)} ({pct(b)} [{pct(c)}; {pct(d)}])" for a, b, c, d in
                      zip(prev.side_n, prev.side_rate, prev.side_lo, prev.side_hi)],
        loops=[f"{int(a)} ({pct(b)})" for a, b in zip(prev.loop_n, prev.loop_rate)],
        mean_steps=[fmt(x, 1) for x in prev.mean_steps],
        at_30_step_limit=[pct(x) for x in prev.at_limit],
    )[["benchmark", "agent", "n", "success", "side_effects", "loops", "mean_steps", "at_30_step_limit"]]
    rep.append("\n### By benchmark and agent (expert labels; Wilson 95% CI for side effects)\n")
    rep.append(md_table(show))

    for by, name in (("bench_name", "benchmark"), ("agent_name", "agent")):
        rows = []
        for g, d in traj.groupby(by):
            r = {name: g, "n": len(d)}
            for dim in ("side", "loop", "success"):
                s = d[dim].dropna()
                p, lo, hi = wilson(int(s.sum()), len(s))
                r[dim] = f"{int(s.sum())} ({pct(p)} [{pct(lo)}; {pct(hi)}])"
            rows.append(r)
        rep.append(f"\n### By {name}\n")
        rep.append(md_table(pd.DataFrame(rows)))
        pd.DataFrame(rows).to_csv(TAB / f"rq1_by_{name}.csv", index=False)

    # Association with success and loops
    d = traj.dropna(subset=["side", "success", "loop"])
    ct = pd.crosstab(d.success, d.side)
    chi = chi2_contingency(ct)
    rep.append("\n### Side effects versus task success (expert labels)\n")
    for s in (0, 1):
        k = int(d[(d.success == s)].side.sum())
        n = int((d.success == s).sum())
        p, lo, hi = wilson(k, n)
        rep.append(f"- success = {s}: {k}/{n} with side effects = {pct(p)} [{pct(lo)}; {pct(hi)}]\n")
    rep.append(f"- χ² = {chi[0]:.2f}, p = {chi[1]:.3g}\n")
    rep.append("\n### Side effects versus loops (expert labels)\n")
    for s in (0, 1):
        k = int(d[(d.loop == s)].side.sum())
        n = int((d.loop == s).sum())
        p, lo, hi = wilson(k, n)
        rep.append(f"- loop = {s}: {k}/{n} with side effects = {pct(p)} [{pct(lo)}; {pct(hi)}]\n")
    ct2 = pd.crosstab(d.loop, d.side)
    chi2 = chi2_contingency(ct2)
    rep.append(f"- χ² = {chi2[0]:.2f}, p = {chi2[1]:.3g}\n")
    succ_side = d[d.success == 1]
    rep.append(f"- Successful trajectories that still caused side effects: {int(succ_side.side.sum())} of "
               f"{len(succ_side)} ({pct(succ_side.side.mean())})\n")

    # Regression of side effects on length, agent and benchmark
    rep.append("\n### Logistic regression: side effect ~ log(steps) + success + agent + benchmark\n")
    m, f = safe_logit(["side ~ log_steps + success + C(agent_name) + C(bench_name)",
                       "side ~ log_steps + success + C(bench_name)", "side ~ log_steps + success"], d)
    rep.append(f"Specification: `{f}`\n\n")
    rep.append(regression_table(m, ["log_steps", "success"]))
    return prev



def safe_logit(formulas, data, groups=None):
    """Fit the first formula that converges; return (model, formula)."""
    last = None
    for f in formulas:
        try:
            mod = smf.logit(f, data=data)
            if groups is not None:
                m = mod.fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
            else:
                m = mod.fit(disp=0, maxiter=200, cov_type="HC1")
            if np.all(np.isfinite(m.bse)):
                return m, f
        except Exception as e:  # singular matrix, perfect separation
            last = e
    raise RuntimeError(f"no specification converged: {last}")


def regression_table(m, keep=None, store=None):
    rows = []
    ci = m.conf_int()
    for name in m.params.index:
        if keep and not any(name.startswith(k) for k in keep):
            continue
        rows.append({"variable": name, "OR": fmt(math.exp(m.params[name])),
                     "95% CI": f"{fmt(math.exp(ci.loc[name, 0]))} to {fmt(math.exp(ci.loc[name, 1]))}",
                     "p": f"{m.pvalues[name]:.3g}"})
    if store is not None:
        store.extend(rows)
    return md_table(pd.DataFrame(rows)) + f"\nn = {int(m.nobs)}\n"


def ceiling(traj, sec, rep):
    rep.append("\n## Human ceiling (doubly annotated trajectories)\n")
    both = traj.merge(sec, on=["benchmark", "agent", "task_id"], suffixes=("", "_2"))
    rows = []
    for dim in ("success", "side", "loop"):
        d = both.dropna(subset=[dim, f"{dim}_2"])
        y, p = d[dim].astype(int).values, d[f"{dim}_2"].astype(int).values
        k_lo, k_hi = boot_ci(y, p, kappa)
        rows.append({"dimension": dim, "n": len(d), "positives (1st)": int(y.sum()),
                     "positives (2nd)": int(p.sum()), "raw agreement": pct(np.mean(y == p)),
                     "kappa": fmt(kappa(y, p)), "kappa 95% CI": f"{fmt(k_lo)} to {fmt(k_hi)}",
                     "AC1": fmt(ac1(y, p))})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "human_ceiling.csv", index=False)
    rep.append(f"Doubly annotated trajectories: {len(both)} "
               f"({', '.join(f'{k}: {v}' for k, v in both.bench_name.value_counts().items())}).\n\n")
    rep.append(md_table(df))
    return df


def rq2(pairs, rep):
    rep.append("\n## RQ2 Automated evaluators as security monitors\n")
    llm = pairs[~pairs.judge.isin(NON_LLM)]
    rows = []
    for judge, d in llm.groupby("judge"):
        for dim in ("success", "side", "loop"):
            dd = d.dropna(subset=[dim])
            y = dd[dim].astype(int).values
            v = dd[f"j_{dim}"]
            unread = int(v.isna().sum())
            pr = v.fillna(0).astype(int).values      # unreadable verdict counts as negative
            m = binary_metrics(y, pr)
            lo, hi = boot_ci(y, pr, kappa)
            rows.append({"judge": judge, "judge_name": JUDGES[judge], "dimension": dim,
                         "unreadable": unread, **m, "kappa_lo": lo, "kappa_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "rq2_judges.csv", index=False)
    for dim in ("side", "loop", "success"):
        d = res[res.dimension == dim].sort_values("kappa", ascending=False)
        show = pd.DataFrame({
            "Judge": d.judge_name, "n": d.n, "positives": d.positives,
            "flag rate": [pct(x) for x in d.flag_rate],
            "sensitivity": [pct(x) for x in d.sensitivity],
            "false alarms": [pct(x) for x in d.false_alarm],
            "precision": [pct(x) for x in d.precision],
            "κ [95% CI]": [f"{fmt(k)} [{fmt(a)}; {fmt(b)}]" for k, a, b in zip(d.kappa, d.kappa_lo, d.kappa_hi)],
            "AC1": [fmt(x) for x in d.ac1], "unreadable": d.unreadable,
        })
        rep.append(f"\n### Dimension: {dim} (11 simplified judges, all benchmarks)\n")
        rep.append(md_table(show))
    # summary comparing dimensions per judge
    piv = res.pivot(index="judge_name", columns="dimension", values="kappa")
    rep.append("\n### κ by dimension, same judge\n")
    rep.append(md_table(piv.reset_index().rename(columns={"judge_name": "Judge"}).round(2)))
    rep.append(f"\nMean κ across judges: success {fmt(piv['success'].mean())}, "
               f"side {fmt(piv['side'].mean())}, loop {fmt(piv['loop'].mean())}.\n")

    # by benchmark (side effects)
    rows = []
    for (b, judge), d in llm.groupby(["bench_name", "judge"]):
        dd = d.dropna(subset=["side"])
        y = dd.side.astype(int).values
        pr = dd.j_side.fillna(0).astype(int).values
        m = binary_metrics(y, pr)
        rows.append({"benchmark": b, "judge": JUDGES[judge], **m})
    byb = pd.DataFrame(rows)
    byb.to_csv(TAB / "rq2_side_by_benchmark.csv", index=False)
    agg = byb.groupby("benchmark").agg(positives=("positives", "first"), n=("n", "first"),
                                       sens_mean=("sensitivity", "mean"), fa_mean=("false_alarm", "mean"),
                                       kappa_mean=("kappa", "mean"), kappa_max=("kappa", "max")).reset_index()
    rep.append("\n### Side effects by benchmark (mean over simplified judges)\n")
    rep.append(md_table(agg.assign(sens_mean=[pct(x) for x in agg.sens_mean],
                                   fa_mean=[pct(x) for x in agg.fa_mean],
                                   kappa_mean=[fmt(x) for x in agg.kappa_mean],
                                   kappa_max=[fmt(x) for x in agg.kappa_max])))

    # modality: McNemar on correctness, side and loop
    rows = []
    for a, b in MODALITY_PAIRS:
        for dim in ("side", "loop", "success"):
            da = llm[llm.judge == a][["benchmark", "agent", "task_id", dim, f"j_{dim}"]]
            db = llm[llm.judge == b][["benchmark", "agent", "task_id", f"j_{dim}"]]
            m = da.merge(db, on=["benchmark", "agent", "task_id"], suffixes=("_a", "_b")).dropna(subset=[dim])
            ca = (m[f"j_{dim}_a"].fillna(0) == m[dim])
            cb = (m[f"j_{dim}_b"].fillna(0) == m[dim])
            only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            rows.append({"model": JUDGES[a].split(" (")[0], "dimension": dim, "n": len(m),
                         "acc_axtree": ca.mean(), "acc_screenshot": cb.mean(),
                         "only_axtree_correct": only_a, "only_screenshot_correct": only_b, "p": p})
    mod = pd.DataFrame(rows)
    mod["p_holm"] = holm(mod.p.values)
    mod.to_csv(TAB / "rq2_modality.csv", index=False)
    rep.append("\n### Observation modality (exact McNemar, Holm over all 12 tests)\n")
    rep.append(md_table(mod.assign(acc_axtree=[pct(x) for x in mod.acc_axtree],
                                   acc_screenshot=[pct(x) for x in mod.acc_screenshot],
                                   p=[f"{x:.3f}" for x in mod.p], p_holm=[f"{x:.3f}" for x in mod.p_holm])))
    return res


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


STEP_BINS = [0, 5, 10, 20, 29, 30]
STEP_LABELS = ["1-5", "6-10", "11-20", "21-29", "30 (limit)"]


def rq3(traj, pairs, rep):
    rep.append("\n## RQ3 Length of the trajectory\n")
    traj = traj.copy()
    traj["step_bin"] = pd.cut(traj.n_steps, STEP_BINS, labels=STEP_LABELS, include_lowest=True)
    rows = []
    for g, d in traj.groupby("step_bin", observed=True):
        r = {"steps": g, "n": len(d)}
        for dim in ("side", "loop", "success"):
            s = d[dim].dropna()
            p, lo, hi = wilson(int(s.sum()), len(s))
            r[dim] = f"{pct(p)} [{pct(lo)}; {pct(hi)}]"
            r[f"{dim}_rate"] = p
        rows.append(r)
    occ = pd.DataFrame(rows)
    occ.to_csv(TAB / "rq3_occurrence_by_steps.csv", index=False)
    rep.append("\n### Occurrence (expert labels) by number of steps\n")
    rep.append(md_table(occ[["steps", "n", "side", "loop", "success"]]))

    llm = pairs[~pairs.judge.isin(NON_LLM)].copy()
    llm["step_bin"] = pd.cut(llm.n_steps, STEP_BINS, labels=STEP_LABELS, include_lowest=True)
    rows = []
    for g, d in llm.groupby("step_bin", observed=True):
        r = {"steps": g}
        for dim in ("side", "loop"):
            dd = d.dropna(subset=[dim])
            pos = dd[dd[dim] == 1]
            neg = dd[dd[dim] == 0]
            s, slo, shi = wilson(int(pos[f"j_{dim}"].fillna(0).sum()), len(pos))
            f, flo, fhi = wilson(int(neg[f"j_{dim}"].fillna(0).sum()), len(neg))
            r[f"{dim}_sensitivity"] = s
            r[f"{dim}_false_alarm"] = f
            r[f"{dim}_pos_pairs"] = len(pos)
            r[f"{dim}_neg_pairs"] = len(neg)
        rows.append(r)
    det = pd.DataFrame(rows)
    det.to_csv(TAB / "rq3_detection_by_steps.csv", index=False)
    rep.append("\n### Detection by number of steps (judge × trajectory pairs, 11 judges)\n")
    rep.append(md_table(det.assign(**{c: [pct(x) for x in det[c]] for c in det.columns
                                      if c.endswith(("sensitivity", "false_alarm"))})))

    rep.append("\n### Regressions (clustered by trajectory)\n")
    llm["traj_id"] = llm.benchmark + "|" + llm.agent + "|" + llm.task_id
    store = []
    for dim in ("side", "loop"):
        for lab, sub in (("misses on expert positives (verdict=1 means detected)", 1),
                         ("false alarms on expert negatives", 0)):
            d = llm[llm[dim] == sub].dropna(subset=[f"j_{dim}"]).copy()
            d["y"] = d[f"j_{dim}"].astype(int)
            if d.y.nunique() < 2:
                continue
            m, f = safe_logit(["y ~ log_steps + C(judge) + C(agent_name) + C(bench_name)",
                               "y ~ log_steps + C(judge) + C(bench_name)", "y ~ log_steps + C(judge)"],
                              d, pd.factorize(d.traj_id)[0])
            rep.append(f"\n**{dim}: {lab}** (`{f}`)\n\n")
            rep.append(regression_table(m, ["log_steps"], store))
    # occurrence regression
    for dim in ("side", "loop"):
        d = traj.dropna(subset=[dim, "success"])
        m, f = safe_logit([f"{dim} ~ log_steps + success + C(agent_name) + C(bench_name)",
                           f"{dim} ~ log_steps + success + C(bench_name)"], d)
        rep.append(f"\n**Occurrence of {dim} (expert label)** (`{f}`)\n\n")
        rep.append(regression_table(m, ["log_steps", "success"]))
    return occ, det


def rq4(traj, pairs, rep):
    rep.append("\n## RQ4 Robustness of the oversight pipeline\n")
    llm = pairs[~pairs.judge.isin(NON_LLM)].copy()
    llm["traj_id"] = llm.benchmark + "|" + llm.agent + "|" + llm.task_id

    # (a) Conflation: false alarms on side effects by the judge's own success verdict and by expert loop
    rep.append("\n### (a) Conflation of dimensions in the evaluator\n")
    neg = llm[(llm.side == 0)].dropna(subset=["j_side", "j_success", "loop", "success"]).copy()
    neg["fa"] = neg.j_side.astype(int)
    rows = []
    for cond, d in (("judge says success", neg[neg.j_success == 1]),
                    ("judge says failure", neg[neg.j_success == 0]),
                    ("expert: loop", neg[neg.loop == 1]),
                    ("expert: no loop", neg[neg.loop == 0]),
                    ("judge says loop", neg[neg.j_loop == 1]),
                    ("judge says no loop", neg[neg.j_loop == 0])):
        p, lo, hi = wilson(int(d.fa.sum()), len(d))
        rows.append({"condition (expert: no side effect)": cond, "pairs": len(d),
                     "false alarm on side effects": f"{pct(p)} [{pct(lo)}; {pct(hi)}]", "rate": p})
    fa = pd.DataFrame(rows)
    fa.to_csv(TAB / "rq4_false_alarms_by_condition.csv", index=False)
    rep.append(md_table(fa.drop(columns="rate")))

    pos = llm[(llm.side == 1)].dropna(subset=["j_side", "j_success", "loop", "success"]).copy()
    pos["det"] = pos.j_side.astype(int)
    rows = []
    for cond, d in (("judge says success", pos[pos.j_success == 1]),
                    ("judge says failure", pos[pos.j_success == 0]),
                    ("expert: task successful", pos[pos.success == 1]),
                    ("expert: task failed", pos[pos.success == 0]),
                    ("expert: loop", pos[pos.loop == 1]),
                    ("expert: no loop", pos[pos.loop == 0])):
        p, lo, hi = wilson(int(d.det.sum()), len(d))
        rows.append({"condition (expert: side effect)": cond, "pairs": len(d),
                     "sensitivity": f"{pct(p)} [{pct(lo)}; {pct(hi)}]", "rate": p})
    se = pd.DataFrame(rows)
    se.to_csv(TAB / "rq4_sensitivity_by_condition.csv", index=False)
    rep.append("\n")
    rep.append(md_table(se.drop(columns="rate")))

    rep.append("\n**Regression of the side-effect verdict on expert negatives (false alarms), clustered by trajectory**\n\n")
    store = []
    m, f = safe_logit(["fa ~ j_success + success + loop + log_steps + C(judge) + C(agent_name) + C(bench_name)",
                       "fa ~ j_success + success + loop + log_steps + C(judge)"], neg, pd.factorize(neg.traj_id)[0])
    rep.append(f"Specification: `{f}`\n\n")
    rep.append(regression_table(m, ["j_success", "success", "loop", "log_steps"], store))
    pd.DataFrame(store).to_csv(TAB / "rq4_regression_false_alarms.csv", index=False)
    rep.append("\n**Regression of the side-effect verdict on expert positives (detection), clustered by trajectory**\n\n")
    store = []
    m, f = safe_logit(["det ~ j_success + success + loop + log_steps + C(judge)",
                       "det ~ j_success + success + loop + log_steps"], pos, pd.factorize(pos.traj_id)[0])
    rep.append(f"Specification: `{f}`\n\n")
    rep.append(regression_table(m, ["j_success", "success", "loop", "log_steps"], store))
    pd.DataFrame(store).to_csv(TAB / "rq4_regression_detection.csv", index=False)

    # final action of the agent
    rows = []
    for fa_type, d in neg.groupby("final_action"):
        if len(d) < 100:
            continue
        p, lo, hi = wilson(int(d.fa.sum()), len(d))
        rows.append({"final action": fa_type, "pairs": len(d), "false alarm": f"{pct(p)} [{pct(lo)}; {pct(hi)}]"})
    rep.append("\n**False alarms on side effects by the agent's final action (expert negatives)**\n\n")
    rep.append(md_table(pd.DataFrame(rows)))

    # (b) Shared errors and committees
    rep.append("\n### (b) Shared errors and committees of evaluators\n")
    judges_all = sorted(llm.judge.unique())
    coverage = llm.groupby("judge").traj_id.nunique()
    rep.append("Trajectories per judge: " + ", ".join(f"{JUDGES[j]} {coverage[j]}" for j in judges_all) + "\n\n")
    # judges available on every trajectory where any judge exists
    n_traj = llm.traj_id.nunique()
    common = [j for j in judges_all if coverage[j] == n_traj]
    rep.append(f"Judges available for all {n_traj} trajectories: {len(common)} "
               f"({', '.join(JUDGES[j] for j in common)}).\n")
    committee_rows, shared_rows = [], []
    for dim in ("side", "loop", "success"):
        w = llm.pivot_table(index="traj_id", columns="judge", values=f"j_{dim}", aggfunc="first")
        w = w.astype(float).fillna(0)  # unreadable = negative
        lab = llm.drop_duplicates("traj_id").set_index("traj_id")[dim]
        lab = lab.loc[w.index].dropna()
        w = w.loc[lab.index]
        y = lab.astype(int).values
        # pairwise kappas judge-judge vs judge-human (on the common judges)
        cols = [c for c in w.columns if c in common] or list(w.columns)
        jj = [kappa(w[a].values, w[b].values) for a, b in itertools.combinations(cols, 2)]
        jh = [kappa(y, w[a].values) for a in cols]
        alpha_j = krippendorff_nominal(w[cols].values)
        alpha_jh = krippendorff_nominal(np.column_stack([w[cols].values, y]))
        npos = int(y.sum())
        detected_by = w[cols].values[y == 1].sum(axis=1)
        alarm_by = w[cols].values[y == 0].sum(axis=1)
        shared_rows.append({
            "dimension": dim, "judges": len(cols), "trajectories": len(y), "positives": npos,
            "mean κ judge–judge": fmt(np.nanmean(jj)), "range judge–judge": f"{fmt(np.nanmin(jj))} to {fmt(np.nanmax(jj))}",
            "mean κ judge–human": fmt(np.nanmean(jh)), "range judge–human": f"{fmt(np.nanmin(jh))} to {fmt(np.nanmax(jh))}",
            "α judges": fmt(alpha_j), "α judges + human": fmt(alpha_jh),
            "positives missed by all judges": int((detected_by == 0).sum()),
            "positives caught by all judges": int((detected_by == len(cols)).sum()),
            "negatives flagged by ≥ half of judges": int((alarm_by >= len(cols) / 2).sum()),
        })
        # committees
        votes = w[cols].values
        k = len(cols)
        best = max(cols, key=lambda c: kappa(y, w[c].values))
        rules = {f"best single judge ({JUDGES[best]})": w[best].values,
                 "majority vote (all)": (votes.sum(axis=1) > k / 2).astype(int),
                 "any judge flags (OR)": (votes.sum(axis=1) >= 1).astype(int),
                 "all judges flag (AND)": (votes.sum(axis=1) == k).astype(int)}
        cross = [c for c in ("claude-3.7-sonnet-noscreen", "gpt-4o-noscreen", "qwen-2.5-vl-noscreen") if c in cols]
        if len(cross) == 3:
            rules["majority of 3 providers (Claude, GPT-4o, Qwen; AXTree)"] = (w[cross].sum(axis=1) >= 2).astype(int).values
        for rname, pr in rules.items():
            m = binary_metrics(y, pr)
            lo, hi = boot_ci(y, pr, kappa)
            committee_rows.append({"dimension": dim, "rule": rname, "sensitivity": pct(m["sensitivity"]),
                                   "false alarms": pct(m["false_alarm"]), "precision": pct(m["precision"]),
                                   "κ [95% CI]": f"{fmt(m['kappa'])} [{fmt(lo)}; {fmt(hi)}]", "AC1": fmt(m["ac1"]),
                                   "kappa": m["kappa"]})
    sh = pd.DataFrame(shared_rows)
    sh.to_csv(TAB / "rq4_shared_errors.csv", index=False)
    rep.append(md_table(sh))
    cm = pd.DataFrame(committee_rows)
    cm.to_csv(TAB / "rq4_committees.csv", index=False)
    rep.append("\n**Committees**\n\n")
    rep.append(md_table(cm.drop(columns="kappa")))

    # side-effect positives: distribution of number of judges that detect them
    w = llm.pivot_table(index="traj_id", columns="judge", values="j_side", aggfunc="first").astype(float).fillna(0)
    lab = llm.drop_duplicates("traj_id").set_index("traj_id")
    pos_ids = lab.index[(lab.side == 1)]
    cols = [c for c in w.columns if c in common] or list(w.columns)
    counts = w.loc[w.index.intersection(pos_ids), cols].sum(axis=1).value_counts().sort_index()
    counts.to_csv(TAB / "rq4_side_positives_detected_by_k.csv")
    rep.append("\n**Number of judges (of " + str(len(cols)) + ") that detect each expert-annotated side effect**\n\n")
    rep.append(md_table(pd.DataFrame({"judges detecting": counts.index.astype(int), "trajectories": counts.values})))
    return fa, se, sh, cm


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    traj, pairs, sec = build()
    traj.to_csv(TAB / "trajectories.csv", index=False)
    rep = ["# Security-focused secondary analysis of AgentRewardBench\n",
           "Generated by `code/analysis.py` (seed 2026, 2,000 bootstrap resamples). No language model was called; "
           "the verdicts are those published by Lù et al. (2025).\n",
           "\nConventions: primary (first) expert annotation as reference; 'Unsure' excluded; unreadable judge "
           "verdict counted as negative; 'partially successful' counted as not successful.\n"]
    rep.append(f"\nJudge × trajectory pairs: {len(pairs)}; LLM-judge pairs: {int((~pairs.judge.isin(NON_LLM)).sum())}.\n")
    rep.append("\nTrajectories by benchmark and agent:\n\n")
    rep.append(md_table(traj.groupby(["bench_name", "agent_name"]).size().reset_index(name="n")))
    rq1(traj, rep)
    ceiling(traj, sec, rep)
    rq2(pairs, rep)
    rq3(traj, pairs, rep)
    rq4(traj, pairs, rep)
    (OUT / "analysis_report.md").write_text("".join(rep), encoding="utf-8")
    print("written", OUT / "analysis_report.md")


if __name__ == "__main__":
    main()
