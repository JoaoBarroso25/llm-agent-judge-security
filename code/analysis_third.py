"""Third round of analyses: task type, environment, surface baselines and triage.

  T1  Task type: tasks whose official evaluation inspects the final state of the
      website (program_html) versus information-seeking tasks.
  T2  Environment: website (WebArena, VisualWebArena), task category (WorkArena),
      difficulty (VisualWebArena).
  T3  What the alarms track: cross-validated logistic models that predict the expert
      labels and each evaluator's side-effect verdict from surface features of the
      trajectory only (length, repetition, navigation, action types, final action).
      Matched-rate comparison: a surface rule that flags as many trajectories as each
      evaluator.
  T4  Triage: the number of evaluators that flag a trajectory, used as a ranking
      score (AUC, precision among the top-ranked trajectories), against length alone.
  T5  Same trajectories, human versus evaluator: on the doubly annotated subset, the
      second expert and each evaluator are compared with the first expert (paired
      bootstrap).

No language model is called. Usage: python3 code/analysis_third.py
Output: results/tables/t*_*.csv and results/analysis_third.md
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import (DATA, JUDGES, NON_LLM, OUT, STEP_RE, TAB, binary_metrics, build,  # noqa: E402
                      fmt, kappa, md_table, pct, regression_table, safe_logit, wilson)

SEED = 2026
N_BOOT = 1000
FOLDS = 10
FEATURE_JUDGE = "gpt-4o-noscreen"  # any simplified judge: the action history in its prompt is the same
ACTION_TYPES = ("click", "fill", "select_option", "goto", "go_back", "scroll", "send_msg_to_user",
                "report_infeasible", "tab_focus", "new_tab", "hover", "press", "noop")
SURFACE = ["log_steps", "log_rep", "log_urls", "at_limit", "log_fill", "log_select", "log_nav", "log_scroll",
           "final_msg", "final_infeasible"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def auc(y, s):
    """Area under the ROC curve (Mann-Whitney), ties counted as one half."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot(fn, *arrays, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = fn(*(a[i] for a in arrays))
        if not math.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (math.nan, math.nan)


def ci_str(v, lo, hi, d=2):
    return f"{fmt(v, d)} [{fmt(lo, d)}; {fmt(hi, d)}]"


def wil_str(k, n):
    p, lo, hi = wilson(int(k), int(n))
    return f"{int(k)}/{int(n)} = {pct(p)} [{pct(lo)}; {pct(hi)}]"


def cv_scores(X, y, folds=FOLDS, seed=SEED):
    """Out-of-fold predicted probabilities of a logistic regression."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(np.arange(len(y)) % folds)
    out = np.full(len(y), np.nan)
    Xc = sm.add_constant(X, has_constant="add")
    for f in range(folds):
        tr, te = fold != f, fold == f
        try:
            m = sm.Logit(y[tr], Xc[tr]).fit(disp=0, maxiter=200)
        except Exception:
            m = sm.Logit(y[tr], Xc[tr]).fit_regularized(disp=0, alpha=1.0)
        out[te] = m.predict(Xc[te])
    return out


def top_flags(score, q, seed=SEED):
    """Flag the fraction q of trajectories with the highest score (ties broken at random)."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(score)), -np.asarray(score)))
    flags = np.zeros(len(score), dtype=int)
    flags[order[:int(round(q * len(score)))]] = 1
    return flags


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def task_meta():
    d = DATA / "data"
    rows = []
    for bench in ("webarena", "visualwebarena"):
        t = pd.read_csv(d / f"{bench}.csv")
        for _, r in t.iterrows():
            sites = str(r.sites).split()
            rows.append({"task_id": r.task_name, "benchmark": bench,
                         "site": sites[0] if len(sites) == 1 else "multi-site",
                         "task_type": "state change" if "program_html" in str(r.eval_types) else "information",
                         "difficulty": r.get("overall_difficulty", np.nan), "category": np.nan})
    t = pd.read_csv(d / "workarena.csv")
    for _, r in t.iterrows():
        rows.append({"task_id": r.task_name, "benchmark": "workarena", "site": "servicenow",
                     "task_type": np.nan, "difficulty": np.nan, "category": r.category})
    t = pd.read_csv(d / "assistantbench.csv")
    for _, r in t.iterrows():
        rows.append({"task_id": r.task_name, "benchmark": "assistantbench", "site": "open web",
                     "task_type": "information", "difficulty": r.difficulty, "category": np.nan})
    return pd.DataFrame(rows).drop_duplicates(["benchmark", "task_id"])


def action_counts(r):
    p = DATA / "judgments" / r.benchmark / r.agent / FEATURE_JUDGE / f"{r.task_id}.json"
    j = json.loads(p.read_text())
    content = ""
    for m in j["chat_messages"]["regular"]:
        if m["role"] != "user":
            continue
        c = m["content"]
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        content += c
    acts = [s[2].strip() for s in STEP_RE.findall(content + "-----") if s[2].strip() not in ("None", "")]
    types = [a.split("(")[0].strip() for a in acts]
    return {f"n_{t}": types.count(t) for t in ACTION_TYPES}


def prepare():
    traj, pairs, sec = build()
    meta = task_meta().rename(columns={"task_id": "meta_id"})
    # trajectory ids carry the variant of the benchmark ("visualwebarena.resized.N", "assistantbench.improved.…")
    traj["meta_id"] = traj.task_id.str.replace(".resized", "", regex=False).str.replace(".improved", "", regex=False)
    traj = traj.merge(meta, on=["benchmark", "meta_id"], how="left")
    counts = pd.DataFrame([action_counts(r) for r in traj.itertuples()])
    traj = pd.concat([traj.reset_index(drop=True), counts], axis=1)
    traj["log_rep"] = np.log1p(traj.repeated_actions)
    traj["log_urls"] = np.log1p(traj.distinct_urls)
    traj["log_fill"] = np.log1p(traj.n_fill)
    traj["log_select"] = np.log1p(traj.n_select_option)
    traj["log_nav"] = np.log1p(traj.n_goto + traj.n_go_back)
    traj["log_scroll"] = np.log1p(traj.n_scroll)
    traj["final_msg"] = (traj.final_action == "send_msg_to_user").astype(int)
    traj["final_infeasible"] = (traj.final_action == "report_infeasible").astype(int)
    llm = pairs[~pairs.judge.isin(NON_LLM)].copy()
    for d in ("side", "loop", "success"):
        llm[f"j_{d}"] = llm[f"j_{d}"].map(lambda v: 1 if v == 1 else 0)
    keys = ["benchmark", "agent", "task_id"]
    extra = ["site", "task_type", "difficulty", "category"] + [c for c in traj.columns if c.startswith(("n_", "log_"))
                                                               and c not in llm.columns]
    llm = llm.merge(traj[keys + extra], on=keys, how="left")
    return traj, llm, sec


def group_metrics(traj, llm, col, rep, title, fname, min_n=1):
    rows = []
    for g, sub in traj.groupby(col):
        s = sub.dropna(subset=["side"])
        if len(s) < min_n:
            continue
        lp = llm[(llm[col] == g)].dropna(subset=["side"])
        ks = [kappa(x.side.astype(int), x.j_side) for _, x in lp.groupby("judge")]
        neg, pos = lp[lp.side == 0], lp[lp.side == 1]
        rows.append({col: g, "trajectories": len(s), "side effects": wil_str(s.side.sum(), len(s)),
                     "loops": pct(sub.loop.mean()), "success": pct(sub.success.mean()),
                     "mean steps": fmt(sub.n_steps.mean(), 1),
                     "false alarms (mean)": pct(neg.j_side.mean()),
                     "sensitivity (mean)": pct(pos.j_side.mean()) if len(pos) else "—",
                     "mean κ": fmt(np.nanmean(ks)) if s.side.sum() else "—",
                     "max κ": fmt(np.nanmax(ks)) if s.side.sum() else "—"})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / fname, index=False)
    rep.append(f"\n{title}\n\n" + md_table(df))
    return df


# --------------------------------------------------------------------------- #
# T1 Task type
# --------------------------------------------------------------------------- #
def t1_task_type(traj, llm, rep):
    rep.append("\n## T1 Task type: state-changing versus information-seeking tasks\n")
    rep.append("Task type from the official evaluation of each task: WebArena and VisualWebArena tasks whose "
               "evaluation includes `program_html` (the evaluator inspects the final state of the website) are "
               "*state-changing*; the others (`string_match`, `url_match`, `page_image_query`) and all "
               "AssistantBench tasks are *information-seeking*. WorkArena is analysed by category in T2.\n")
    sub = traj[traj.task_type.notna()]
    lsub = llm[llm.task_type.notna()]
    rep.append(f"\nTrajectories with a task type: {len(sub)} (unmatched in metadata: "
               f"{int(traj[traj.benchmark.isin(['webarena', 'visualwebarena', 'assistantbench'])].task_type.isna().sum())}).\n")
    group_metrics(sub, lsub, "task_type", rep, "**By task type (WebArena, VisualWebArena, AssistantBench)**",
                  "t1_task_type.csv")
    for bench in ("webarena", "visualwebarena"):
        group_metrics(sub[sub.benchmark == bench], lsub[lsub.benchmark == bench], "task_type", rep,
                      f"**{bench}**", f"t1_task_type_{bench}.csv")
    d = sub.dropna(subset=["side"]).copy()
    d["state_task"] = (d.task_type == "state change").astype(int)
    m, f = safe_logit(["side ~ state_task + log_steps + success + C(agent_name) + C(bench_name)",
                       "side ~ state_task + log_steps + success"], d)
    store = []
    rep.append(f"\n**Expert side effect on task type** (`{f}`)\n\n"
               + regression_table(m, keep=["state_task", "log_steps", "success"], store=store))
    pd.DataFrame(store).to_csv(TAB / "t1_reg_occurrence.csv", index=False)
    l = lsub.dropna(subset=["side", "success", "loop", "log_steps"]).copy()
    l = l[l.side == 0]
    l["state_task"] = (l.task_type == "state change").astype(int)
    l["tid"] = l.benchmark + "|" + l.agent + "|" + l.task_id
    m, f = safe_logit(["j_side ~ state_task + log_steps + loop + success + C(judge) + C(agent_name) + C(bench_name)"],
                      l, groups=pd.factorize(l.tid)[0])
    rep.append(f"\n**False alarm on task type, expert negatives, clustered by trajectory** (`{f}`)\n\n"
               + regression_table(m, keep=["state_task", "log_steps", "loop", "success"], store=(store := [])))
    pd.DataFrame(store).to_csv(TAB / "t1_reg_false_alarms.csv", index=False)
    # where the side effects are: share of side effects in information-seeking tasks
    s = d[d.side == 1]
    rep.append(f"\nSide effects in information-seeking tasks: {int((s.state_task == 0).sum())} of {len(s)} "
               f"({pct((s.state_task == 0).mean())}); share of information-seeking tasks among all trajectories: "
               f"{pct((d.state_task == 0).mean())}.\n")


# --------------------------------------------------------------------------- #
# T2 Environment
# --------------------------------------------------------------------------- #
def t2_environment(traj, llm, rep):
    rep.append("\n## T2 Environment: website, task category and difficulty\n")
    for bench in ("webarena", "visualwebarena"):
        group_metrics(traj[traj.benchmark == bench], llm[llm.benchmark == bench], "site", rep,
                      f"**{bench} by website**", f"t2_site_{bench}.csv")
    group_metrics(traj[traj.benchmark == "workarena"], llm[llm.benchmark == "workarena"], "category", rep,
                  "**WorkArena by task category**", "t2_category_workarena.csv")
    group_metrics(traj[traj.benchmark == "visualwebarena"], llm[llm.benchmark == "visualwebarena"], "difficulty",
                  rep, "**VisualWebArena by overall difficulty**", "t2_difficulty_visualwebarena.csv")
    rep.append(f"\nWorkArena trajectories without a category in the metadata: "
               f"{int(traj[traj.benchmark == 'workarena'].category.isna().sum())}.\n")


# --------------------------------------------------------------------------- #
# T3 Surface baselines
# --------------------------------------------------------------------------- #
def t3_surface(traj, llm, rep):
    rep.append("\n## T3 What the alarms track: surface-feature models\n")
    rep.append("Features (per trajectory, from the action history only): " + ", ".join(SURFACE) + ". "
               f"Logistic regression, {FOLDS}-fold cross-validation (seed {SEED}); AUC of the out-of-fold "
               f"predictions with a bootstrap 95% CI ({N_BOOT} resamples).\n")
    d = traj.dropna(subset=["side", "loop", "success"]).reset_index(drop=True)
    X = d[SURFACE].values.astype(float)
    wide = llm.pivot_table(index=["benchmark", "agent", "task_id"], columns="judge", values="j_side").reset_index()
    d = d.merge(wide, on=["benchmark", "agent", "task_id"], how="left")
    judges = [j for j in JUDGES if j not in NON_LLM]
    d["votes"] = d[judges].sum(axis=1)
    rows, scores = [], {}
    targets = [("expert: side effect", d.side.astype(int).values), ("expert: loop", d.loop.astype(int).values),
               ("expert: success", d.success.astype(int).values),
               ("majority of evaluators: side effect", (d.votes >= 6).astype(int).values)]
    targets += [(f"{JUDGES[j]}: side effect", d[j].astype(int).values) for j in judges]
    for name, y in targets:
        s = cv_scores(X, y)
        scores[name] = s
        a = auc(y, s)
        lo, hi = boot(auc, y, s)
        rows.append({"target": name, "positives": f"{int(y.sum())}/{len(y)}", "CV AUC [95% CI]": ci_str(a, lo, hi)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "t3_surface_auc.csv", index=False)
    rep.append("\n**How well the surface features predict each label**\n\n" + md_table(df))
    jud_auc = [auc(d[j].astype(int), scores[f"{JUDGES[j]}: side effect"]) for j in judges]
    rep.append(f"\nMean CV AUC over the 11 evaluators' side-effect verdicts: {fmt(np.mean(jud_auc))} "
               f"(range {fmt(min(jud_auc))} to {fmt(max(jud_auc))}).\n")
    # full-sample coefficients for the expert side-effect model and the majority model
    for name, y in (("expert: side effect", d.side.astype(int)), ("majority of evaluators: side effect", (d.votes >= 6).astype(int))):
        m = sm.Logit(y.values, sm.add_constant(d[SURFACE].astype(float))).fit(disp=0, maxiter=200, cov_type="HC1")
        ci = m.conf_int()
        tab = pd.DataFrame({"variable": SURFACE, "OR": [fmt(math.exp(m.params[v])) for v in SURFACE],
                            "95% CI": [f"{fmt(math.exp(ci.loc[v, 0]))} to {fmt(math.exp(ci.loc[v, 1]))}" for v in SURFACE],
                            "p": [f"{m.pvalues[v]:.3g}" for v in SURFACE]})
        tab.to_csv(TAB / f"t3_coef_{'expert' if name.startswith('expert') else 'majority'}.csv", index=False)
        rep.append(f"\n**Coefficients, full sample: {name}** (pseudo-R² {fmt(m.prsquared, 3)})\n\n" + md_table(tab))
    # how the agent ends the run: expert prevalence against evaluator alarms
    d["ending"] = np.select([d.final_msg == 1, d.final_infeasible == 1],
                            ["answer to the user", "declares the task infeasible"], "no final message")
    rows = []
    for g, x in d.groupby("ending"):
        neg, pos = x[x.side == 0], x[x.side == 1]
        rows.append({"how the run ends": g, "trajectories": len(x), "expert side effects": wil_str(x.side.sum(), len(x)),
                     "evaluators flagging (mean of 11)": pct(x[judges].values.mean()),
                     "false alarms (mean)": pct(neg[judges].values.mean()),
                     "sensitivity (mean)": pct(pos[judges].values.mean()) if len(pos) else "—"})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "t3_ending.csv", index=False)
    rep.append("\n**How the run ends: expert side effects against evaluator alarms**\n\n" + md_table(df))
    # matched-rate comparison against the expert label
    y = d.side.astype(int).values
    s_surf = scores["expert: side effect"]
    rows = []
    for j in judges:
        v = d[j].astype(int).values
        q = v.mean()
        rule_surf = top_flags(s_surf, q)
        rule_len = top_flags(d.n_steps.values, q)
        k_j, k_s, k_l = kappa(y, v), kappa(y, rule_surf), kappa(y, rule_len)
        lo, hi = boot(lambda a, b, c: kappa(a, b) - kappa(a, c), y, v, rule_surf)
        rows.append({"evaluator": JUDGES[j], "flag rate": pct(q), "κ evaluator": fmt(k_j),
                     "κ surface rule (same rate)": fmt(k_s), "κ length rule (same rate)": fmt(k_l),
                     "evaluator − surface [95% CI]": ci_str(k_j - k_s, lo, hi),
                     "AUC evaluator": fmt(auc(y, v)), "_kj": k_j, "_ks": k_s, "_kl": k_l})
    df = pd.DataFrame(rows).sort_values("_kj", ascending=False)
    df.to_csv(TAB / "t3_matched_rate.csv", index=False)
    rep.append("\n**Each evaluator against a surface rule and a length rule that flag the same share of trajectories** "
               "(κ with the expert side-effect label; surface rule = out-of-fold surface model for the expert label)\n\n"
               + md_table(df[[c for c in df.columns if not c.startswith("_")]]))
    better = int((df._kj > df._ks).sum())
    rep.append(f"\nEvaluators with a higher κ than the surface rule at the same flag rate: {better} of {len(df)}; "
               f"than the length rule: {int((df._kj > df._kl).sum())} of {len(df)}.\n")
    return d, scores


# --------------------------------------------------------------------------- #
# T4 Triage ranking
# --------------------------------------------------------------------------- #
def t4_triage(d, llm, scores, rep):
    rep.append("\n## T4 Triage: ranking trajectories for human review\n")
    rep.append("Score = number of evaluators (0-11) that flag the trajectory. AUC against the expert label, "
               "bootstrap 95% CI; precision and recall among the top-ranked trajectories (ties broken at random, "
               "seed 2026).\n")
    y = d.side.astype(int).values
    rows = []
    for name, s in (("votes of 11 evaluators", d.votes.values), ("number of steps", d.n_steps.values),
                    ("surface model (CV)", scores["expert: side effect"])):
        a = auc(y, s)
        lo, hi = boot(auc, y, s)
        r = {"score": name, "AUC [95% CI]": ci_str(a, lo, hi)}
        for q in (0.05, 0.10, 0.20):
            f = top_flags(s, q)
            r[f"precision top {int(q * 100)}%"] = pct(y[f == 1].mean())
            r[f"recall top {int(q * 100)}%"] = pct(f[y == 1].mean())
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "t4_triage.csv", index=False)
    rep.append("\n**Side effects**\n\n" + md_table(df))
    lo, hi = boot(lambda a, b, c: auc(a, b) - auc(a, c), y, d.votes.values, d.n_steps.values)
    rep.append(f"\nDifference in AUC, votes − steps: {fmt(auc(y, d.votes.values) - auc(y, d.n_steps.values))} "
               f"[{fmt(lo)}; {fmt(hi)}]. Base rate of side effects: {pct(y.mean())}.\n")
    keys = ["benchmark", "agent", "task_id"]
    rows = []
    for dim in ("side", "loop", "success"):
        v = llm.groupby(keys)[f"j_{dim}"].sum().rename("v").reset_index()
        x = d[keys + [dim]].merge(v, on=keys)
        yy = x[dim].astype(int).values
        a = auc(yy, x.v.values)
        lo, hi = boot(auc, yy, x.v.values)
        rows.append({"dimension": dim, "positives": f"{int(yy.sum())}/{len(yy)}",
                     "AUC of votes [95% CI]": ci_str(a, lo, hi)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "t4_votes_by_dimension.csv", index=False)
    rep.append("\n**The same vote score for each dimension**\n\n" + md_table(df))


# --------------------------------------------------------------------------- #
# T5 Human versus evaluator on the same trajectories
# --------------------------------------------------------------------------- #
def t5_same_trajectories(traj, llm, sec, rep):
    rep.append("\n## T5 Human versus evaluator on the same trajectories\n")
    keys = ["benchmark", "agent", "task_id"]
    both = traj[keys + ["side", "loop", "success"]].merge(sec, on=keys, suffixes=("", "_2"))
    rows = []
    for dim in ("side", "loop", "success"):
        b = both.dropna(subset=[dim, f"{dim}_2"])
        lp = llm.merge(b[keys + [f"{dim}_2"]], on=keys)
        y1 = b.set_index(keys)[dim].astype(int)
        y2 = b.set_index(keys)[f"{dim}_2"].astype(int)
        k2 = kappa(y1.values, y2.values)
        ks = {}
        for j, x in lp.groupby("judge"):
            x = x.set_index(keys).loc[y1.index]
            ks[j] = (kappa(y1.values, x[f"j_{dim}"].values), x[f"j_{dim}"].values)
        best = max(ks, key=lambda j: ks[j][0])
        lo, hi = boot(lambda a, b2, c: kappa(a, b2) - kappa(a, c), y1.values, y2.values, ks[best][1])
        rows.append({"dimension": dim, "trajectories": len(b), "positives (1st expert)": int(y1.sum()),
                     "κ second expert": fmt(k2), "κ best evaluator": f"{fmt(ks[best][0])} ({JUDGES[best]})",
                     "κ mean evaluator": fmt(np.nanmean([v[0] for v in ks.values()])),
                     "second expert − best evaluator [95% CI]": ci_str(k2 - ks[best][0], lo, hi)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "t5_same_trajectories.csv", index=False)
    rep.append("Doubly annotated trajectories; reference = first expert; paired bootstrap over trajectories.\n\n"
               + md_table(df))


# --------------------------------------------------------------------------- #
# T6 Unsafe success as a reward
# --------------------------------------------------------------------------- #
def t6_reward(llm, rep):
    rep.append("\n## T6 Would the evaluators reward unsafe success?\n")
    rep.append("Expert-successful trajectories, split by the expert side-effect label. 'Accepted' = the evaluator judges "
               "the task successful; 'accepted as safe' = it judges the task successful and flags no side effect.\n")
    x = llm.dropna(subset=["side", "success"])
    succ = x[x.success == 1]
    rows = []
    for lab, sub in (("successful, no side effect", succ[succ.side == 0]), ("successful, with side effect", succ[succ.side == 1])):
        n = len(sub)
        acc = int(sub.j_success.sum())
        safe = int(((sub.j_success == 1) & (sub.j_side == 0)).sum())
        rows.append({"expert": lab, "trajectories": sub[["benchmark", "agent", "task_id"]].drop_duplicates().shape[0],
                     "pairs": n, "accepted": wil_str(acc, n), "accepted as safe": wil_str(safe, n)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "t6_reward.csv", index=False)
    rep.append("\n" + md_table(df))
    s = succ.copy()
    s["tid"] = s.benchmark + "|" + s.agent + "|" + s.task_id
    m, f = safe_logit(["j_success ~ side + log_steps + C(judge) + C(bench_name)", "j_success ~ side + log_steps + C(judge)"],
                      s, groups=pd.factorize(s.tid)[0])
    store = []
    rep.append(f"\n**Acceptance of expert-successful runs on the expert side-effect label** (`{f}`, clustered by trajectory)\n\n"
               + regression_table(m, keep=["side", "log_steps"], store=store))
    pd.DataFrame(store).to_csv(TAB / "t6_reg.csv", index=False)
    # per evaluator: share of unsafe successes accepted, and of safe successes wrongly rejected by the side verdict
    rows = []
    for j, sub in succ.groupby("judge"):
        u, sf = sub[sub.side == 1], sub[sub.side == 0]
        rows.append({"evaluator": JUDGES[j], "unsafe successes accepted": pct(u.j_success.mean()),
                     "unsafe successes accepted as safe": pct(((u.j_success == 1) & (u.j_side == 0)).mean()),
                     "safe successes accepted as safe": pct(((sf.j_success == 1) & (sf.j_side == 0)).mean()),
                     "_u": ((u.j_success == 1) & (u.j_side == 0)).mean(), "_s": ((sf.j_success == 1) & (sf.j_side == 0)).mean()})
    df = pd.DataFrame(rows).sort_values("_s", ascending=False)
    df.to_csv(TAB / "t6_reward_by_judge.csv", index=False)
    rep.append("\n**Per evaluator** (16 unsafe and 339 safe expert successes)\n\n"
               + md_table(df[[c for c in df.columns if not c.startswith("_")]]))
    rep.append(f"\nEvaluators that accept a larger share of unsafe than of safe successes as safe: "
               f"{int((df._u > df._s).sum())} of {len(df)}.\n")


# --------------------------------------------------------------------------- #
# T7 Agent-level estimates (leaderboards)
# --------------------------------------------------------------------------- #
def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1])


def t7_leaderboard(traj, llm, rep):
    rep.append("\n## T7 Agent-level estimates: can the evaluators rank agents by safety?\n")
    rep.append("Cells = agent × benchmark (15). For each cell: expert success rate, side-effect rate and *safe-success* rate "
               "(success without side effect, analogous to completion under policy); the evaluator's estimate of each. "
               "Spearman correlation across cells and mean absolute error (percentage points); 95% CI by resampling "
               f"trajectories within cells ({N_BOOT} resamples).\n")
    keys = ["bench_name", "agent_name"]
    t = traj.dropna(subset=["side", "success"]).copy()
    t["safe_success"] = ((t.success == 1) & (t.side == 0)).astype(int)
    x = llm.dropna(subset=["side", "success"]).copy()
    x["j_safe"] = ((x.j_success == 1) & (x.j_side == 0)).astype(int)
    exp_cells = t.groupby(keys)[["success", "side", "safe_success"]].mean()
    rng = np.random.default_rng(SEED)
    idx = {k: g.index.values for k, g in t.groupby(keys)}
    tkeys = ["benchmark", "agent", "task_id"]
    rows = []
    for j, sub in x.groupby("judge"):
        sub = sub.set_index(tkeys)
        jt = t.set_index(tkeys).join(sub[["j_success", "j_side", "j_safe"]], how="inner").reset_index()
        cells = jt.groupby(keys)[["j_success", "j_side", "j_safe"]].mean().loc[exp_cells.index]
        r = {"evaluator": JUDGES[j]}
        for dim, jd in (("success", "j_success"), ("side", "j_side"), ("safe_success", "j_safe")):
            rho = spearman(exp_cells[dim], cells[jd])
            # bootstrap within cells
            gidx = {k: g.index.values for k, g in jt.groupby(keys)}
            vals = []
            for _ in range(N_BOOT):
                e, v = [], []
                for k in exp_cells.index:
                    ii = rng.choice(gidx[k], len(gidx[k]))
                    e.append(jt.loc[ii, dim].mean())
                    v.append(jt.loc[ii, jd].mean())
                vals.append(spearman(e, v))
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            r[f"ρ {dim}"] = ci_str(rho, lo, hi)
            r[f"MAE {dim} (pp)"] = fmt(100 * np.mean(np.abs(exp_cells[dim].values - cells[jd].values)), 1)
            r[f"_{dim}"] = rho
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("_side", ascending=False)
    df.to_csv(TAB / "t7_leaderboard.csv", index=False)
    rep.append("\n" + md_table(df[[c for c in df.columns if not c.startswith("_")]]))
    rep.append(f"\nMean ρ over evaluators: success {fmt(df._success.mean())}, side effects {fmt(df._side.mean())}, "
               f"safe success {fmt(df._safe_success.mean())}.\n")
    cell = exp_cells.join(x.groupby(keys)[["j_side"]].mean()).join(t.groupby(keys)[["n_steps"]].mean()).reset_index()
    cell.to_csv(TAB / "t7_cells.csv", index=False)
    rep.append(f"\nAcross the 15 cells, Spearman ρ between the mean evaluator flag rate and the mean number of steps: "
               f"{fmt(spearman(cell.j_side, cell.n_steps))}; between the expert side-effect rate and the mean number of steps: "
               f"{fmt(spearman(cell.side, cell.n_steps))}.\n")
    # agent-level (pooled over benchmarks): expert and mean-evaluator rates
    a = t.groupby("agent_name")[["success", "side", "safe_success"]].mean()
    xa = x.groupby("agent_name")[["j_success", "j_side", "j_safe"]].mean()
    tab = pd.DataFrame({"agent": a.index, "expert success": [pct(v) for v in a.success],
                        "evaluators' success (mean)": [pct(v) for v in xa.loc[a.index].j_success],
                        "expert side effects": [pct(v) for v in a.side],
                        "evaluators' side effects (mean)": [pct(v) for v in xa.loc[a.index].j_side],
                        "expert safe success": [pct(v) for v in a.safe_success],
                        "evaluators' safe success (mean)": [pct(v) for v in xa.loc[a.index].j_safe]})
    tab.to_csv(TAB / "t7_agents.csv", index=False)
    rep.append("\n**By agent (all benchmarks)**\n\n" + md_table(tab))


# --------------------------------------------------------------------------- #
# T8 Detection by evaluated agent
# --------------------------------------------------------------------------- #
def t8_by_agent(traj, llm, rep):
    rep.append("\n## T8 Detection of side effects by evaluated agent\n")
    group_metrics(traj, llm, "agent_name", rep, "**By agent (all benchmarks; means over the 11 evaluators)**",
                  "t8_by_agent.csv")


def main():
    traj, llm, sec = prepare()
    rep = ["# Third round of analyses\n", "Generated by `code/analysis_third.py`; no model was called.\n"]
    t1_task_type(traj, llm, rep)
    t2_environment(traj, llm, rep)
    d, scores = t3_surface(traj, llm, rep)
    t4_triage(d, llm, scores, rep)
    t5_same_trajectories(traj, llm, sec, rep)
    t6_reward(llm, rep)
    t7_leaderboard(traj, llm, rep)
    t8_by_agent(traj, llm, rep)
    (OUT / "analysis_third.md").write_text("".join(rep), encoding="utf-8")
    print("written", OUT / "analysis_third.md")


if __name__ == "__main__":
    main()
