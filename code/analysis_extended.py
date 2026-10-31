"""Extended analyses (second round): mechanisms and robustness checks.

  E1  Content analysis of the evaluators' side-effect justifications
      (repetition cues versus state-change cues), by expert label.
  E2  Mechanism: observed repeated actions and false alarms.
  E3  Same-family preference on safety (does an evaluator flag fewer side
      effects for the agent of its own provider?).
  E4  Robustness checks: (a) excluding trajectories at the 30-step limit;
      (b) per-evaluator effect of its own success verdict on false alarms;
      (c) test split only.
  E5  Cost of oversight: alarms to review per true side effect found.

No language model is called. Usage: python3 code/analysis_extended.py
Output: results/tables/ext_*.csv and results/analysis_extended.md
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import (DATA, JUDGES, NON_LLM, OUT, TAB, binary_metrics, boot_ci, build,  # noqa: E402
                      fmt, kappa, md_table, pct, regression_table, safe_logit, wilson)

REPETITION = re.compile(r"repeat|repetit|redundan|loop|multiple times|several times|again and again|same (action|button|element|page|query|step)|over and over|cycl|unproductive|no progress|not mak\w* progress", re.I)
STATE = re.compile(r"submit|purchas|\bbuy|bought|order(ed)?\b|checkout|check out|delet|remov|\bpost(ed)?\b|sent\b|send\w* (an? )?(email|message)|chang\w* (the )?(setting|status|address|password|data|record|value)|modif|edit(ed)?\b|creat\w* (a |an |new )|add\w* to (the )?(cart|wishlist|comparison)|subscrib|cancel|updat|saved?\b|assign|approv|reject|merge|commit|upload|reserv|book(ed)?\b", re.I)
SAME_FAMILY = {"claude": "GenericAgent-anthropic_claude-3.7-sonnet", "gpt-4o": "GenericAgent-gpt-4o-2024-11-20",
               "llama": "GenericAgent-meta-llama_Llama-3.3-70B-Instruct", "qwen": "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct"}


def side_text_src(msg: str):
    """The part of an evaluator's answer that justifies its side-effect verdict.

    Returns (text, source): 'specific' when the answer has a passage about side effects,
    'global' when the evaluator wrote a single reasoning for all questions (then the
    whole reasoning is the only available justification).
    """
    if not msg:
        return "", "none"
    m = re.search(r"Question 2[^\n]*\n?(.*?)(?=Question 3|<optimal>|<side>|$)", msg, re.S | re.I)
    if m and len(m.group(1)) > 30:
        return m.group(1), "specific"
    # answers with one <reasoning> block per question: take the block before <side>
    blocks = re.findall(r"<reasoning>(.*?)</reasoning>\s*<(\w+)>", msg, re.S)
    for txt, nxt in blocks:
        if nxt == "side":
            return txt, "specific"
    sents = re.split(r"(?<=[.!?])\s+", msg)
    spec = " ".join(s for s in sents if re.search(r"side effect|unnecessary|unintended", s, re.I))
    if len(spec) > 30:
        return spec, "specific"
    whole = re.search(r"<reasoning>(.*?)</reasoning>", msg, re.S)
    return (whole.group(1) if whole else msg), "global"


def side_text(msg: str) -> str:
    return side_text_src(msg)[0]


def family(judge):
    for k, a in SAME_FAMILY.items():
        if judge.startswith(k):
            return a
    return None


def e1_content(pairs, rep):
    rep.append("\n## E1 What the evaluators say when they flag a side effect\n")
    rows = []
    for _, r in pairs.iterrows():
        if r.j_side != 1:
            continue
        p = DATA / "judgments" / r.benchmark / r.agent / r.judge / f"{r.task_id}.json"
        j = json.loads(p.read_text())
        ch = (j.get("response") or {}).get("choices") or []
        msg = (ch[0]["message"]["content"] or "") if ch else ""
        t, src = side_text_src(msg)
        rows.append({"judge": r.judge, "expert_side": r.side, "rep": bool(REPETITION.search(t)),
                     "state": bool(STATE.search(t)), "chars": len(t), "source": src})
    d = pd.DataFrame(rows)
    rep.append(f"Justification found in a passage specific to side effects: {pct((d.source == 'specific').mean())} of the alarms; "
               f"in the remaining {pct((d.source != 'specific').mean())} the evaluator wrote one reasoning for all questions, "
               "which is then used as the justification.\n\n")
    spec = d[d.source == "specific"]
    rows_s = []
    for lab, sub in (("false alarms", spec[spec.expert_side == 0]), ("true detections", spec[spec.expert_side == 1])):
        rows_s.append({"alarms (specific passages only)": lab, "n": len(sub), "any repetition cue": pct(sub.rep.mean()),
                       "any state-change cue": pct(sub.state.mean()),
                       "repetition only": pct((sub.rep & ~sub.state).mean())})
    pd.DataFrame(rows_s).to_csv(TAB / "ext_justifications_specific.csv", index=False)
    rep.append(md_table(pd.DataFrame(rows_s)) + "\n")
    d["category"] = np.select([d.rep & ~d.state, ~d.rep & d.state, d.rep & d.state],
                              ["repetition only", "state change only", "both"], "neither")
    d.to_csv(TAB / "ext_justifications.csv", index=False)
    out = []
    for lab, sub in (("false alarms (expert: no side effect)", d[d.expert_side == 0]),
                     ("true detections (expert: side effect)", d[d.expert_side == 1])):
        n = len(sub)
        r = {"alarms": lab, "n": n}
        for c in ("repetition only", "state change only", "both", "neither"):
            k = int((sub.category == c).sum())
            p, lo, hi = wilson(k, n)
            r[c] = f"{pct(p)} [{pct(lo)}; {pct(hi)}]"
            r[c + "_rate"] = p
        r["any repetition cue"] = pct(sub.rep.mean())
        r["any state-change cue"] = pct(sub.state.mean())
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(TAB / "ext_justifications_summary.csv", index=False)
    rep.append("Cue dictionaries (regular expressions) are defined in the script. Unit: evaluator × trajectory pairs in which the evaluator flagged a side effect.\n\n")
    rep.append(md_table(res[[c for c in res.columns if not c.endswith("_rate")]]))
    by_j = d.groupby(["judge", "expert_side"]).rep.mean().unstack()
    by_j.columns = ["rep. cue in false alarms", "rep. cue in true detections"]
    rep.append("\nShare of alarms that mention repetition, per evaluator:\n\n")
    rep.append(md_table(by_j.reset_index().assign(judge=lambda x: x.judge.map(JUDGES)).round(3)))
    return res


def e2_mechanism(pairs, traj, rep):
    rep.append("\n## E2 Repeated actions and false alarms\n")
    traj = traj.copy()
    traj["log_rep"] = np.log1p(traj.repeated_actions)
    m, f = safe_logit(["side ~ log_rep + log_steps + success + C(agent_name) + C(bench_name)",
                       "side ~ log_rep + log_steps + success"], traj.dropna(subset=["side", "success"]))
    rep.append("\n**Expert side effect on repeated actions** (`" + f + "`)\n\n")
    store_occ = []
    rep.append(regression_table(m, ["log_rep", "log_steps"], store_occ))
    llm = pairs[~pairs.judge.isin(NON_LLM)].copy()
    llm["log_rep"] = np.log1p(llm.repeated_actions)
    llm["traj_id"] = llm.benchmark + "|" + llm.agent + "|" + llm.task_id
    neg = llm[llm.side == 0].dropna(subset=["j_side", "loop", "success"]).copy()
    neg["fa"] = neg.j_side.astype(int)
    m, f = safe_logit(["fa ~ log_rep + log_steps + loop + success + C(judge) + C(agent_name) + C(bench_name)"],
                      neg, pd.factorize(neg.traj_id)[0])
    rep.append("\n**False alarm on repeated actions, controlling for length and expert loop** (`" + f + "`)\n\n")
    store = []
    rep.append(regression_table(m, ["log_rep", "log_steps", "loop", "success"], store))
    pd.DataFrame(store).to_csv(TAB / "ext_regression_repetition.csv", index=False)
    pd.DataFrame(store_occ).to_csv(TAB / "ext_regression_repetition_occurrence.csv", index=False)
    # descriptive: by bands of repeated actions
    neg["rep_band"] = pd.cut(neg.repeated_actions, [-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", ">10"])
    rows = []
    for g, d in neg.groupby("rep_band", observed=True):
        p, lo, hi = wilson(int(d.fa.sum()), len(d))
        rows.append({"repeated actions": g, "pairs": len(d), "false alarms": f"{pct(p)} [{pct(lo)}; {pct(hi)}]", "rate": p})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "ext_false_alarms_by_repetition.csv", index=False)
    rep.append("\n" + md_table(t.drop(columns="rate")))
    return store


def e3_family(pairs, rep):
    rep.append("\n## E3 Same-family preference on safety\n")
    llm = pairs[~pairs.judge.isin(NON_LLM)].copy()
    llm["traj_id"] = llm.benchmark + "|" + llm.agent + "|" + llm.task_id
    llm["same_family"] = [int(family(j) == a) for j, a in zip(llm.judge, llm.agent)]
    d = llm.dropna(subset=["j_side", "side", "loop", "success"]).copy()
    d["y"] = d.j_side.astype(int)
    m, f = safe_logit(["y ~ same_family + side + loop + success + log_steps + C(judge) + C(agent_name) + C(bench_name)"],
                      d, pd.factorize(d.traj_id)[0])
    store = []
    rep.append("`" + f + "`\n\n")
    rep.append(regression_table(m, ["same_family"], store))
    pd.DataFrame(store).to_csv(TAB / "ext_regression_family.csv", index=False)
    rows = []
    for (sf, s), g in d.groupby(["same_family", "side"]):
        p, lo, hi = wilson(int(g.y.sum()), len(g))
        rows.append({"same family": sf, "expert side effect": int(s), "pairs": len(g), "flag rate": f"{pct(p)} [{pct(lo)}; {pct(hi)}]"})
    rep.append("\n" + md_table(pd.DataFrame(rows)))
    return store


def e4_robustness(pairs, rep):
    rep.append("\n## E4 Robustness checks\n")
    llm = pairs[~pairs.judge.isin(NON_LLM)].copy()
    llm["traj_id"] = llm.benchmark + "|" + llm.agent + "|" + llm.task_id
    # (a) excluding trajectories at the step limit
    sub = llm[llm.n_steps < 30]
    rows = []
    for judge, d in sub.groupby("judge"):
        dd = d.dropna(subset=["side"])
        y, p = dd.side.astype(int).values, dd.j_side.fillna(0).astype(int).values
        mm = binary_metrics(y, p)
        lo, hi = boot_ci(y, p, kappa)
        rows.append({"judge": JUDGES[judge], "n": mm["n"], "positives": mm["positives"],
                     "sensitivity": mm["sensitivity"], "false_alarm": mm["false_alarm"],
                     "precision": mm["precision"], "kappa": mm["kappa"], "kappa_lo": lo, "kappa_hi": hi})
    a = pd.DataFrame(rows).sort_values("kappa", ascending=False)
    a.to_csv(TAB / "ext_side_without_limit.csv", index=False)
    rep.append(f"\n### (a) Side effects, trajectories below the 30-step limit only ({sub.traj_id.nunique()} trajectories)\n\n")
    rep.append(md_table(a.assign(sensitivity=[pct(x) for x in a.sensitivity], false_alarm=[pct(x) for x in a.false_alarm],
                                 precision=[pct(x) for x in a.precision],
                                 kappa=[f"{fmt(k)} [{fmt(l)}; {fmt(h)}]" for k, l, h in zip(a.kappa, a.kappa_lo, a.kappa_hi)]).drop(columns=["kappa_lo", "kappa_hi"])))
    neg = sub[sub.side == 0].dropna(subset=["j_side", "j_success", "loop", "success"]).copy()
    neg["fa"] = neg.j_side.astype(int)
    m, f = safe_logit(["fa ~ j_success + success + loop + log_steps + C(judge) + C(agent_name) + C(bench_name)"],
                      neg, pd.factorize(neg.traj_id)[0])
    store_a = []
    rep.append("\nConflation regression below the limit (`" + f + "`)\n\n")
    rep.append(regression_table(m, ["j_success", "success", "loop", "log_steps"], store_a))
    pd.DataFrame(store_a).to_csv(TAB / "ext_regression_fa_without_limit.csv", index=False)

    # (b) per-evaluator effect of its own success verdict
    rows = []
    negall = llm[llm.side == 0].dropna(subset=["j_side", "j_success", "loop", "success"]).copy()
    negall["fa"] = negall.j_side.astype(int)
    for judge, d in negall.groupby("judge"):
        try:
            m, f = safe_logit(["fa ~ j_success + success + loop + log_steps + C(agent_name) + C(bench_name)",
                               "fa ~ j_success + success + loop + log_steps"], d)
            ci = m.conf_int().loc["j_success"]
            rows.append({"judge": JUDGES[judge], "OR": math.exp(m.params["j_success"]),
                         "lo": math.exp(ci[0]), "hi": math.exp(ci[1]), "p": m.pvalues["j_success"]})
        except RuntimeError:
            continue
    b = pd.DataFrame(rows).sort_values("OR")
    b.to_csv(TAB / "ext_success_effect_by_judge.csv", index=False)
    rep.append("\n### (b) Effect of the evaluator's own success verdict on false alarms, per evaluator\n\n")
    rep.append(md_table(b.assign(OR=[fmt(x) for x in b.OR], CI=[f"{fmt(l)} to {fmt(h)}" for l, h in zip(b.lo, b.hi)],
                                 p=[f"{x:.3g}" for x in b.p]).drop(columns=["lo", "hi"])))
    rep.append(f"\nEvaluators with OR < 1: {(b.OR < 1).sum()} of {len(b)}; significant at 5% with OR < 1: {((b.OR < 1) & (b.p < 0.05)).sum()}.\n")

    # (c) test split only
    sp = pd.read_csv(DATA / "data" / "splits.csv")
    split = dict(zip(sp.task_id.astype(str), sp.split))
    llm["split"] = [split.get(t, split.get(t.rsplit(".", 1)[0], "unknown")) for t in llm.task_id]
    rows = []
    for s, d in llm.groupby("split"):
        dd = d.dropna(subset=["side"])
        ks = [kappa(g.side.astype(int).values, g.j_side.fillna(0).astype(int).values) for _, g in dd.groupby("judge")]
        rows.append({"split": s, "trajectories": dd.traj_id.nunique(), "positives": int(dd.drop_duplicates("traj_id").side.sum()),
                     "mean κ side": fmt(np.nanmean(ks)), "max κ side": fmt(np.nanmax(ks))})
    c = pd.DataFrame(rows)
    c.to_csv(TAB / "ext_by_split.csv", index=False)
    rep.append("\n### (c) By split of the original benchmark\n\n" + md_table(c))
    return a, b, c


def e5_cost(rep):
    d = pd.read_csv(TAB / "rq2_judges.csv")
    d = d[d.dimension == "side"].copy()
    d["alarms_per_true"] = (d.tp + d.fp) / d.tp
    d["missed"] = d.fn
    d = d.sort_values("alarms_per_true")
    d[["judge_name", "tp", "fp", "fn", "alarms_per_true"]].to_csv(TAB / "ext_cost.csv", index=False)
    rep.append("\n## E5 Cost of oversight\n\nAlarms that a human reviewer must inspect to find one real side effect, and side effects missed (of 84).\n\n")
    rep.append(md_table(pd.DataFrame({"Evaluator": d.judge_name, "alarms": d.tp + d.fp, "true": d.tp,
                                      "alarms per true side effect": [fmt(x, 1) for x in d.alarms_per_true],
                                      "missed": d.fn})))
    base = 1301 / 84
    rep.append(f"\nReviewing trajectories at random would require {base:.1f} trajectories per side effect found.\n")


def main():
    traj, pairs, _ = build()
    rep = ["# Extended analyses (second round)\n", "Generated by `code/analysis_extended.py`; no model was called.\n"]
    llm_pairs = pairs[~pairs.judge.isin(NON_LLM)]
    e1_content(llm_pairs, rep)
    e2_mechanism(pairs, traj, rep)
    e3_family(pairs, rep)
    e4_robustness(pairs, rep)
    e5_cost(rep)
    (OUT / "analysis_extended.md").write_text("".join(rep), encoding="utf-8")
    print("written", OUT / "analysis_extended.md")


if __name__ == "__main__":
    main()


def validation_sample(n=60, seed=2026):
    """Blind sample of alarm justifications for manual validation of the cue dictionaries.

    Writes results/validation/justifications_sample.csv (without expert labels or automatic
    categories) and a key file. Never overwrites an existing sample, so manual labels are kept.
    """
    vdir = OUT / "validation"
    vdir.mkdir(exist_ok=True)
    target = vdir / "justifications_sample.csv"
    if target.exists():
        print("validation sample exists; not overwritten")
        return
    traj, pairs, _ = build()
    llm = pairs[(~pairs.judge.isin(NON_LLM)) & (pairs.j_side == 1)].sample(n, random_state=seed)
    rows, key = [], []
    for i, (_, r) in enumerate(llm.iterrows(), 1):
        p = DATA / "judgments" / r.benchmark / r.agent / r.judge / f"{r.task_id}.json"
        msg = json.loads(p.read_text())["response"]["choices"][0]["message"]["content"] or ""
        t = " ".join(side_text(msg).split())
        rows.append({"id": i, "goal": " ".join(str(r.goal).split())[:300], "justification": t[:1500],
                     "manual_category (repetition / state change / both / neither)": ""})
        key.append({"id": i, "benchmark": r.benchmark, "agent": r.agent, "task_id": r.task_id, "judge": r.judge,
                    "expert_side": r.side, "auto_rep": bool(REPETITION.search(t)), "auto_state": bool(STATE.search(t))})
    pd.DataFrame(rows).to_csv(target, index=False)
    pd.DataFrame(key).to_csv(vdir / "justifications_sample_KEY.csv", index=False)
    print("validation sample written")


if __name__ == "__main__" and "--validation" in sys.argv:
    validation_sample()
