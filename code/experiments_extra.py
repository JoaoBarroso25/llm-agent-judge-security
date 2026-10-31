"""Further analyses of the open-weight evaluator experiments (E8), on the verdicts already collected.

No model is called. Three questions, all on the attack subset (84 trajectories with a side effect, 100 without):
  (a) a committee under attack: majority and any-of-three votes of the three main open-weight evaluators,
      with and without each attack;
  (b) how each attack moves the verdicts: side effects hidden (yes -> no on positives) and false alarms
      before and after the attack (on negatives), for every open-weight model;
  (c) unreadable answers: their rate per condition, and the share hidden when unreadable answers are excluded
      instead of counted as "no".

Usage: python3 code/experiments_extra.py
Output: results/experiments/report_extra.md and results/tables/exp_extra_*.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import OUT, TAB, md_table, pct, wilson  # noqa: E402
from experiments import LOCAL_FULL, LOCAL_MODELS, SHORT_NAMES, attack_rows  # noqa: E402

EXP = OUT / "experiments"
KEYS = ["benchmark", "agent", "task_id"]
ATTACKS = [("A2", "Injection"), ("J1", "Persona"), ("J2", "False authority"), ("J3", "Base64")]


def short(m):
    return SHORT_NAMES.get(m, m)


def load():
    s = pd.read_csv(EXP / "sample.csv", dtype={"task_id": str})
    rows = [json.loads(l) for l in (EXP / "raw.jsonl").read_text().splitlines() if l.strip()]
    r = pd.DataFrame(rows)
    if "error" in r:
        r = r[r.error.isna()]
    r = r[r.cond.str.contains("@", regex=False)].drop_duplicates(["cond"] + KEYS, keep="last")
    r[["base", "model"]] = r.cond.str.split("@", n=1, expand=True)
    sub = attack_rows(s)[KEYS + ["side"]].rename(columns={"side": "y"})
    return r.merge(sub, on=KEYS)


def verdicts(r, base, model):
    d = r[(r.base == base) & (r.model == model)].set_index(KEYS)
    return d.side, d.y


def ci(k, n):
    if not n:
        return "—"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(p)} [{pct(lo)}; {pct(hi)}]"


def committee(r):
    rows = []
    for base, lab in [("O", "No attack")] + ATTACKS + [("G", "Guided, no attack"), ("A4", "Guided + injection"),
                                                       ("J2G", "Guided + false authority")]:
        vs = [verdicts(r, base, m) for m in LOCAL_FULL]
        idx = vs[0][0].index
        for v, _ in vs[1:]:
            idx = idx.intersection(v.index)
        y = vs[0][1].loc[idx]
        votes = sum(v.loc[idx].fillna(0).astype(int) for v, _ in vs)
        for rule, need in (("Majority (2 of 3)", 2), ("Any of the three", 1)):
            flag = (votes >= need).astype(int)
            rows.append({"condition": lab, "base": base, "rule": rule, "n": len(idx),
                         "sensitivity": flag[y == 1].mean(), "false alarms": flag[y == 0].mean(), "_flag": flag, "_y": y})
    out = []
    ref = {(x["rule"], x["base"]): x for x in rows}
    for x in rows:
        base0 = "G" if x["base"] in ("G", "A4", "J2G") else "O"
        r0 = ref[(x["rule"], base0)]
        hid = "—"
        if x["base"] != base0:
            det = r0["_flag"][(r0["_y"] == 1) & (r0["_flag"] == 1)].index.intersection(x["_flag"].index)
            k = int((x["_flag"].loc[det] == 0).sum())
            hid = ci(k, len(det))
        out.append({"Condition": x["condition"], "Committee": x["rule"], "Trajectories": x["n"],
                    "Sensitivity": pct(x["sensitivity"]), "False alarms": pct(x["false alarms"]),
                    "Detected side effects hidden": hid})
    return pd.DataFrame(out)


def movement(r):
    rows = []
    for m in LOCAL_MODELS:
        o, y = verdicts(r, "O", m)
        if o.empty:
            continue
        row = {"Evaluator": short(m), "False alarms, no attack": pct(o[y == 0].fillna(0).mean())}
        for base, lab in ATTACKS:
            a, _ = verdicts(r, base, m)
            idx = o.index.intersection(a.index)
            neg = idx[y.loc[idx] == 0]
            row[f"False alarms, {lab.lower()}"] = pct(a.loc[neg].fillna(0).mean()) if len(neg) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def unreadable(r):
    rows = []
    for m in LOCAL_MODELS:
        o, y = verdicts(r, "O", m)
        if o.empty:
            continue
        row = {"Evaluator": short(m), "Unreadable, no attack": pct(o.isna().mean())}
        for base, lab in ATTACKS:
            a, _ = verdicts(r, base, m)
            idx = o.index.intersection(a.index)
            det = idx[(y.loc[idx] == 1) & (o.loc[idx] == 1)]
            k_all = int((a.loc[det].fillna(0) == 0).sum())
            readable = det[a.loc[det].notna()]
            k_read = int((a.loc[readable] == 0).sum())
            row[f"Unreadable, {lab.lower()}"] = pct(a.loc[idx].isna().mean())
            row[f"Hidden, {lab.lower()} (unreadable = no)"] = f"{k_all}/{len(det)}"
            row[f"Hidden, {lab.lower()} (unreadable excluded)"] = f"{k_read}/{len(readable)}"
        rows.append(row)
    return pd.DataFrame(rows)


def final_tests(r):
    """Injection in a middle step (A2M, against O) and through the web page (A2E, against OX); empty if not run."""
    rows = []
    for m in LOCAL_FULL:
        for base, ref, lab in (("A2", "O", "Injection, last step (agent)"), ("A2M", "O", "Injection, middle step (agent)"),
                               ("A2E", "OX", "Injection on the web page (third party)")):
            a, y = verdicts(r, base, m)
            o, _ = verdicts(r, ref, m)
            if a.empty or o.empty:
                continue
            idx = o.index.intersection(a.index)
            det = idx[(y.loc[idx] == 1) & (o.loc[idx] == 1)]
            k = int((a.loc[det].fillna(0) == 0).sum())
            neg = idx[y.loc[idx] == 0]
            rows.append({"Evaluator": short(m), "Attack": lab, "Detected side effects hidden": ci(k, len(det)),
                         "False alarms before": pct(o.loc[neg].fillna(0).mean()),
                         "False alarms after": pct(a.loc[neg].fillna(0).mean())})
    return pd.DataFrame(rows)


def gemma_temp0():
    """Gemma 4 31B with default sampling and at temperature 0, original and guided prompt, on all 350."""
    s = pd.read_csv(EXP / "sample.csv", dtype={"task_id": str})
    rows = [json.loads(l) for l in (EXP / "raw.jsonl").read_text().splitlines() if l.strip()]
    r = pd.DataFrame(rows)
    if "error" in r:
        r = r[r.error.isna()]
    r = r.drop_duplicates(["cond"] + KEYS, keep="last").merge(s[KEYS + ["side"]].rename(columns={"side": "y"}),
                                                                on=KEYS, suffixes=("", "_s"))
    out = []
    for cond, lab in (("O@gemma4:31b", "Original prompt, default"), ("OTF@gemma4:31b", "Original prompt, temperature 0"),
                      ("G@gemma4:31b", "Guided prompt, default"), ("GTF@gemma4:31b", "Guided prompt, temperature 0")):
        d = r[r.cond == cond]
        if len(d) < len(s):
            continue
        p, y = d.side.fillna(0).astype(int), d.y.astype(int)
        w = y.map({1: 84 / int((y == 1).sum()), 0: 1217 / int((y == 0).sum())})
        tp, fp = (w * ((y == 1) & (p == 1))).sum(), (w * ((y == 0) & (p == 1))).sum()
        fn, tn = (w * ((y == 1) & (p == 0))).sum(), (w * ((y == 0) & (p == 0))).sum()
        W = tp + fp + fn + tn
        po, py, pp = (tp + tn) / W, (tp + fn) / W, (tp + fp) / W
        pe = py * pp + (1 - py) * (1 - pp)
        out.append({"Condition": lab, "Sensitivity": pct(p[y == 1].mean()), "False alarms": pct(p[y == 0].mean()),
                    "κ (population)": f"{(po - pe) / (1 - pe):.2f}"})
    return pd.DataFrame(out)


def main():
    r = load()
    a, b, c = committee(r), movement(r), unreadable(r)
    d, e = final_tests(r), gemma_temp0()
    if len(d[d.Attack != "Injection, last step (agent)"]) if len(d) else False:
        d.to_csv(TAB / "exp_extra_final.csv", index=False)
    if len(e) == 4:
        e.to_csv(TAB / "exp_extra_gemma_temp0.csv", index=False)
    a.to_csv(TAB / "exp_extra_committee.csv", index=False)
    b.to_csv(TAB / "exp_extra_false_alarms.csv", index=False)
    c.to_csv(TAB / "exp_extra_unreadable.csv", index=False)
    rep = ["# Further analyses of the open-weight evaluator experiments", "",
           "Attack subset: 84 trajectories with a side effect and 100 without. No new model calls.", "",
           "## (a) Committee of the three main open-weight evaluators (gemma4:31b, qwen3.6:35b-a3b, gpt-oss:20b)", "",
           md_table(a), "", "## (b) False alarms before and after each attack (trajectories without side effects)", "",
           md_table(b), "", "## (c) Unreadable answers and the share of detected side effects hidden", "", md_table(c)]
    (EXP / "report_extra.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
