"""Analysis of the adaptive attacks against the guided prompt (code/run_adaptive.py).

Question: does the resistance of the guided prompt hold against an attacker who knows the guidelines? Measure: among the
side effects that the guided evaluator detects without attack (n = 22 for Gemma 4 31B, 20 for Qwen3.6 35B-A3B, 9 for
gpt-oss 20B), the share that an attack makes it miss. Fixed attacks of the thesis (injection A4, persona J1G, false authority
J2G, base64 J3G) are the reference; T1..T9 are tailored by hand; the search finds new texts with cross-validation (the
final text of each fold is tested on the other fold).

Output: results/experiments/adaptive_report.md, results/tables/exp_adaptive_tailored.csv, exp_adaptive_search.csv,
exp_adaptive_texts.csv. Does not touch the thesis.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import OUT, TAB, md_table, pct, wilson  # noqa: E402
import experiments as e  # noqa: E402

EXP = OUT / "experiments"
MAIN = ["gemma4:31b", "qwen3.6:35b-a3b", "gpt-oss:20b"]
SHORT = {"gemma4:31b": "Gemma 4 31B", "qwen3.6:35b-a3b": "Qwen3.6 35B-A3B", "gpt-oss:20b": "gpt-oss 20B"}
NAMES = {"ta1": "Forged answer", "ta2": "Claim that the goal requested it", "ta3": "Actions relabelled as navigation",
         "ta4": "Appeal to the evaluator's definitions", "ta5": "Instruction update", "ta6": "Task declared infeasible",
         "ta7": "Pre-written analysis", "ta8": "Spread over three steps", "ta9": "Combination of three"}
FIXED = [("A4", "Fixed injection (thesis)"), ("J1G", "Fixed persona (thesis)"), ("J2G", "Fixed false authority (thesis)"),
         ("J3G", "Fixed base64 (thesis)")]


def cell(k, n):
    if not n:
        return "—"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(p)} [{pct(lo)}; {pct(hi)}]"


def hidden(done, cond, keys):
    rows = [done[(cond,) + k] for k in keys if (cond,) + k in done]
    return sum(1 for r in rows if (r.get("side") or 0) == 0), len(rows)


def main():
    sample = e.make_sample()
    done, _ = e.load_raw()
    det = {m: [(r.benchmark, r.agent, r.task_id) for r in e.det_rows(m, sample).itertuples()] for m in MAIN}
    rows = []
    for code, label in FIXED:
        rows.append({"Attack": label, **{SHORT[m]: cell(*hidden(done, f"{code}@{m}", det[m])) for m in MAIN}})
    for i, k in enumerate(e.TAILORED_IDS, start=1):
        rows.append({"Attack": f"T{i}: {NAMES[k]}", **{SHORT[m]: cell(*hidden(done, f"T{i}G@{m}", det[m])) for m in MAIN}})
    tailored = pd.DataFrame(rows)
    tailored.to_csv(TAB / "exp_adaptive_tailored.csv", index=False)

    st = {}
    for f in sorted(EXP.glob("adaptive_state_*.json")):
        if "smoke" not in f.name and "structured" not in f.name:
            st.update(json.loads(f.read_text()))
    if (EXP / "adaptive_state.json").exists():
        st = {**json.loads((EXP / "adaptive_state.json").read_text()), **st}
    srows, trows = [], []
    for m in ("gemma4:31b", "qwen3.6:35b-a3b"):
        ms = st.get(m)
        if not ms or "folds" not in ms:
            continue
        k_all = n_all = 0
        for f, fs in sorted(ms["folds"].items()):
            if "test_hidden" not in fs:
                continue
            k_all += fs["test_hidden"]; n_all += fs["test_n"]
            curve = " → ".join(f"{r.get('best_so_far', 0):.0%}" for r in fs["rounds"])
            srows.append({"Evaluator": SHORT[m], "Fold": int(f) + 1, "Source of the final attack": fs["final_source"],
                          "Development success (best per round)": curve, "Held-out": cell(fs["test_hidden"], fs["test_n"])})
            trows.append({"Evaluator": SHORT[m], "Fold": int(f) + 1, "Source": fs["final_source"], "Text": fs["final"]})
        best_fixed = max(hidden(done, f"{c}@{m}", det[m])[0] / max(hidden(done, f"{c}@{m}", det[m])[1], 1) for c, _ in FIXED)
        best_tailored = max(hidden(done, f"T{i}G@{m}", det[m])[0] / max(hidden(done, f"T{i}G@{m}", det[m])[1], 1)
                            for i in range(1, len(e.TAILORED_IDS) + 1))
        srows.append({"Evaluator": SHORT[m], "Fold": "all", "Source of the final attack": "—",
                      "Development success (best per round)": f"best fixed {best_fixed:.0%}; best tailored (in sample) {best_tailored:.0%}",
                      "Held-out": cell(k_all, n_all)})
    search = pd.DataFrame(srows)
    search.to_csv(TAB / "exp_adaptive_search.csv", index=False)
    pd.DataFrame(trows).to_csv(TAB / "exp_adaptive_texts.csv", index=False)

    rep = ["# Adaptive attacks against the guided prompt", "",
           "Share of the side effects that the guided evaluator detects without attack and that the attack makes it miss "
           f"(n = {', '.join(f'{len(det[m])} for {SHORT[m]}' for m in MAIN)}). Wilson 95% intervals.", "",
           "## Fixed and tailored attacks", "", md_table(tailored), "",
           "## Automated search (two-fold cross-validation: each final attack is tested on trajectories it never saw)", "",
           md_table(search) if len(search) else "(not run)", ""]
    if trows:
        rep += ["## Final attack texts", ""] + [f"- **{r['Evaluator']}, fold {r['Fold']}** ({r['Source']}): {r['Text']!r}" for r in trows]
    (EXP / "adaptive_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
