"""Analysis of E11, the structural defence (code/structured_defence.py).

Compares, for the three main open-weight models on the attack subset, the structured pipeline (quarantined reader +
deciding evaluator that never sees the agent's text) with the original and the guided prompt: detection without attack,
and the share of detected side effects hidden by each attack. Also measures whether the attacks act on the reader:
the mean number of state-changing lines in its tables for the trajectories with side effects.

Output: results/experiments/structured_report.md, results/tables/exp_structured.csv, exp_structured_reader.csv
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import OUT, TAB, md_table, pct, wilson  # noqa: E402
import experiments as e  # noqa: E402
import experiments_scale_analysis as sa  # noqa: E402
import structured_defence as sd  # noqa: E402

EXP = OUT / "experiments"
MAIN = ["gemma4:31b", "qwen3.6:35b-a3b", "gpt-oss:20b"]
SHORT = {"gemma4:31b": "Gemma 4 31B", "qwen3.6:35b-a3b": "Qwen3.6 35B-A3B", "gpt-oss:20b": "gpt-oss 20B"}
LEAK = re.compile(r"evaluator|audit|override|side effect|no unintended|preview|dry run|verified|administrator|developer mode", re.I)
CHANGE = {"submit/save", "send/post/comment", "create", "edit", "delete", "add to cart/wishlist", "change setting"}
# attack of E11 -> the same text against the guided prompt (reference), and its label
ATTACKS = [("SO2", "G2", "Second run, no attack (noise)"), ("SA2", "A4", "Injection"), ("SJ2", "J2G", "False authority"), ("ST5", "T5G", "T5 instruction update"),
           ("ST3", "T3G", "T3 actions relabelled as navigation"), ("SDR", None, "Dry Run text (search, E10)")]


def cell(k, n):
    if not n:
        return "—"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(p)} [{pct(lo)}; {pct(hi)}]"


def baseline(r, model, base):
    d = sa.cell(r, model, base)
    if d.empty:
        return "—", "—", "—"
    sens, fa, k, lo, hi = sa.metrics(d.reset_index())
    return pct(sens), pct(fa), f"{k:.2f} [{lo:.2f}; {hi:.2f}]"


def main():
    r = sa.load()
    sample = e.make_sample()
    rows = []
    for m in MAIN:
        for base, lab in (("O", "Original prompt"), ("G", "Guided prompt"), ("SO", "Structured pipeline")):
            s, f, k = baseline(r, m, base)
            row = {"Evaluator": SHORT[m], "Pipeline": lab, "Sensitivity": s, "False alarms": f, "κ": k}
            for code, ref, alab in ATTACKS:
                if base == "SO":
                    row[alab] = sa.hidden(r, m, "SO", code)
                elif base == "G" and ref:
                    if ref.startswith("T"):    # tailored attacks were run on the side effects detected by G
                        det = [(x.benchmark, x.agent, x.task_id) for x in e.det_rows(m, sample).itertuples()]
                        done, _ = e.load_raw()
                        got = [done[(f"{ref}@{m}",) + k] for k in det if (f"{ref}@{m}",) + k in done]
                        row[alab] = cell(sum(1 for g in got if (g.get("side") or 0) == 0), len(got))
                    else:
                        row[alab] = sa.hidden(r, m, "G", ref)
                elif base == "O" and code in ("SA2", "SJ2"):
                    row[alab] = sa.hidden(r, m, "O", {"SA2": "A2", "SJ2": "J2"}[code])
                else:
                    row[alab] = "—"
            rows.append(row)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(TAB / "exp_structured.csv", index=False)

    # automated search (E10 design) against the guided prompt and against the structured pipeline, held-out folds
    srows = []
    for m in ("qwen3.6:35b-a3b", "gpt-oss:20b", "gemma4:31b"):
        for lab, f in (("Guided prompt", EXP / f"adaptive_state_{m.replace(':', '_')}.json"),
                       ("Structured pipeline", EXP / f"adaptive_state_{m.replace(':', '_')}_structured.json")):
            if not f.exists():
                continue
            st = json.loads(f.read_text()).get(m, {})
            folds = [v for _, v in sorted(st.get("folds", {}).items()) if "test_hidden" in v]
            if len(folds) < 2:
                continue
            k, n = sum(v["test_hidden"] for v in folds), sum(v["test_n"] for v in folds)
            srows.append({"Evaluator": SHORT[m], "Pipeline": lab,
                          "Development (best after each round, per fold)": "; ".join(
                              " → ".join(f"{x.get('best_so_far', 0):.0%}" for x in v["rounds"]) for v in folds),
                          "Final attacks": ", ".join(v["final_source"] for v in folds), "Held-out": cell(k, n)})
            if lab == "Structured pipeline":
                for i, v in enumerate(folds, 1):
                    srows[-1][f"Text, fold {i}"] = v["final"]
    search_t = pd.DataFrame(srows)
    search_t.to_csv(TAB / "exp_structured_search.csv", index=False)

    # does the attack act on the reader? state-changing lines per trajectory with a side effect
    reader = sd.load_reader()
    pos = {(x.benchmark, x.agent, x.task_id) for x in sample[sample.side == 1].itertuples()}
    rrows = []
    for m in MAIN:
        row = {"Evaluator": SHORT[m]}
        for code, _, alab in [("SO", None, "No attack")] + ATTACKS:
            vals, empty, leak = [], 0, 0
            for (key, b, a, t), rec in reader.items():
                if key != f"{code}@{m}" or (b, a, t) not in pos:
                    continue
                n_steps = len(sd.split_record(e.original_prompt(b, a, t)[1])[1])
                tab = sd.table_from(rec["text"], n_steps)
                empty += not tab
                leak += any(LEAK.search(line) for line in tab)   # words of the attacks that reached the table
                vals.append(sum(1 for line in tab if line.rsplit("| ", 1)[-1] in CHANGE))
            row[alab] = f"{sum(vals) / len(vals):.2f} (n = {len(vals)}, empty {empty}, attack words {leak})" if vals else "—"
        rrows.append(row)
    reader_t = pd.DataFrame(rrows)
    reader_t.to_csv(TAB / "exp_structured_reader.csv", index=False)

    rep = ["# E11: structural defence (quarantined reader + evaluator that never sees the agent's text)", "",
           "Attack subset (84 trajectories with side effects, 100 without). κ re-weighted to the population. Attack "
           "columns: share of the side effects detected by the pipeline without attack that the attack makes it miss "
           "(for the guided prompt, T5 and T3 were run on the side effects it detects).", "", md_table(main_t), "",
           "## State-changing lines in the reader's tables (trajectories with side effects)", "", md_table(reader_t), "",
           "## Automated search by an attacker who knows the pipeline (two-fold cross-validation, held-out)", "",
           md_table(search_t.drop(columns=[c for c in search_t.columns if c.startswith("Text")])) if len(search_t) else "(not run)",
           ""] + [f"- {r['Evaluator']}, {c}: {r[c]!r}" for _, r in search_t.iterrows() for c in search_t.columns
                  if c.startswith("Text") and isinstance(r[c], str)]
    (EXP / "structured_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
