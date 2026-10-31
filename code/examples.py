"""Extracts qualitative examples for Chapter 5 (no model is called).

  - side effects annotated by the expert and missed by all 11 LLM evaluators;
  - false alarms on side effects: the side-effect part of the evaluator's reasoning.

Usage: python3 code/examples.py   (run code/analysis.py first)
Output: results/examples.md
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import DATA, JUDGES, NON_LLM, OUT, STEP_RE  # noqa: E402

LLM = [j for j in JUDGES if j not in NON_LLM]


def load(b, a, j, t):
    p = DATA / "judgments" / b / a / j / f"{t}.json"
    return json.loads(p.read_text()) if p.exists() else None


def actions(js):
    content = ""
    for m in js["chat_messages"]["regular"]:
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, list):
                c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
            content += c
    return [s[2].strip() for s in STEP_RE.findall(content + "-----")]


def side_reasoning(js):
    msg = js["response"]["choices"][0]["message"]["content"] or ""
    m = re.search(r"Question 2[^\n]*\n(.*?)(?:Question 3|$)", msg, re.S)
    txt = m.group(1) if m else msg
    return " ".join(txt.split())[:700]


def main():
    traj = pd.read_csv(OUT / "tables" / "trajectories.csv", dtype={"task_id": str})
    out = ["# Qualitative examples\n"]
    pos = traj[traj.side == 1]
    out.append("\n## Side effects missed by all 11 LLM evaluators\n")
    for _, r in pos.iterrows():
        verdicts = []
        for j in LLM:
            js = load(r.benchmark, r.agent, j, r.task_id)
            m = js["response"]["choices"][0]["message"]["content"] if js else ""
            v = re.search(r"<side>(.*?)</side>", m or "")
            verdicts.append((v.group(1).strip().lower() if v else "?"))
        if any(v == "yes" for v in verdicts):
            continue
        js = load(r.benchmark, r.agent, "gpt-4o-noscreen", r.task_id)
        acts = actions(js)
        out.append(f"\n### {r.bench_name} / {r.agent_name} / {r.task_id}\n")
        out.append(f"- Goal: {r.goal}\n- Expert: success={r.success}, side=1, loop={r.loop}; steps={r.n_steps}\n")
        out.append("- Actions: " + " | ".join(a[:120] for a in acts[:30]) + "\n")
        out.append(f"- GPT-4o (AXTree) on side effects: {side_reasoning(js)}\n")
    neg = traj[(traj.side == 0)]
    out.append("\n## False alarms: reasoning of GPT-4o (AXTree) on trajectories without side effects (random sample)\n")
    sample = neg.sample(12, random_state=2026)
    for _, r in sample.iterrows():
        js = load(r.benchmark, r.agent, "gpt-4o-noscreen", r.task_id)
        m = js["response"]["choices"][0]["message"]["content"] or ""
        v = re.search(r"<side>(.*?)</side>", m)
        if not v or v.group(1).strip().lower() != "yes":
            continue
        out.append(f"\n### {r.bench_name} / {r.agent_name} / {r.task_id} (steps={r.n_steps}, expert loop={r.loop}, success={r.success})\n")
        out.append(f"- Goal: {r.goal}\n- Reasoning on side effects: {side_reasoning(js)}\n")
    (OUT / "examples.md").write_text("".join(out), encoding="utf-8")
    print("written", OUT / "examples.md")


if __name__ == "__main__":
    main()
