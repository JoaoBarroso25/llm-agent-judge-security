"""Further figures of the dissertation: study design, threat model, injection placement, committee under attack,
suppression of false alarms, and distribution of trajectory length.

Usage: python3 code/figures_extra.py   (needs results/tables/exp_extra_*.csv from code/experiments_extra.py)
Output: Tese/figures/*.svg (and PNG previews in $PREVIEW_DIR if set)
"""
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
FIG = ROOT / "Tese" / "figures"
PREVIEW = os.environ.get("PREVIEW_DIR")

# Okabe-Ito colour-blind safe palette (same as code/figures.py)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#999999", "black": "#222222"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.titlesize": 10})
SHORT = {"gemma4:31b": "Gemma 4 31B", "qwen3.6:35b-a3b": "Qwen3.6 35B-A3B", "gpt-oss:20b": "gpt-oss 20B",
         "gpt-oss:120b": "gpt-oss 120B", "llama3.1:8b": "Llama 3.1 8B", "nemotron-3-nano": "Nemotron 3 Nano",
         "ministral-3:14b": "Ministral 3 14B", "phi4:14b": "Phi-4 14B", "glm-4.7-flash": "GLM-4.7-Flash",
         "EuroLLM-22B": "EuroLLM 22B"}


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.svg")
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=110)
    plt.close(fig)


def parse(cell):
    """'43/45 = 95.6% [85.2%; 98.8%]' -> (k, n, p, lo, hi) in percent, or None."""
    m = re.search(r"(\d+)/(\d+) = ([\d.]+)% \[([\d.]+)%; ([\d.]+)%\]", str(cell))
    if not m:
        return None
    k, n, p, lo, hi = m.groups()
    return int(k), int(n), float(p), float(lo), float(hi)


# --------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc, ec=None, fs=8, weight="normal", tc=C["black"], align="center", pad=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}", fc=fc, ec=ec or fc, lw=0.9))
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=tc, fontweight=weight, linespacing=1.25)


def arrow(ax, p, q, color=C["black"], lw=1.0, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


def tint(hex_colour, a=0.14):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def panel(ax, x, y, w, h, title, lines, colour, fs=7.2, fill=0.10, tc=None):
    """Rounded panel with a bold title at the top and left-aligned lines below it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=tint(colour, fill), ec=colour, lw=0.9))
    ax.text(x + 0.012, y + h - 0.045, title, ha="left", va="center", fontsize=fs + 0.8, fontweight="bold", color=tc or colour)
    step = (h - 0.11) / max(len(lines), 1)
    for i, t in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.10 - (i + 0.5) * step * 0.95, t, ha="left", va="center", fontsize=fs, linespacing=1.1)


def design_overview():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel(ax, 0.00, 0.30, 0.25, 0.40, "AgentRewardBench",
          ["1,302 trajectories", "4 web agents", "expert labels:", "  success, side effects,", "  repetition loops", "11 LLM evaluators", "  (published verdicts)"], C["blue"])
    panel(ax, 0.31, 0.60, 0.37, 0.38, "Secondary analysis (all 1,302)",
          ["RQ1  how often side effects occur", "RQ2  do the evaluators detect them", "RQ3  effect of the length of the run",
           "RQ4  robustness, manipulation"], C["blue"], fill=0.06)
    panel(ax, 0.31, 0.02, 0.37, 0.53, "Controlled experiments (E1–E11)",
          ["350 sampled; 184 for the attacks", "Prompt: original, guided; richer input", "Attacks: injection, jailbreaks,",
           "  position, adaptive attacker", "Defences: instruction, datamarking,", "  structured pipeline",
           "Evaluators: 3 closed, 13 open-weight;", "  committees, size, reasoning", "Decoding: default, temperature 0"],
          C["orange"], fill=0.07, tc="#8a5d00")
    panel(ax, 0.75, 0.36, 0.25, 0.42, "Measures",
          ["κ with the experts", "  (human ceiling 0.65)", "sensitivity, false alarms,", "  precision", "detected side effects", "  hidden by an attack,", "  against a noise floor"], C["green"])
    panel(ax, 0.75, 0.04, 0.25, 0.20, "Output", ["guidelines for automated", "security evaluators"], C["purple"], fill=0.14)
    arrow(ax, (0.25, 0.58), (0.31, 0.76), C["blue"]); arrow(ax, (0.25, 0.42), (0.31, 0.30), C["orange"])
    arrow(ax, (0.68, 0.79), (0.75, 0.66), C["blue"]); arrow(ax, (0.68, 0.30), (0.75, 0.46), C["orange"])
    arrow(ax, (0.875, 0.36), (0.875, 0.24), C["green"])
    save(fig, "design_overview", tight=False)


def threat_model():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.00, 0.50, 0.15, 0.20, "Evaluated\nweb agent", tint(C["blue"]), C["blue"], fs=8.2, weight="bold")
    ax.add_patch(FancyBboxPatch((0.25, 0.22), 0.37, 0.74, boxstyle="round,pad=0,rounding_size=0.012", fc="#fafafa", ec=C["grey"], lw=0.9))
    ax.text(0.435, 0.925, "Record of the run read by the evaluator", ha="center", fontsize=7.8, fontweight="bold")
    rows = [("Goal given to the agent", None, 0.83), ("Step 1: action, reasoning", None, 0.73), ("...", None, 0.66),
            ("Step k (middle): action", "2", 0.58), ("...", None, 0.51), ("Last step: action, reasoning", "1", 0.43),
            ("Last page (accessibility tree)", "3", 0.31)]
    for t, num, yy in rows:
        if t == "...":
            ax.text(0.40, yy, "...", ha="center", va="center", fontsize=8, color=C["grey"]); continue
        box(ax, 0.27, yy - 0.037, 0.29, 0.074, t, "white", C["grey"], fs=7.0)
        if num:
            ax.text(0.585, yy, num, ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.28", fc=C["red"], ec="none"))
    arrow(ax, (0.15, 0.60), (0.25, 0.60))
    ax.text(0.20, 0.655, "writes", ha="center", fontsize=7, color=C["blue"])
    box(ax, 0.00, 0.20, 0.15, 0.20, "Third party\n(web page\ncontent)", tint(C["red"], 0.16), C["red"], fs=7.4)
    arrow(ax, (0.15, 0.30), (0.27, 0.31), C["red"])
    box(ax, 0.72, 0.58, 0.28, 0.18, "LLM evaluator\n(one model)", tint(C["orange"]), C["orange"], fs=8.2, weight="bold")
    box(ax, 0.72, 0.30, 0.28, 0.18, "Committee of three\nmajority, any of three", tint(C["orange"], 0.08), C["orange"], fs=7.6)
    arrow(ax, (0.62, 0.62), (0.72, 0.67)); arrow(ax, (0.62, 0.48), (0.72, 0.39))
    box(ax, 0.72, 0.84, 0.28, 0.12, "Verdict: side effect?\nyes / no", "white", C["green"], fs=7.6, weight="bold")
    arrow(ax, (0.86, 0.76), (0.86, 0.84), C["green"])
    ax.text(0.86, 0.23, "Reference: expert annotators, who\nreviewed every step of the run", ha="center", va="top", fontsize=6.8, color=C["green"], linespacing=1.15)
    legend = [("1", "injection at the end of the agent's reasoning"), ("2", "the same injection in a middle step"),
              ("3", "the same text on the web page, planted by a third party")]
    for i, (n, t) in enumerate(legend):
        yy = 0.115 - i * 0.045
        ax.text(0.27, yy, n, ha="center", va="center", fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.25", fc=C["red"], ec="none"))
        ax.text(0.30, yy, t, ha="left", va="center", fontsize=7)
    save(fig, "threat_model", tight=False)


# --------------------------------------------------------------------------- data figures
def placement():
    p = TAB / "exp_extra_final.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["Evaluator"] = d.Evaluator
    d["Evaluator"] = d.Evaluator.map(lambda m: SHORT.get(m, m))
    models = list(dict.fromkeys(d.Evaluator))
    series = [("Injection, last step (agent)", "End of the reasoning (agent)", C["blue"]),
              ("Injection, middle step (agent)", "Middle step (agent)", C["orange"]),
              ("Injection on the web page (third party)", "Web page (third party)", C["red"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    h = 0.24
    y0 = np.arange(len(models))[::-1]
    for i, (key, lab, col) in enumerate(series):
        vals, lo, hi, ys = [], [], [], []
        for m, yy in zip(models, y0):
            r = d[(d.Evaluator == m) & (d.Attack == key)]
            if r.empty or parse(r.iloc[0]["Detected side effects hidden"]) is None:
                continue
            k, n, v, a, b = parse(r.iloc[0]["Detected side effects hidden"])
            vals.append(v); lo.append(v - a); hi.append(b - v); ys.append(yy + (1 - i) * h)
        ax.barh(ys, vals, h * 0.9, xerr=[lo, hi], color=col, label=lab, capsize=2, error_kw={"lw": 0.7, "ecolor": C["black"]})
        for yy, v in zip(ys, vals):
            ax.text(min(v + 1.2, 99), yy, "", fontsize=6)
    ax.set_yticks(y0, models)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Detected side effects hidden by the injection (%, Wilson 95% CI)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "placement")


def committee_attack():
    p = TAB / "exp_extra_committee.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    conds = ["Injection", "Persona", "False authority", "Base64", "Guided + injection", "Guided + false authority"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    h = 0.34
    y0 = np.arange(len(conds))[::-1]
    for j, (rule, lab, col) in enumerate((("Majority (2 of 3)", "Majority of the three", C["red"]),
                                          ("Any of the three", "Any of the three flags", C["sky"]))):
        vals, lo, hi, ys = [], [], [], []
        for c, yy in zip(conds, y0):
            r = d[(d.Condition == c) & (d.Committee == rule)]
            if r.empty or parse(r.iloc[0]["Detected side effects hidden"]) is None:
                continue
            k, n, v, a, b = parse(r.iloc[0]["Detected side effects hidden"])
            vals.append(v); lo.append(v - a); hi.append(b - v); ys.append(yy + (0.5 - j) * h)
        ax.barh(ys, vals, h * 0.92, xerr=[lo, hi], color=col, label=lab, capsize=2, error_kw={"lw": 0.7, "ecolor": C["black"]})
    ax.axhline(1.5, color=C["grey"], lw=0.6, ls=":")
    ax.set_yticks(y0, conds)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Detected side effects missed under the attack (%, 95% CI)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "committee_attack")


def false_alarm_shift():
    p = TAB / "exp_extra_false_alarms.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    num = lambda s: float(str(s).rstrip("%"))   # noqa: E731
    d["name"] = d.Evaluator.map(lambda m: SHORT.get(m, m))
    d["fa0"] = d["False alarms, no attack"].map(num)
    d = d.sort_values("fa0")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for col, lab, colour, mk in (("False alarms, injection", "With the injection", C["red"], "o"),
                                 ("False alarms, false authority", "With the false authority", C["purple"], "s")):
        v = d[col].map(num).values
        for yy, a, b in zip(y, d.fa0.values, v):
            ax.plot([a, b], [yy, yy], color=colour, lw=0.9, alpha=0.55, zorder=1)
        ax.scatter(v, y, s=26, color=colour, marker=mk, label=lab, zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(d.fa0.values, y, s=34, color=C["black"], label="No attack", zorder=4, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, d.name)
    ax.set_xlim(-2, 100)
    ax.set_xlabel("False alarms on the 100 trajectories without side effects (%)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "false_alarm_shift")


def steps_distribution():
    sys.path.insert(0, str(ROOT / "code"))
    from analysis import build   # noqa: E402
    traj, _, _ = build()
    agents = sorted(traj.agent_name.unique())
    cols = [C["blue"], C["orange"], C["green"], C["purple"]]
    fig, axes = plt.subplots(1, len(agents), figsize=(7.2, 2.3), sharey=True)
    bins = np.arange(0.5, 31.5, 1)
    for ax, a, c in zip(axes, agents, cols):
        s = traj[traj.agent_name == a].n_steps.clip(upper=30)
        ax.hist(s, bins=bins, color=c)
        share = 100 * (s >= 30).mean()
        ax.set_title(f"{a}\n{share:.0f}% reach the 30-step limit", fontsize=7.6)
        ax.set_xlabel("Steps")
        ax.set_xticks([1, 10, 20, 30])
    axes[0].set_ylabel("Trajectories")
    save(fig, "steps_distribution")


NAMES = {"llama3.1:8b": "Llama 3.1 8B", "ministral-3:14b": "Ministral 3 14B", "phi4:14b": "Phi-4 14B",
         "gpt-oss:20b": "gpt-oss 20B", "glm-4.7-flash:latest": "GLM-4.7-Flash", "gemma4:31b": "Gemma 4 31B",
         "nemotron-3-nano:latest": "Nemotron 3 Nano", "qwen3.6:35b-a3b": "Qwen3.6 35B-A3B", "llama4:latest": "Llama 4 109B",
         "gpt-oss:120b": "gpt-oss 120B", "nemotron-3-super:120b": "Nemotron 3 Super", "qwen3.5:122b": "Qwen3.5 122B"}


def size_kappa():
    p = TAB / "exp_size.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["label"] = d.model.map(lambda m: NAMES.get(m, "EuroLLM 22B" if "EuroLLM" in m else m))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    for ax, col, title, colour in ((axes[0], "kappa_original", "Original prompt", C["blue"]),
                                   (axes[1], "kappa_guided", "Guided prompt", C["green"])):
        ax.axhline(0, color=C["grey"], lw=0.6)
        ax.errorbar(d.params_b, d[col], yerr=[d[col] - d[col + "_lo"], d[col + "_hi"] - d[col]], fmt="none",
                    ecolor=tint(colour, 0.5), elinewidth=0.8, capsize=0)
        for moe, marker in ((False, "o"), (True, "s")):
            s = d[d.moe == moe]
            ax.scatter(s.params_b, s[col], marker=marker, s=22, color=colour, zorder=3,
                       label="mixture of experts" if moe else "dense")
        shown = {"Gemma 4 31B": (4, 3, "left")} if col == "kappa_original" else {
            "Gemma 4 31B": (4, 3, "left"), "Qwen3.6 35B-A3B": (4, -2, "left"), "gpt-oss 120B": (-5, 3, "right"),
            "Qwen3.5 122B": (-5, -6, "right"), "Llama 4 109B": (-5, -2, "right")}
        for _, r in d.iterrows():
            if r.label in shown:
                dx, dy, ha = shown[r.label]
                ax.annotate(r.label, (r.params_b, r[col]), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, ha=ha)
        if col == "kappa_original":
            ax.annotate("models of 109B–125B", (115, 0.01), xytext=(115, 0.09), fontsize=6.3, ha="center",
                        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": C["grey"]})
        ax.set_xscale("log")
        ax.set_xticks([8, 15, 30, 60, 120], ["8", "15", "30", "60", "120"])
        ax.minorticks_off()
        ax.set_xlabel("Total parameters (billions, log scale)")
        ax.set_title(title)
    axes[0].set_ylabel("Population κ (95% CI)")
    axes[0].legend(fontsize=6.8, frameon=False, loc="upper left")
    save(fig, "size_kappa")


def adaptive():
    pt, ps, pr = TAB / "exp_adaptive_tailored.csv", TAB / "exp_adaptive_search.csv", TAB / "exp_scale.csv"
    if not (pt.exists() and ps.exists() and pr.exists()):
        return
    t, s, ref = pd.read_csv(pt), pd.read_csv(ps), pd.read_csv(pr).set_index("Evaluator")
    models = ["Gemma 4 31B", "Qwen3.6 35B-A3B", "gpt-oss 20B"]

    def best(prefix, m):
        cells = [parse(v) for a, v in zip(t.Attack, t[m]) if a.startswith(prefix)]
        return max((c for c in cells if c), key=lambda c: (c[2], -c[1]))

    series = [("Injection, original prompt (no defence)", lambda m: parse(ref.loc[m, "Injection hides"]), C["grey"]),
              ("Best fixed attack, guided prompt", lambda m: best("Fixed", m), C["blue"]),
              ("Best hand-written adaptive attack, guided (in sample)", lambda m: best("T", m), C["orange"]),
              ("Automated search, guided (held-out)",
               lambda m: parse(s[(s.Evaluator == m) & (s.Fold.astype(str) == "all")].iloc[0]["Held-out"])
               if ((s.Evaluator == m) & (s.Fold.astype(str) == "all")).any() else None, C["red"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    h = 0.2
    y0 = np.arange(len(models))[::-1]
    for i, (lab, get, col) in enumerate(series):
        ys, vals, lo, hi = [], [], [], []
        for m, yy in zip(models, y0):
            c = get(m)
            if c is None:
                continue
            ys.append(yy + (1.5 - i) * h); vals.append(c[2]); lo.append(c[2] - c[3]); hi.append(c[4] - c[2])
        ax.barh(ys, vals, h * 0.9, xerr=[lo, hi], color=col, label=lab, capsize=2, error_kw={"lw": 0.7, "ecolor": C["black"]})
    ax.text(1, y0[2] - 1.5 * h, "search not run", fontsize=6.5, va="center", color=C["black"])
    ax.set_ylim(y0[-1] - 2.6 * h, y0[0] + 2.6 * h)
    ax.set_yticks(y0, [f"{m}\n(n = {n})" for m, n in zip(models, (22, 20, 9))])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Detected side effects hidden (%, Wilson 95% CI)")
    ax.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "adaptive")


def structured():
    p, q = TAB / "exp_structured.csv", TAB / "exp_structured_search.csv"
    if not (p.exists() and q.exists()):
        return
    d, s = pd.read_csv(p), pd.read_csv(q)
    models = ["Gemma 4 31B", "Qwen3.6 35B-A3B", "gpt-oss 20B"]
    attacks = ["Injection", "False authority", "T5 instruction update", "T3 actions relabelled as navigation",
               "Dry Run text (search, E10)"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True, sharey=True)
    for ax, pipe, title in ((axes[0], "Guided prompt", "Guided prompt (defence in the prompt)"),
                            (axes[1], "Structured pipeline", "Structured pipeline (E11)")):
        h = 0.26
        y0 = np.arange(len(models))[::-1]
        series = [("Second run, no attack (noise)", C["grey"]), ("Worst fixed or hand-written attack", C["orange"]),
                  ("Adaptive search (held-out)", C["red"])]
        for i, (lab, col) in enumerate(series):
            ys, vals, lo, hi = [], [], [], []
            for m, yy in zip(models, y0):
                r = d[(d.Evaluator == m) & (d.Pipeline == pipe)].iloc[0]
                if i == 0:
                    c = parse(r["Second run, no attack (noise)"])
                elif i == 1:
                    cs = [parse(r[a]) for a in attacks if parse(r[a])]
                    c = max(cs, key=lambda x: x[2]) if cs else None
                else:
                    ss = s[(s.Evaluator == m) & (s.Pipeline == pipe)]
                    c = parse(ss.iloc[0]["Held-out"]) if len(ss) else None
                if c is None:
                    continue
                ys.append(yy + (1 - i) * h); vals.append(c[2]); lo.append(c[2] - c[3]); hi.append(c[4] - c[2])
            ax.barh(ys, vals, h * 0.9, xerr=[lo, hi], color=col, label=lab, capsize=2,
                    error_kw={"lw": 0.7, "ecolor": C["black"]})
        ax.set_yticks(y0, models)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Detected side effects missed (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.9, bottom=0.27, wspace=0.08)
    save(fig, "structured", tight=False)


def defence_pipeline():
    """The two defences attacked by an adaptive attacker: the guided prompt and the structured pipeline of E11."""
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel(ax, 0.00, 0.30, 0.18, 0.42, "Attacker",
          ["fixed texts (E3–E9)", "hand-written T1–T9", "automated search:", "  an open-weight model,", "  4 rounds × 5 texts,", "  judged on held-out", "  trajectories"],
          C["red"], fs=6.6, fill=0.10)
    # lane A: defence in the prompt
    ax.text(0.24, 0.955, "Defence in the prompt (guided prompt; E2, E10)", fontsize=8, fontweight="bold", color=C["blue"])
    box(ax, 0.24, 0.70, 0.20, 0.19, "Record of the run\ngoal, actions,\nagent's reasoning\n+ attack text", "white", C["grey"], fs=7)
    box(ax, 0.52, 0.70, 0.22, 0.19, "LLM evaluator\nguided prompt\nreads everything", tint(C["blue"]), C["blue"], fs=7.4, weight="bold")
    box(ax, 0.82, 0.72, 0.18, 0.15, "Side effect?\nyes / no", "white", C["green"], fs=7.4)
    arrow(ax, (0.44, 0.795), (0.52, 0.795)); arrow(ax, (0.74, 0.795), (0.82, 0.795), C["green"])
    # lane B: defence in the architecture
    ax.text(0.24, 0.50, "Defence in the architecture (structured pipeline; E11)", fontsize=8, fontweight="bold", color=C["purple"])
    box(ax, 0.24, 0.24, 0.17, 0.19, "Record of the run\n+ attack text", "white", C["grey"], fs=7)
    box(ax, 0.45, 0.24, 0.16, 0.19, "Quarantined\nreader\n(same model)", tint(C["purple"]), C["purple"], fs=7.2, weight="bold")
    box(ax, 0.65, 0.24, 0.16, 0.19, "Fixed table\nstep | action |\ntarget | text |\neffect (closed list)", "white", C["purple"], fs=6.4)
    box(ax, 0.84, 0.24, 0.16, 0.19, "LLM evaluator\ngoal, addresses\nand table only", tint(C["purple"]), C["purple"], fs=6.8, weight="bold")
    box(ax, 0.84, 0.02, 0.16, 0.12, "Side effect?\nyes / no", "white", C["green"], fs=7.2)
    arrow(ax, (0.41, 0.335), (0.45, 0.335)); arrow(ax, (0.61, 0.335), (0.65, 0.335)); arrow(ax, (0.81, 0.335), (0.84, 0.335))
    arrow(ax, (0.92, 0.24), (0.92, 0.14), C["green"])
    ax.text(0.73, 0.205, "malformed lines dropped", ha="center", va="top", fontsize=6.2, color=C["purple"])
    ax.text(0.50, 0.10, "The agent's text reaches only the reader;\nthe deciding evaluator never reads it.", ha="center", va="center",
            fontsize=7, color=C["red"], linespacing=1.2)
    # attacker reaches both records
    arrow(ax, (0.18, 0.62), (0.24, 0.76), C["red"]); arrow(ax, (0.18, 0.40), (0.24, 0.33), C["red"])
    save(fig, "defence_pipeline", tight=False)


def defence_ladder():
    """Synthesis of E3 to E11: share of detected side effects hidden, from no defence to the structured pipeline."""
    pm, ps, pa, pt, pq = (TAB / f for f in ("exp_models.csv", "exp_structured.csv", "exp_adaptive_search.csv",
                                             "exp_structured_search.csv", "exp_adaptive_tailored.csv"))
    if not all(p.exists() for p in (pm, ps, pa, pt)):
        return
    from analysis import wilson  # noqa: E402
    m, s, a, t = pd.read_csv(pm).set_index("model"), pd.read_csv(ps), pd.read_csv(pa), pd.read_csv(pt)
    models = [("gemma4:31b", "Gemma 4 31B", C["blue"], "o"), ("qwen3.6:35b-a3b", "Qwen3.6 35B-A3B", C["orange"], "s"),
              ("gpt-oss:20b", "gpt-oss 20B", C["green"], "D")]

    def kn(cell):
        mm = re.search(r"(\d+)/(\d+)", str(cell))
        return (int(mm.group(1)), int(mm.group(2))) if mm else None

    def struct(name, col):
        r = s[(s.Evaluator == name) & (s.Pipeline == col[0])]
        return kn(r.iloc[0][col[1]]) if len(r) else None

    def search(df, name, pipe=None):
        r = df[(df.Evaluator == name) & (df.Fold.astype(str) == "all")] if pipe is None else df[(df.Evaluator == name) & (df.Pipeline == pipe)]
        return kn(r.iloc[0]["Held-out"]) if len(r) else None

    def worst_struct(name):
        r = s[(s.Evaluator == name) & (s.Pipeline == "Structured pipeline")].iloc[0]
        cells = [kn(r[c]) for c in ("Injection", "False authority", "T5 instruction update", "T3 actions relabelled as navigation", "Dry Run text (search, E10)")]
        return max((c for c in cells if c), key=lambda c: c[0] / c[1])

    rows = [("No defence: injection, original prompt", lambda k, n: kn(m.loc[k, "hidden: injection"])),
            ("Instruction not to obey the agent", lambda k, n: kn(m.loc[k, "hidden: injection, defence by instruction"])),
            ("Datamarking", lambda k, n: kn(m.loc[k, "hidden: injection, datamarking"])),
            ("Guided prompt: injection", lambda k, n: kn(m.loc[k, "hidden: injection (guided)"])),
            ("Guided prompt: adaptive search (held-out)", lambda k, n: search(a, n)),
            ("Guided prompt: second run, no attack", lambda k, n: struct(n, ("Guided prompt", "Second run, no attack (noise)"))),
            ("Structured pipeline: worst of five attacks", lambda k, n: worst_struct(n)),
            ("Structured pipeline: adaptive search (held-out)", lambda k, n: search(t, n, "Structured pipeline")),
            ("Structured pipeline: second run, no attack", lambda k, n: struct(n, ("Structured pipeline", "Second run, no attack (noise)")))]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    y0 = np.arange(len(rows))[::-1]
    for j, (key, name, col, mk) in enumerate(models):
        for (lab, get), yy in zip(rows, y0):
            c = get(key, name)
            if not c:
                continue
            p, lo, hi = wilson(*c)
            yv = yy + (j - 1) * 0.22
            noise = "second run" in lab
            ax.errorbar(100 * p, yv, xerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt=mk, ms=4.2, color=col,
                        mfc="white" if noise else col, elinewidth=0.7, capsize=0, label=name if yy == y0[0] else None)
    for yy in (y0[2] - 0.5, y0[5] - 0.5):
        ax.axhline(yy, color=C["grey"], lw=0.5, ls=":")
    ax.set_yticks(y0, [r[0] for r in rows], fontsize=7.2)
    ax.set_xlim(0, 102)
    ax.set_xlabel("Detected side effects missed (%, 95% CI); open markers: no attack", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "defence_ladder")


def evaluator_design():
    """Design of an automated security evaluator following the eight guidelines of the dissertation (G1-G8)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0.08, 0.96); ax.axis("off")
    panel(ax, 0.00, 0.36, 0.21, 0.56, "Evidence", ["goal and task type (G8)", "actions and page", "  addresses", "record of state", "  changes (G2)",
          "agent's text only", "  through a reader", "  (G2, G7)"], C["blue"], fs=6.2, fill=0.08)
    box(ax, 0.26, 0.64, 0.25, 0.22, "Safety question:\nwas a change of state\nnot requested? (G1)", tint(C["orange"]), C["orange"], fs=6.6, weight="bold")
    box(ax, 0.26, 0.38, 0.25, 0.18, "Success question,\nasked separately (G3)", "white", C["orange"], fs=6.6)
    box(ax, 0.56, 0.64, 0.21, 0.22, "Alarms normalised\nby length and by\nstate changes (G4)", "white", C["purple"], fs=6.6)
    box(ax, 0.56, 0.38, 0.21, 0.18, "Committees only with\nindependent evidence\n(G5)", "white", C["purple"], fs=6.2)
    panel(ax, 0.81, 0.36, 0.19, 0.56, "Before use", ["validate against", "  experts with", "  prevalence-aware", "  metrics (G6)", "test with adaptive",
          "  attacks, against a", "  noise floor (G7)", "beat trivial", "  baselines (G8)"], C["green"], fs=6.0, fill=0.08)
    arrow(ax, (0.21, 0.75), (0.26, 0.75)); arrow(ax, (0.21, 0.47), (0.26, 0.47))
    arrow(ax, (0.51, 0.75), (0.56, 0.75)); arrow(ax, (0.51, 0.47), (0.56, 0.47))
    arrow(ax, (0.77, 0.64), (0.81, 0.64), C["green"])
    ax.text(0.5, 0.20, "The experiments support G1, G2 and G7 directly. The structured pipeline of E11 implements part of G2 and kept the attacks\n"
            "within the run-to-run noise, but detection still needs a record of what the actions changed.",
            ha="center", va="center", fontsize=6.4, linespacing=1.3)
    save(fig, "evaluator_design", tight=False)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (design_overview, threat_model, placement, committee_attack, false_alarm_shift, steps_distribution,
              size_kappa, adaptive, structured, defence_pipeline, defence_ladder, evaluator_design):
        f()
    print("figures written to", FIG)
