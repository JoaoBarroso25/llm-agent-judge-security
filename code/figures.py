"""Figures of the dissertation, generated from results/tables/*.csv.

Usage: python3 code/figures.py   (run code/analysis.py first)
Output: Tese/figures/*.svg
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
FIG = ROOT / "Tese" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colour-blind safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#999999", "black": "#222222"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "axes.titlesize": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)


def prevalence():
    d = pd.read_csv(TAB / "rq1_prevalence.csv")
    d = d[d.benchmark != "All"]
    benches = sorted(d.benchmark.unique())
    agents = sorted(d.agent.unique())
    colors = [C["blue"], C["orange"], C["green"], C["purple"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, dim, title in ((axes[0], "side", "Side effects"), (axes[1], "loop", "Repetition loops")):
        w = 0.8 / len(agents)
        x = np.arange(len(benches))
        for i, a in enumerate(agents):
            vals = [d[(d.benchmark == b) & (d.agent == a)][f"{dim}_rate"].mean() * 100 for b in benches]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=a, color=colors[i])
        ax.set_xticks(x, benches, rotation=15)
        ax.set_ylabel("% of trajectories (expert label)")
        ax.set_title(title)
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "prevalence_by_benchmark")


def kappa_by_dimension():
    d = pd.read_csv(TAB / "rq2_judges.csv")
    ceil = pd.read_csv(TAB / "human_ceiling.csv").set_index("dimension")
    order = d[d.dimension == "success"].sort_values("kappa").judge_name.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for dim, col, mk, lab in (("success", C["blue"], "o", "Task success"),
                              ("loop", C["green"], "s", "Repetition loops"),
                              ("side", C["red"], "D", "Side effects")):
        dd = d[d.dimension == dim].set_index("judge_name").loc[order]
        ax.errorbar(dd.kappa, y, xerr=[dd.kappa - dd.kappa_lo, dd.kappa_hi - dd.kappa],
                    fmt=mk, color=col, ms=4, capsize=2, lw=0.8, label=lab)
        k = pd.to_numeric(ceil.loc[dim, "kappa"], errors="coerce")
        if not np.isnan(k):
            ax.axvline(k, color=col, ls="--", lw=0.8, alpha=0.7)
    ax.set_yticks(y, order)
    ax.set_xlabel("Cohen's κ with the expert label (95% bootstrap CI)\ndashed: human ceiling")
    ax.axvline(0, color=C["grey"], lw=0.6)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    save(fig, "kappa_by_dimension")


def sens_fa():
    d = pd.read_csv(TAB / "rq2_judges.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True)
    for ax, dim, title in ((axes[0], "side", "Side effects"), (axes[1], "loop", "Repetition loops")):
        dd = d[d.dimension == dim]
        ax.scatter(dd.false_alarm * 100, dd.sensitivity * 100, color=C["red"] if dim == "side" else C["green"], s=18)
        for _, r in dd.iterrows():
            ax.annotate(r.judge_name.replace(" (", "\n("), (r.false_alarm * 100, r.sensitivity * 100),
                        fontsize=5, xytext=(3, 2), textcoords="offset points")
        ax.plot([0, 100], [0, 100], color=C["grey"], lw=0.6, ls=":")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("False alarms (% of expert negatives flagged)")
        ax.set_title(title)
    axes[0].set_ylabel("Sensitivity (% of expert positives flagged)")
    save(fig, "sensitivity_false_alarms")


def by_steps():
    occ = pd.read_csv(TAB / "rq3_occurrence_by_steps.csv")
    det = pd.read_csv(TAB / "rq3_detection_by_steps.csv")
    x = np.arange(len(occ))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(x, occ.side_rate * 100, "D-", color=C["red"], label="Side effects")
    axes[0].plot(x, occ.loop_rate * 100, "s-", color=C["green"], label="Loops")
    axes[0].plot(x, occ.success_rate * 100, "o-", color=C["blue"], label="Success")
    axes[0].set_title("Occurrence (expert labels)")
    axes[0].set_ylabel("% of trajectories")
    axes[1].plot(x, det.side_sensitivity * 100, "D-", color=C["red"], label="Side effects: sensitivity")
    axes[1].plot(x, det.side_false_alarm * 100, "D--", color=C["red"], alpha=0.6, label="Side effects: false alarms")
    axes[1].plot(x, det.loop_sensitivity * 100, "s-", color=C["green"], label="Loops: sensitivity")
    axes[1].plot(x, det.loop_false_alarm * 100, "s--", color=C["green"], alpha=0.6, label="Loops: false alarms")
    axes[1].set_title("Detection by the 11 LLM judges")
    for ax in axes:
        ax.set_xticks(x, occ.steps)
        ax.set_xlabel("Number of steps")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=6.5, frameon=False)
    save(fig, "by_steps")


def false_alarm_conditions():
    d = pd.read_csv(TAB / "rq4_false_alarms_by_condition.csv")
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    cols = [C["sky"], C["blue"], C["sky"], C["blue"], C["sky"], C["blue"]]
    ax.barh(np.arange(len(d)), d.rate * 100, color=cols)
    ax.set_yticks(np.arange(len(d)), d.iloc[:, 0])
    ax.invert_yaxis()
    ax.set_xlabel("False alarms on side effects (% of pairs, expert: no side effect)")
    save(fig, "false_alarms_conditions")


def detected_by_k():
    d = pd.read_csv(TAB / "rq4_side_positives_detected_by_k.csv")
    d.columns = ["k", "n"]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.bar(d.k, d.n, color=C["red"])
    ax.set_xlabel("Number of LLM judges that flag the side effect")
    ax.set_ylabel("Trajectories with an expert-\nannotated side effect")
    ax.set_xticks(d.k)
    save(fig, "side_detected_by_k")


if __name__ == "__main__":
    prevalence()
    kappa_by_dimension()
    sens_fa()
    by_steps()
    false_alarm_conditions()
    detected_by_k()
    print("figures written to", FIG)


def justifications():
    d = pd.read_csv(TAB / "ext_justifications_summary.csv")
    cats = ["repetition only", "both", "state change only", "neither"]
    cols = [C["green"], C["sky"], C["red"], C["grey"]]
    labels = ["False alarms\n(expert: no side effect)", "True detections\n(expert: side effect)"]
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    left = np.zeros(len(d))
    for c, col in zip(cats, cols):
        v = d[c + "_rate"].values * 100
        ax.barh(labels, v, left=left, color=col, label=c)
        for i, (l, x) in enumerate(zip(left, v)):
            if x > 6:
                ax.text(l + x / 2, i, f"{x:.0f}%", ha="center", va="center", fontsize=7, color="white")
        left += v
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("% of side-effect alarms, by cue found in the evaluator's justification")
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    save(fig, "justifications")


def cost():
    d = pd.read_csv(TAB / "ext_cost.csv").sort_values("alarms_per_true")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y = np.arange(len(d))
    ax.barh(y, d.alarms_per_true, color=C["blue"])
    ax.axvline(1301 / 84, color=C["red"], ls="--", lw=1)
    ax.text(1301 / 84 + 0.2, len(d) - 0.6, "random review", color=C["red"], fontsize=7)
    ax.set_yticks(y, d.judge_name)
    ax.invert_yaxis()
    ax.set_xlabel("Alarms to review per real side effect found")
    save(fig, "cost_of_oversight")


def forest():
    d = pd.read_csv(TAB / "ext_success_effect_by_judge.csv").sort_values("OR")
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y = np.arange(len(d))
    ax.errorbar(d.OR, y, xerr=[d.OR - d.lo, d.hi - d.OR], fmt="o", color=C["blue"], ms=4, capsize=2, lw=0.8)
    ax.axvline(1, color=C["grey"], lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, d.judge)
    ax.set_xlabel("Odds ratio of a false alarm when the evaluator\njudges the task successful (log scale)")
    save(fig, "success_effect_by_judge")


justifications()
cost()
forest()


def surface_matched():
    """Third round: each evaluator against surface and length rules at the same flag rate."""
    d = pd.read_csv(TAB / "t3_matched_rate.csv").sort_values("_kj")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r._kj, r._ks], [i, i], color=C["grey"], lw=1, zorder=1)
    ax.scatter(d._kj, y, color=C["red"], label="LLM evaluator", zorder=3)
    ax.scatter(d._ks, y, color=C["blue"], marker="s", label="Surface rule, same flag rate", zorder=3)
    ax.scatter(d._kl, y, color=C["grey"], marker="|", s=80, label="Length rule, same flag rate", zorder=2)
    ax.axvline(0, color=C["black"], lw=0.6)
    ax.set_yticks(y, [f"{e} ({r})" for e, r in zip(d.evaluator, d["flag rate"])], fontsize=7.5)
    ax.set_xlabel("Cohen's κ with the expert side-effect label")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "surface_matched")


def ending():
    """Third round: how the run ends, expert side effects against evaluator alarms."""
    d = pd.read_csv(TAB / "t3_ending.csv")
    order = ["answer to the user", "no final message", "declares the task infeasible"]
    d = d.set_index("how the run ends").loc[order]
    exp = [float(re.search(r"= ([\d.]+)%", s).group(1)) for s in d["expert side effects"]]
    lo = [float(re.search(r"\[([\d.]+)%", s).group(1)) for s in d["expert side effects"]]
    hi = [float(re.search(r"; ([\d.]+)%\]", s).group(1)) for s in d["expert side effects"]]
    fl = [float(s.rstrip("%")) for s in d["evaluators flagging (mean of 11)"]]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.bar(x - 0.2, exp, 0.4, color=C["red"], label="Expert: side effect",
           yerr=[np.subtract(exp, lo), np.subtract(hi, exp)], capsize=3, error_kw={"lw": 0.8})
    ax.bar(x + 0.2, fl, 0.4, color=C["sky"], label="Evaluators flagging (mean of 11)")
    ax.set_xticks(x, [f"{o}\n(n = {n})" for o, n in zip(order, d.trajectories)], fontsize=8)
    ax.set_ylabel("% of trajectories")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "ending")


def leaderboard_cells():
    """Expert side-effect rate against the evaluators' mean flag rate, per agent × benchmark."""
    d = pd.read_csv(TAB / "t7_cells.csv")
    marks = {"AssistantBench": "o", "VisualWebArena": "s", "WebArena": "^", "WorkArena": "D"}
    cols = {"Claude 3.7 Sonnet": C["orange"], "GPT-4o": C["blue"], "Llama 3.3 70B": C["green"], "Qwen2.5-VL 72B": C["purple"]}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for _, r in d.iterrows():
        ax.scatter(100 * r.side, 100 * r.j_side, marker=marks[r.bench_name], color=cols[r.agent_name], s=40)
    for b, m in marks.items():
        ax.scatter([], [], marker=m, color=C["grey"], label=b)
    for a, c in cols.items():
        ax.scatter([], [], marker="o", color=c, label=a)
    ax.set_xlabel("Side effects according to the experts (%)")
    ax.set_ylabel("Trajectories flagged by the\nevaluators (%, mean of 11)")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right")
    save(fig, "leaderboard_cells")


def attacks():
    """Evaluator experiments: share of detected side effects hidden by each attack, per model and defence."""
    p = TAB / "exp_attacks.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    lab, val, lo, hi, col = [], [], [], [], []
    for _, r in d.iterrows():
        m = re.search(r"(\d+)/(\d+) = ([\d.]+)% \[([\d.]+)%; ([\d.]+)%\]", r["hidden by the attack"])
        k, n, v, a, b = m.groups()
        short = "injection" if r.attack.startswith("Injection") else "false self-report"
        lab.append(f"{r.judge}\n{short} (n = {n})")
        val.append(float(v)); lo.append(float(v) - float(a)); hi.append(float(b) - float(v))
        col.append(C["red"] if short == "injection" else C["grey"])
    y = np.arange(len(lab))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(y, val, xerr=[lo, hi], color=col, capsize=3, error_kw={"lw": 0.8})
    ax.set_yticks(y, lab, fontsize=7)
    ax.set_xlabel("Detected side effects hidden by the attack (%, Wilson 95% CI)")
    ax.set_xlim(0, 105)
    save(fig, "attacks")


def attacks_by_model():
    """E7/E8: share of detected side effects hidden by each attack, by evaluator model (original prompt)."""
    p = TAB / "exp_models.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    atts = [("_injection", "Injection", C["red"]), ("_persona", "Persona", C["orange"]),
            ("_false authority", "False authority", C["purple"]), ("_base64", "Base64", C["sky"])]
    d = d[d["_injection"].notna()].copy()
    d = d.sort_values("_injection")
    d["model"] = d.model.replace({"glm-4.7-flash:latest": "glm-4.7-flash", "nemotron-3-nano:latest": "nemotron-3-nano",
                                  "hf.co/mradermacher/EuroLLM-22B-Instruct-2512-i1-GGUF:Q4_K_M": "EuroLLM-22B-Instruct"})
    y = np.arange(len(d))
    h = 0.2
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for i, (col, lab, colour) in enumerate(atts):
        if col in d:
            ax.barh(y + (i - 1.5) * h, 100 * d[col].fillna(0), h, color=colour, label=lab)
    ax.set_yticks(y, [f"{m} ({'open' if w == 'open' else 'closed'})" for m, w in zip(d.model, d.weights)], fontsize=7.5)
    ax.set_xlabel("Detected side effects hidden by the attack (%)")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "attacks_by_model")


surface_matched()
ending()
leaderboard_cells()
attacks()
attacks_by_model()
