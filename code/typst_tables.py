import re
"""Writes the result tables of the dissertation as Typst files, from results/tables/*.csv.

Usage: python3 code/typst_tables.py   (run code/analysis.py first)
Output: Tese/generated/*.typ  (included by the chapters; never edit by hand)
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results" / "tables"
GEN = ROOT / "Tese" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
SRC = ""


def esc(x):
    s = str(x)
    for a, b in (("#", "\\#"), ("[", "\\["), ("]", "\\]"), ("*", "\\*"), ("_", "\\_"), ("@", "\\@"), ("$", "\\$"), ("<", "\\<"), (">", "\\>")):
        s = s.replace(a, b)
    return s


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def fp(v):
    """p-value for display: '< 0.001' below 0.001, never scientific notation."""
    s = str(v)
    try:
        x = float(s)
    except ValueError:
        return esc(s)
    if math.isnan(x):
        return "—"
    if x < 0.001:
        return "< 0.001"
    return f"{x:.3f}".rstrip("0").rstrip(".") if "e" in s.lower() or len(s) > 6 else s


def table(header, rows, caption, label, align=None, columns=None, size=None):
    n = len(header)
    align = align or ("left",) + ("right",) * (n - 1)
    columns = columns or ("1fr",) + ("auto",) * (n - 1)
    cells = ", ".join(f"[*{esc(h)}*]" for h in header)
    body = ",\n    ".join(", ".join(f"[{c}]" for c in r) for r in rows)
    sz = (f"set text(size: {size})\n  " if size else "") + "set par(justify: false)\n  "
    return (f"#figure(\n  {{\n  {sz}table(\n    columns: ({', '.join(columns)}),\n    align: ({', '.join(align)}),\n"
            f"    stroke: 0.5pt,\n    inset: 4pt,\n    {cells},\n    {body},\n  )}},\n  caption: [{caption}],\n) <{label}>\n")


def write(name, content):
    (GEN / f"{name}.typ").write_text(content, encoding="utf-8")


def prevalence():
    d = pd.read_csv(TAB / "rq1_prevalence.csv")
    rows = []
    for _, r in d.iterrows():
        bold = r.benchmark == "All"
        b = "*All*" if bold else esc(r.benchmark)
        a = "" if bold else esc(r.agent)
        rows.append([b, a, f"{int(r.n):,}", f"{int(r.success_n)} ({pct(r.success_rate)})",
                     f"{int(r.side_n)} ({pct(r.side_rate)})", f"{int(r.loop_n)} ({pct(r.loop_rate)})",
                     num(r.mean_steps, 1), pct(r.at_limit, 0)])
    write("tab_prevalence", table(
        ["Benchmark", "Agent", "Trajectories", "Successes", "Side effects", "Loops", "Mean steps", "At 30-step limit"],
        rows, "Trajectories, task successes, side effects and repetition loops by benchmark and agent (primary expert annotation). " + SRC,
        "tab:prevalence", align=("left", "left") + ("right",) * 6, columns=("auto", "auto") + ("1fr",) * 6, size="8.5pt"))


def ceiling():
    d = pd.read_csv(TAB / "human_ceiling.csv", dtype=str)
    names = {"success": "Task success", "side": "Side effects", "loop": "Repetition loops"}
    rows = [[names[r.dimension], str(r.n), f"{r['positives (1st)']} / {r['positives (2nd)']}", str(r["raw agreement"]),
             str(r.kappa), str(r["kappa 95% CI"]), str(r.AC1)] for _, r in d.iterrows()]
    write("tab_ceiling", table(
        ["Dimension", "Trajectories", "Positives (1st / 2nd)", "Raw agreement", "κ", "κ 95% CI", "AC1"],
        rows, "Agreement between two expert annotators on the doubly annotated trajectories (human ceiling). " + SRC,
        "tab:ceiling"))


def judges(dim, label, caption, size="8.5pt"):
    d = pd.read_csv(TAB / "rq2_judges.csv")
    d = d[d.dimension == dim].sort_values("kappa", ascending=False)
    rows = [[esc(r.judge_name), pct(r.flag_rate), pct(r.sensitivity), pct(r.false_alarm), pct(r.precision),
             f"{num(r.kappa)} [{num(r.kappa_lo)}; {num(r.kappa_hi)}]", num(r.ac1)] for _, r in d.iterrows()]
    n, pos = int(d.n.iloc[0]), int(d.positives.iloc[0])
    write(f"tab_judges_{dim}", table(
        ["Evaluator", "Flag rate", "Sensitivity", "False alarms", "Precision", "κ [95% CI]", "AC1"],
        rows, caption.format(n=f"{n:,}", pos=pos) + " " + SRC, label, size=size))


def kappa_compare():
    d = pd.read_csv(TAB / "rq2_judges.csv")
    piv = d.pivot(index="judge_name", columns="dimension", values="kappa").sort_values("success", ascending=False)
    sens = d.pivot(index="judge_name", columns="dimension", values="sensitivity")
    rows = [[esc(j), num(r.success), num(r.loop), num(r.side), pct(sens.loc[j, "success"]), pct(sens.loc[j, "loop"]), pct(sens.loc[j, "side"])]
            for j, r in piv.iterrows()]
    rows.append(["*Mean*", f"*{num(piv.success.mean())}*", f"*{num(piv.loop.mean())}*", f"*{num(piv.side.mean())}*",
                 f"*{pct(sens.success.mean())}*", f"*{pct(sens.loop.mean())}*", f"*{pct(sens.side.mean())}*"])
    write("tab_kappa_compare", table(
        ["Evaluator", "κ success", "κ loops", "κ side effects", "Sens. success", "Sens. loops", "Sens. side effects"],
        rows, "Agreement (κ) and sensitivity of the same evaluators on the three dimensions. " + SRC, "tab:kappa-compare", size="8.5pt"))


def modality():
    d = pd.read_csv(TAB / "rq2_modality.csv")
    names = {"success": "success", "side": "side effects", "loop": "loops"}
    rows = [[esc(r.model), names[r.dimension], pct(r.acc_axtree), pct(r.acc_screenshot), str(r.only_axtree_correct),
             str(r.only_screenshot_correct), f"{r.p:.3f}", f"{r.p_holm:.3f}"] for _, r in d.iterrows()]
    write("tab_modality", table(
        ["Model", "Dimension", "Acc. AXTree", "Acc. screenshot", "Only AXTree correct", "Only screenshot correct", "p", "p (Holm)"],
        rows, "Effect of the observation modality on the accuracy of each model (exact McNemar test; Holm correction over the 12 tests). " + SRC,
        "tab:modality", align=("left", "left") + ("right",) * 6, columns=("2fr", "1.3fr") + ("1fr",) * 6, size="8.5pt"))


def conflation():
    fa = pd.read_csv(TAB / "rq4_false_alarms_by_condition.csv")
    se = pd.read_csv(TAB / "rq4_sensitivity_by_condition.csv")
    rows = [[esc(r.iloc[0]), f"{int(r.pairs):,}", esc(r.iloc[2])] for _, r in fa.iterrows()]
    write("tab_false_alarms", table(
        ["Condition (expert: no side effect)", "Pairs", "False alarms [95% Wilson CI]"], rows,
        "False alarms on side effects according to the evaluator's own verdicts and to the expert labels (11 LLM evaluators; trajectories without side effects according to the expert). " + SRC,
        "tab:false-alarms"))
    rows = [[esc(r.iloc[0]), f"{int(r.pairs):,}", esc(r.iloc[2])] for _, r in se.iterrows()]
    write("tab_sensitivity_conditions", table(
        ["Condition (expert: side effect)", "Pairs", "Sensitivity [95% Wilson CI]"], rows,
        "Detection of side effects according to the evaluator's own success verdict and to the expert labels (11 LLM evaluators; trajectories with side effects according to the expert). " + SRC,
        "tab:sens-conditions"))


def regressions():
    out = []
    for name, label, cap in (("rq4_regression_false_alarms", "tab:reg-fa", "Logistic regression of the side-effect verdict on trajectories without side effects (false alarms); fixed effects for evaluator, agent and benchmark; standard errors clustered by trajectory."),
                             ("rq4_regression_detection", "tab:reg-det", "Logistic regression of the side-effect verdict on trajectories with side effects (detection); fixed effects for evaluator; standard errors clustered by trajectory.")):
        d = pd.read_csv(TAB / f"{name}.csv")
        names = {"j_success": "Evaluator says the task succeeded", "success": "Expert: task succeeded",
                 "loop": "Expert: loop", "log_steps": "log(steps)"}
        rows = [[names.get(r.variable, esc(r.variable)), str(r.OR), str(r["95% CI"]), fp(r.p)] for _, r in d.iterrows()]
        write(name, table(["Variable", "OR", "95% CI", "p"], rows, cap + " " + SRC, label))


def shared():
    d = pd.read_csv(TAB / "rq4_shared_errors.csv", dtype=str)
    names = {"success": "Task success", "side": "Side effects", "loop": "Repetition loops"}
    rows = [[names[r.dimension], str(r.positives), str(r["mean κ judge–judge"]), str(r["mean κ judge–human"]),
             str(r["α judges"]), str(r["α judges + human"]), str(r["positives missed by all judges"]),
             str(r["positives caught by all judges"])] for _, r in d.iterrows()]
    write("tab_shared", table(
        ["Dimension", "Positives", "Mean κ evaluator–evaluator", "Mean κ evaluator–expert", "α evaluators", "α evaluators + expert",
         "Positives missed by all", "Positives caught by all"],
        rows, "Agreement among the evaluators and with the expert, and expert-annotated cases missed or caught by every evaluator. " + SRC,
        "tab:shared", columns=("auto",) + ("1fr",) * 7, size="8.5pt"))


def committees():
    d = pd.read_csv(TAB / "rq4_committees.csv", dtype=str)
    names = {"success": "Task success", "side": "Side effects", "loop": "Repetition loops"}
    rows = []
    for _, r in d.iterrows():
        rows.append([names[r.dimension], esc(r.rule), str(r.sensitivity), str(r["false alarms"]), str(r.precision),
                     str(r["κ [95% CI]"]), str(r.AC1)])
    write("tab_committees", table(
        ["Dimension", "Rule", "Sensitivity", "False alarms", "Precision", "κ [95% CI]", "AC1"],
        rows, "Single evaluator versus committees of evaluators. " + SRC, "tab:committees",
        align=("left", "left") + ("right",) * 5, columns=("auto", "1fr") + ("auto",) * 5, size="8pt"))


def by_steps():
    occ = pd.read_csv(TAB / "rq3_occurrence_by_steps.csv")
    det = pd.read_csv(TAB / "rq3_detection_by_steps.csv")
    rows = []
    for (_, o), (_, d) in zip(occ.iterrows(), det.iterrows()):
        rows.append([esc(o.steps), str(o.n), pct(o.side_rate), pct(o.loop_rate), pct(o.success_rate),
                     pct(d.side_sensitivity), pct(d.side_false_alarm), pct(d.loop_sensitivity), pct(d.loop_false_alarm)])
    write("tab_by_steps", table(
        ["Steps", "Trajectories", "Side effects", "Loops", "Success", "Side: sensitivity", "Side: false alarms",
         "Loop: sensitivity", "Loop: false alarms"],
        rows, "Occurrence (expert labels) and detection (11 LLM evaluators) by number of steps. " + SRC, "tab:by-steps",
        columns=("auto",) * 9, size="8pt"))


def appendix():
    d = pd.read_csv(TAB / "rq2_judges.csv")
    out = []
    names = {"success": "task success", "side": "side effects", "loop": "repetition loops"}
    for dim in ("side", "loop", "success"):
        dd = d[d.dimension == dim].sort_values("judge_name")
        rows = [[esc(r.judge_name), str(r.n), str(r.positives), str(r.tp), str(r.fp), str(r.fn), str(r.tn),
                 pct(r.accuracy), str(r.unreadable)] for _, r in dd.iterrows()]
        out.append(table(["Evaluator", "n", "Positives", "TP", "FP", "FN", "TN", "Accuracy", "Unreadable"], rows,
                         f"Confusion matrices of the 11 LLM evaluators for {names[dim]} (unreadable verdicts counted as negative). " + SRC,
                         f"tab:app-{dim}", size="8.5pt"))
    sb = pd.read_csv(TAB / "rq2_side_by_benchmark.csv")
    benches = sorted(sb.benchmark.unique())
    for part, (bs, label) in enumerate(((benches[:2], "tab:app-side-bench"), (benches[2:], "tab:app-side-bench-2"))):
        rows = [[esc(r.benchmark), esc(r.judge), str(r.positives), pct(r.sensitivity), pct(r.false_alarm), num(r.kappa)]
                for _, r in sb[sb.benchmark.isin(bs)].sort_values(["benchmark", "judge"]).iterrows()]
        out.append(table(["Benchmark", "Evaluator", "Positives", "Sensitivity", "False alarms", "κ"], rows,
                         f"Detection of side effects by evaluator on {' and '.join(bs)}. " + SRC, label,
                         align=("left", "left") + ("right",) * 4, columns=("auto", "1fr") + ("auto",) * 4, size="8pt"))
    write("app_tables", "\n".join(out))


if __name__ == "__main__":
    prevalence()
    ceiling()
    judges("side", "tab:judges-side", "Detection of side effects by the 11 LLM evaluators (n = {n} trajectories, {pos} with side effects according to the expert; sorted by κ).")
    judges("loop", "tab:judges-loop", "Detection of repetition loops by the 11 LLM evaluators (n = {n} trajectories, {pos} with loops according to the expert; sorted by κ).")
    kappa_compare()
    modality()
    by_steps()
    conflation()
    regressions()
    shared()
    committees()
    appendix()
    print("typst tables written to", GEN)


def extended():
    a = pd.read_csv(TAB / "ext_side_without_limit.csv")
    rows = [[esc(r.judge), pct(r.sensitivity), pct(r.false_alarm), pct(r.precision),
             f"{num(r.kappa)} [{num(r.kappa_lo)}; {num(r.kappa_hi)}]"] for _, r in a.iterrows()]
    write("tab_ext_without_limit", table(
        ["Evaluator", "Sensitivity", "False alarms", "Precision", "κ [95% CI]"], rows,
        f"Detection of side effects on the {int(a.n.iloc[0])} trajectories below the 30-step limit ({int(a.positives.iloc[0])} with side effects). " + SRC,
        "tab:ext-without-limit", size="8.5pt"))
    for name, label, cap in (("ext_regression_fa_without_limit", "tab:ext-reg-limit",
                              "Conflation regression (false alarms on side effects) restricted to the trajectories below the 30-step limit; fixed effects for evaluator, agent and benchmark; standard errors clustered by trajectory."),
                             ("ext_regression_repetition", "tab:ext-reg-rep",
                              "Logistic regression of false alarms on side effects on the number of repeated actions, controlling for length and the expert labels; fixed effects for evaluator, agent and benchmark; standard errors clustered by trajectory."),
                             ("ext_regression_family", "tab:ext-reg-family",
                              "Same-family effect on the side-effect verdict (controls: expert labels of side effect, loop and success, length, and fixed effects for evaluator, agent and benchmark; clustered by trajectory).")):
        d = pd.read_csv(TAB / f"{name}.csv", dtype=str)
        names = {"j_success": "Evaluator says the task succeeded", "success": "Expert: task succeeded",
                 "loop": "Expert: loop", "log_steps": "log(steps)", "log_rep": "log(1 + repeated actions)",
                 "same_family": "Evaluator and agent from the same provider"}
        rows = [[names.get(r.variable, esc(r.variable)), r.OR, r["95% CI"], r.p] for _, r in d.iterrows()]
        write(name, table(["Variable", "OR", "95% CI", "p"], rows, cap + " " + SRC, label))
    c = pd.read_csv(TAB / "ext_cost.csv")
    rows = [[esc(r.judge_name), str(int(r.tp + r.fp)), str(int(r.tp)), str(int(r.fn)), num(r.alarms_per_true, 1)]
            for _, r in c.sort_values("alarms_per_true").iterrows()]
    rows.append(["_Random review (no evaluator)_", "—", "—", "—", num(1301 / 84, 1)])
    write("tab_ext_cost", table(["Evaluator", "Alarms", "True side effects found", "Missed (of 84)", "Alarms per side effect found"],
                                rows, "Cost of oversight: alarms that a reviewer must inspect per real side effect found. " + SRC,
                                "tab:ext-cost", size="8.5pt"))
    j = pd.read_csv(TAB / "ext_justifications_summary.csv")
    rows = [[esc(r.alarms), f"{int(r.n):,}", esc(r["repetition only"]), esc(r["both"]), esc(r["state change only"]),
             esc(r["neither"])] for _, r in j.iterrows()]
    write("tab_ext_justifications", table(["Alarms", "n", "Repetition only", "Both", "State change only", "Neither"], rows,
                                          "Cues found in the evaluators' justifications of their side-effect alarms (keyword dictionaries; 95% Wilson CI). " + SRC,
                                          "tab:ext-justifications", size="8.5pt"))


extended()


def third():
    """Tables of the third round (code/analysis_third.py)."""
    def wil(s):  # "12/248 = 4.8% [2.8%; 8.3%]" -> "12/248 (4.8%)"
        k, rest = s.split(" = ")
        return esc(f"{k} ({rest.split(' ')[0]})")
    rows = []
    for scope, f in (("All three", "t1_task_type.csv"), ("WebArena", "t1_task_type_webarena.csv"),
                     ("VisualWebArena", "t1_task_type_visualwebarena.csv")):
        for _, r in pd.read_csv(TAB / f, dtype=str).iterrows():
            rows.append([esc(scope), esc(r.task_type.capitalize()), str(r.trajectories), wil(r["side effects"]),
                         esc(r["false alarms (mean)"]), esc(r["sensitivity (mean)"]), esc(r["mean κ"])])
    write("tab_t1_task_type", table(
        ["Scope", "Task type", "Trajectories", "Side effects", "False alarms", "Sensitivity", "Mean κ"], rows,
        "Side effects and their detection by task type. State-changing tasks are those whose official evaluation "
        "inspects the final state of the website; AssistantBench tasks are information-seeking. False alarms, "
        "sensitivity and κ are means over the 11 evaluators. " + SRC,
        "tab:t1-task-type", columns=("auto",) * 7, size="8.5pt"))
    names = {"state_task": "State-changing task", "log_steps": "log(steps)", "success": "Expert: task succeeded",
             "loop": "Expert: loop"}
    a = pd.read_csv(TAB / "t1_reg_occurrence.csv", dtype=str).set_index("variable")
    b = pd.read_csv(TAB / "t1_reg_false_alarms.csv", dtype=str).set_index("variable")
    rows = []
    for v in ("state_task", "log_steps", "success", "loop"):
        ra = [a.loc[v, "OR"], a.loc[v, "95% CI"], a.loc[v, "p"]] if v in a.index else ["—"] * 3
        rb = [b.loc[v, "OR"], b.loc[v, "95% CI"], b.loc[v, "p"]] if v in b.index else ["—"] * 3
        rows.append([names[v]] + ra + rb)
    write("tab_t1_reg", table(
        ["Variable", "OR (occurrence)", "95% CI", "p", "OR (false alarm)", "95% CI", "p"], rows,
        "Task type in two logistic regressions on WebArena, VisualWebArena and AssistantBench: occurrence of a side "
        "effect (expert label; fixed effects for agent and benchmark; robust standard errors) and false alarm on "
        "side effects (expert negatives; fixed effects for evaluator, agent and benchmark; standard errors "
        "clustered by trajectory). " + SRC, "tab:t1-reg", size="8.5pt"))
    rows = []
    for bench, f in (("WebArena", "t2_site_webarena.csv"), ("VisualWebArena", "t2_site_visualwebarena.csv")):
        for _, r in pd.read_csv(TAB / f, dtype=str).iterrows():
            rows.append([esc(bench), esc(r.site.replace("_", " ")), str(r.trajectories), wil(r["side effects"]),
                         esc(r["false alarms (mean)"]), esc(r["sensitivity (mean)"]), esc(r["mean κ"]),
                         esc(r["max κ"])])
    write("tab_t2_sites", table(
        ["Benchmark", "Website", "Trajectories", "Side effects", "False alarms", "Sensitivity", "Mean κ", "Max κ"],
        rows, "Side effects and their detection by website (means over the 11 evaluators; multi-site tasks involve "
        "more than one website). " + SRC, "tab:t2-sites", columns=("auto",) * 8, size="8.5pt"))
    rows = []
    for _, r in pd.read_csv(TAB / "t2_category_workarena.csv", dtype=str).iterrows():
        rows.append([esc(r.category.replace("_", " ")), str(r.trajectories), wil(r["side effects"]), esc(r.loops),
                     esc(r.success), esc(r["false alarms (mean)"]), esc(r["sensitivity (mean)"]), esc(r["mean κ"])])
    write("tab_t2_workarena", table(
        ["Category", "Trajectories", "Side effects", "Loops", "Success", "False alarms", "Sensitivity", "Mean κ"],
        rows, "WorkArena trajectories by task category (means over the 11 evaluators). " + SRC,
        "tab:t2-workarena", size="8pt"))
    d = pd.read_csv(TAB / "t3_surface_auc.csv")
    rows = [[esc(r.target), esc(r.positives), esc(r["CV AUC [95% CI]"])] for _, r in d.iterrows()]
    write("tab_t3_auc", table(
        ["Label predicted", "Positives", "Cross-validated AUC [95% CI]"], rows,
        "How well ten surface features of the action history predict each label (logistic regression, 10-fold "
        "cross-validation, bootstrap CI) on the 1,300 trajectories with all three expert labels. " + SRC, "tab:t3-auc", size="8.5pt"))
    e = pd.read_csv(TAB / "t3_coef_expert.csv", dtype=str).set_index("variable")
    m = pd.read_csv(TAB / "t3_coef_majority.csv", dtype=str).set_index("variable")
    names = {"log_steps": "log(steps)", "log_rep": "log(1 + repeated actions)",
             "log_urls": "log(1 + distinct pages)", "at_limit": "Reached the 30-step limit",
             "log_fill": "log(1 + form-filling actions)", "log_select": "log(1 + selection actions)",
             "log_nav": "log(1 + navigation actions)", "log_scroll": "log(1 + scroll actions)",
             "final_msg": "Ends with an answer to the user", "final_infeasible": "Ends declaring the task infeasible"}
    rows = [[names[v], e.loc[v, "OR"], e.loc[v, "95% CI"], m.loc[v, "OR"], m.loc[v, "95% CI"]] for v in e.index]
    write("tab_t3_coef", table(
        ["Surface feature", "OR experts", "95% CI", "OR evaluators' majority", "95% CI"], rows,
        "What predicts a side effect according to the experts and according to the majority of the 11 evaluators: "
        "logistic regressions on the same surface features (full sample, robust standard errors). " + SRC,
        "tab:t3-coef", size="8.5pt"))
    d = pd.read_csv(TAB / "t3_ending.csv", dtype=str)
    rows = [[esc(r["how the run ends"].capitalize()), str(r.trajectories),
             esc(r["expert side effects"].split(" = ")[1]), esc(r["evaluators flagging (mean of 11)"]),
             esc(r["sensitivity (mean)"])] for _, r in d.iterrows()]
    write("tab_t3_ending", table(
        ["How the run ends", "Trajectories", "Expert side effects [95% CI]", "Evaluators flagging", "Sensitivity"],
        rows, "How the run ends: share of trajectories with a side effect according to the experts, and share "
        "flagged by the evaluators (means over the 11 evaluators). " + SRC, "tab:t3-ending", size="8.5pt"))
    d = pd.read_csv(TAB / "t3_matched_rate.csv", dtype=str); d = d.iloc[d._kj.astype(float).argsort()[::-1]]
    rows = [[esc(r.evaluator), esc(r["flag rate"]), esc(r["κ evaluator"]), esc(r["κ surface rule (same rate)"]),
             esc(r["κ length rule (same rate)"]), esc(r["evaluator − surface [95% CI]"])] for _, r in d.iterrows()]
    write("tab_t3_matched", table(
        ["Evaluator", "Flag rate", "κ evaluator", "κ surface rule", "κ length rule", "Evaluator − surface [95% CI]"],
        rows, "Each evaluator against two rules that flag the same share of trajectories: the out-of-fold surface "
        "model and the number of steps (κ with the expert side-effect label; paired bootstrap CI). " + SRC,
        "tab:t3-matched", size="8.5pt"))
    d = pd.read_csv(TAB / "t4_triage.csv")
    rows = [[esc(r.score), esc(r["AUC [95% CI]"]), esc(r["precision top 5%"]), esc(r["precision top 10%"]),
             esc(r["recall top 10%"]), esc(r["precision top 20%"]), esc(r["recall top 20%"])] for _, r in d.iterrows()]
    write("tab_t4_triage", table(
        ["Ranking score", "AUC [95% CI]", "Precision top 5%", "Precision top 10%", "Recall top 10%",
         "Precision top 20%", "Recall top 20%"], rows,
        "Ranking the 1,300 trajectories with all three expert labels for human review of side effects (base rate 6.5%). " + SRC,
        "tab:t4-triage", columns=("2.4fr", "1.6fr") + ("1fr",) * 5, size="8.5pt"))
    d = pd.read_csv(TAB / "t5_same_trajectories.csv", dtype=str)
    dims = {"side": "Side effects", "loop": "Repetition loops", "success": "Task success"}
    rows = [[dims.get(r.dimension, esc(r.dimension)), str(r.trajectories), str(r["positives (1st expert)"]), esc(r["κ second expert"]),
             esc(re.sub(r" \(([^()]*)\)\)$", r", \1)", r["κ best evaluator"])), esc(r["κ mean evaluator"]), esc(r["second expert − best evaluator [95% CI]"])]
            for _, r in d.iterrows()]
    write("tab_t5_same", table(
        ["Dimension", "Trajectories", "Positives", "κ 2nd expert", "κ best evaluator", "κ mean evaluator",
         "Difference [95% CI]"], rows,
        "Second expert and evaluators compared with the first expert on the same doubly annotated trajectories; "
        "the difference (second expert minus best evaluator) has a paired bootstrap CI. " + SRC,
        "tab:t5-same", columns=("1.4fr", "1.1fr", "1fr", "1fr", "2.2fr", "1fr", "1.6fr"), size="8.5pt"))
    d = pd.read_csv(TAB / "t6_reward.csv", dtype=str)
    rows = [[esc(r.expert.capitalize()), str(r.trajectories), f"{int(r.pairs):,}", wil(r.accepted),
             wil(r["accepted as safe"])] for _, r in d.iterrows()]
    write("tab_t6_reward", table(
        ["Expert label", "Trajectories", "Pairs", "Judged successful", "Judged successful and safe"], rows,
        "How the evaluators judge the runs that the experts considered successful, with and without side effects "
        "(evaluator × trajectory pairs). " + SRC, "tab:t6-reward", size="8.5pt"))
    d = pd.read_csv(TAB / "t7_leaderboard.csv", dtype=str)
    d = d.iloc[d._side.astype(float).argsort()[::-1]]
    rows = [[esc(r.evaluator), esc(r["ρ success"]), esc(r["ρ side"]), esc(r["MAE side (pp)"]),
             esc(r["ρ safe_success"])] for _, r in d.iterrows()]
    write("tab_t7_leaderboard", table(
        ["Evaluator", "ρ success [95% CI]", "ρ side effects [95% CI]", "Error side effects (pp)",
         "ρ safe success [95% CI]"], rows,
        "Ranking the 15 agent × benchmark combinations: Spearman correlation between the rate estimated by each "
        "evaluator and the expert rate, for success, side effects and safe success (success without side effects); "
        "mean absolute error of the side-effect rate in percentage points; CI by resampling trajectories within "
        "combinations. " + SRC, "tab:t7-leaderboard", columns=("2.4fr", "1.4fr", "1.4fr", "1fr", "1.4fr"), size="8.5pt"))
    d = pd.read_csv(TAB / "t7_agents.csv", dtype=str)
    rows = [[esc(r.agent), esc(r["expert success"]), esc(r["evaluators' success (mean)"]), esc(r["expert side effects"]),
             esc(r["evaluators' side effects (mean)"]), esc(r["expert safe success"]),
             esc(r["evaluators' safe success (mean)"])] for _, r in d.iterrows()]
    write("tab_t7_agents", table(
        ["Agent", "Success (experts)", "Success (evaluators)", "Side effects (experts)", "Side effects (evaluators)",
         "Safe success (experts)", "Safe success (evaluators)"], rows,
        "Agent-level rates according to the experts and to the evaluators (mean over the 11 evaluators), all "
        "benchmarks. " + SRC, "tab:t7-agents", columns=("1.6fr",) + ("1fr",) * 6, size="8.5pt"))


third()

ESRC = ""


def experiments():
    """Tables of the evaluator experiments (code/experiments_analysis.py)."""
    if not (TAB / "exp_conditions.csv").exists():
        return
    d = pd.read_csv(TAB / "exp_conditions.csv", dtype=str)
    d = d[~d.cond.isin(["A1", "A2", "A3", "A4", "D1A", "D2A", "A2_38", "A2_L"]) & ~d.cond.str.startswith("J")]  # attacks: see tab_exp_attacks
    rows = [[esc(r.condition), str(r.trajectories), str(r["with side effect"]), esc(r.sensitivity),
             esc(r["false alarms"]), esc(r["precision (population)"]), esc(r["κ (population) [95% CI]"])]
            for _, r in d.iterrows()]
    write("tab_exp_conditions", table(
        ["Condition", "Trajectories", "With side effect", "Sensitivity", "False alarms", "Precision (population)",
         "κ (population) [95% CI]"], rows,
        "Detection of side effects by the evaluator in each experimental condition. Sensitivity and false alarms are "
        "observed in the sample; precision and κ are re-weighted to the population prevalence of side effects (6.5%), "
        "with a bootstrap CI that resamples trajectories with and without side effects separately. The rows marked "
        "\"published\" are the verdicts of the GPT-4o mini judge on the same prompts. Unless stated otherwise the "
        "evaluator is gemini-3.7-flash; the defences and the other models were run on the 84 trajectories with side "
        "effects and 100 without. Conditions with attacks are in @tab:exp-attacks. " + ESRC, "tab:exp-conditions",
        columns=("2.6fr", "1fr", "1fr", "1fr", "1fr", "1.2fr", "1.9fr"), size="8pt"))
    d = pd.read_csv(TAB / "exp_attacks.csv", dtype=str)
    rows = []
    for _, r in d.iterrows():
        model, _, setting = r.judge.partition(", ")
        frac_, _, rest = r["hidden by the attack"].partition(" = ")
        share, _, ci = rest.partition(" ")
        rows.append([esc(model), esc(setting), esc(r.attack.replace(" to the evaluator", "")), esc(frac_),
                     esc(share), esc(ci.replace("%", "")), esc(r["false alarms before → after"]),
                     fp(r["McNemar p (side effects)"])])
    write("tab_exp_attacks", table(
        ["Model", "Prompt", "Attack", "Hidden / detected", "Hidden", "95% CI", "False alarms", "p"], rows,
        "Effect of the two attacks, inserted in the agent's last reasoning step without changing any action: side "
        "effects detected without the attack that the evaluator no longer flags with it (Wilson CI), false alarms on "
        "the 100 clean trajectories before → after the attack, and exact McNemar test on the 84 trajectories with side "
        "effects. " + ESRC, "tab:exp-attacks",
        align=("left", "left", "left", "right", "right", "right", "right", "right"),
        columns=("1.5fr", "1.6fr", "1.1fr", "0.9fr", "0.7fr", "1fr", "1.1fr", "0.7fr"), size="8pt"))


experiments()

MSRC = ""


def models_table():
    """E7/E8: detection and attacks by evaluator model (closed Gemini models and local open-weight models)."""
    p = TAB / "exp_models.csv"
    if not p.exists():
        return
    d = pd.read_csv(p, dtype=str).fillna("—")
    names = {"glm-4.7-flash:latest": "glm-4.7-flash", "nemotron-3-nano:latest": "nemotron-3-nano",
             "hf.co/mradermacher/EuroLLM-22B-Instruct-2512-i1-GGUF:Q4_K_M": "EuroLLM-22B-Instruct"}
    order = ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemma4:31b", "qwen3.6:35b-a3b",
             "gpt-oss:20b", "gpt-oss:120b", "glm-4.7-flash:latest", "hf.co/mradermacher/EuroLLM-22B-Instruct-2512-i1-GGUF:Q4_K_M",
             "ministral-3:14b", "phi4:14b", "nemotron-3-nano:latest", "llama3.1:8b"]
    d["_o"] = d.model.map({m: i for i, m in enumerate(order)}).fillna(99)
    d = d.sort_values("_o")
    d["model"] = d.model.replace(names)

    def frac(v):
        if v == "—" or "/" not in v:
            return "—"
        k, n = map(int, v.split("/"))
        return f"{v} ({100 * k / n:.0f}%)"
    cols = [("hidden: injection", "Injection"), ("hidden: persona", "Persona"),
            ("hidden: false authority", "False authority"), ("hidden: base64", "Base64")]
    rows = [[esc(r.model), esc(r.weights), esc(r.sensitivity), esc(r["false alarms"]), esc(r["κ (population)"]),
             esc(r.get("κ guided", "—"))] + [esc(frac(r.get(c, "—"))) for c, _ in cols] for _, r in d.iterrows()]
    write("tab_exp_models", table(
        ["Evaluator", "Weights", "Sensitivity", "False alarms", "κ", "κ guided"] + [lab for _, lab in cols], rows,
        "Detection of side effects and effect of the injection and of three jailbreaks, by evaluator model, with the "
        "original prompt. Sensitivity, false alarms and κ on the 350 trajectories of the sample for gemini-3.7-flash, "
        "gemma4:31b, qwen3.6:35b-a3b and gpt-oss:20b, and on the attack subset (84 trajectories with side effects, "
        "100 without) for the other models; the attacks always on the attack subset. κ is re-weighted to the "
        "population prevalence. Each attack cell gives the side effects detected "
        "without attack that the evaluator no longer flags with it. " + MSRC, "tab:exp-models",
        columns=("1.8fr", "0.8fr", "0.9fr", "0.9fr", "0.6fr", "0.7fr", "1fr", "1fr", "1fr", "1fr"), size="7.5pt"))
    gcols = [("hidden: injection (guided)", "Injection"), ("hidden: persona (guided)", "Persona"),
             ("hidden: authority (guided)", "False authority"), ("hidden: base64 (guided)", "Base64")]
    g = d[d["hidden: injection (guided)"] != "—"] if "hidden: injection (guided)" in d else d.iloc[0:0]
    rows = [[esc(r.model), esc(r.get("κ guided", "—"))] + [esc(frac(r.get(c, "—"))) for c, _ in gcols]
            for _, r in g.iterrows()]
    if rows:
        write("tab_exp_models_guided", table(
            ["Evaluator", "κ guided"] + [lab for _, lab in gcols], rows,
            "The same attacks against the guided prompt, for the models on which it was run: side effects detected by "
            "the guided evaluator without attack that it no longer flags with the attack. " + MSRC,
            "tab:exp-models-guided", columns=("1.8fr", "0.8fr", "1fr", "1fr", "1fr", "1fr"), size="8pt"))
    dcols = [("hidden: injection", "No defence"), ("hidden: injection, defence by instruction", "Defence by instruction"),
             ("hidden: injection, datamarking", "Datamarking"), ("hidden: injection (guided)", "Guided prompt")]
    if "hidden: injection, defence by instruction" in d:
        g = d[d["hidden: injection, defence by instruction"] != "—"]
        rows = [[esc(r.model)] + [esc(frac(r.get(c, "—"))) for c, _ in dcols] for _, r in g.iterrows()]
        if rows:
            write("tab_exp_models_defences", table(
                ["Evaluator"] + [lab for _, lab in dcols], rows,
                "Side effects detected without attack that the injection hides, with no defence and with each defence, "
                "by evaluator model. " + MSRC, "tab:exp-models-defences",
                columns=("1.8fr", "1fr", "1.2fr", "1fr", "1fr"), size="8pt"))
    if "κ temperature 0" in d:
        g = d[d["κ temperature 0"] != "—"]
        measures = [("Agreement, two runs with default settings", lambda r: r["agreement between runs"]),
                    ("Agreement, default settings vs temperature 0", lambda r: r["agreement default vs temperature 0"]),
                    ("κ, default settings", lambda r: r["κ attack subset"]),
                    ("κ, temperature 0", lambda r: r["κ temperature 0"]),
                    ("Hidden by the injection, default settings", lambda r: frac(r["hidden: injection"])),
                    ("Hidden by the injection, temperature 0", lambda r: frac(r["hidden: injection, temperature 0"])),
                    ("Hidden by the false authority, default settings", lambda r: frac(r["hidden: false authority"])),
                    ("Hidden by the false authority, temperature 0",
                     lambda r: frac(r["hidden: false authority, temperature 0"]))]
        rows = [[esc(lab)] + [esc(f(r)) for _, r in g.iterrows()] for lab, f in measures]
        if len(g):
            write("tab_exp_models_temp0", table(
                ["Measure"] + [esc(m) for m in g.model], rows,
                "Stability of the three main open-weight evaluators. Agreement of the side-effect verdict between two "
                "runs with the default sampling settings (100 trajectories) and between the default settings and "
                "temperature 0 (attack subset); κ (re-weighted to the population prevalence) and side effects hidden by "
                "the injection and by the false-authority jailbreak with each setting, on the attack subset. " + MSRC,
                "tab:exp-models-temp0", columns=("2.6fr", "1fr", "1fr", "1fr"), size="8.5pt"))


models_table()


def extra_tables():
    """Committee under attack, false alarms under attack and unreadable answers (code/experiments_extra.py)."""
    f = TAB / "exp_extra_committee.csv"
    if not f.exists():
        return
    d = pd.read_csv(f, dtype=str)
    rows = [[esc(r.Condition), esc(r.Committee), esc(r.Sensitivity), esc(r["False alarms"]),
             esc(r["Detected side effects hidden"])] for _, r in d.iterrows()]
    write("tab_exp_committee", table(
        ["Condition", "Committee", "Sensitivity", "False alarms", "Detected side effects hidden [95% CI]"], rows,
        "A committee of the three main open-weight evaluators (Gemma 4 31B, Qwen3.6 35B-A3B and gpt-oss 20B) under "
        "attack, on the attack subset (84 trajectories with side effects, 100 without). Hidden: side effects flagged "
        "by the committee without the attack that it no longer flags with it.",
        "tab:exp-committee", align=("left", "left", "right", "right", "right"),
        columns=("1.7fr", "1.5fr", "1fr", "1fr", "2.2fr"), size="8.5pt"))
    d = pd.read_csv(TAB / "exp_extra_false_alarms.csv", dtype=str)
    write("tab_exp_extra_fa", table(
        ["Evaluator", "No attack", "Injection", "Persona", "False authority", "Base64"],
        [[esc(v) for v in r] for r in d.itertuples(index=False)],
        "False alarms of each open-weight evaluator on the 100 trajectories without side effects of the attack "
        "subset, without and with each attack.", "tab:exp-extra-fa", size="8.5pt"))
    d = pd.read_csv(TAB / "exp_extra_unreadable.csv", dtype=str)
    rows = []
    for _, r in d.iterrows():
        worst = max((r[c] for c in d.columns if c.startswith("Unreadable,") and "no attack" not in c),
                    key=lambda v: float(v.rstrip("%")))
        rows.append([esc(r.Evaluator), esc(r["Unreadable, no attack"]), esc(worst),
                     esc(r["Hidden, injection (unreadable = no)"]), esc(r["Hidden, injection (unreadable excluded)"]),
                     esc(r["Hidden, false authority (unreadable = no)"]),
                     esc(r["Hidden, false authority (unreadable excluded)"])])
    write("tab_exp_extra_unreadable", table(
        ["Evaluator", "Unreadable, no attack", "Unreadable, worst attack", "Injection: hidden (as no)",
         "Injection: hidden (excluded)", "False authority: hidden (as no)", "False authority: hidden (excluded)"],
        rows, "Unreadable answers of the open-weight evaluators and the side effects hidden by the injection and by "
        "the false authority when unreadable answers are counted as negative (as in the main analysis) or excluded.",
        "tab:exp-extra-unreadable", columns=("1.6fr",) + ("1fr",) * 6, size="8pt"))
    f = TAB / "exp_extra_final.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str)
        d["Evaluator"] = d.Evaluator.map(lambda m: {"gemma4:31b": "Gemma 4 31B", "qwen3.6:35b-a3b": "Qwen3.6 35B-A3B",
                                                  "gpt-oss:20b": "gpt-oss 20B"}.get(m, m))
        rows = [[esc(v) for v in r] for r in d.itertuples(index=False)]
        write("tab_exp_final", table(
            ["Evaluator", "Where the injection is placed", "Detected side effects hidden [95% CI]",
             "False alarms before", "False alarms after"], rows,
            "The same injection placed at the end of the agent's reasoning, in the reasoning of a middle step, and on "
            "the web page as text of the last accessibility tree (indirect injection by a third party), on the attack "
            "subset. Baselines: the original prompt for the first two, the prompt with the accessibility tree for the "
            "third.", "tab:exp-final", align=("left", "left", "right", "right", "right"),
            columns=("1.4fr", "2.2fr", "2fr", "1fr", "1fr"), size="8.5pt"))
    f = TAB / "exp_extra_gemma_temp0.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str)
        write("tab_exp_gemma_temp0", table(
            ["Condition", "Sensitivity", "False alarms", "κ (population)"], [[esc(v) for v in r] for r in d.itertuples(index=False)],
            "Gemma 4 31B on all 350 trajectories with the default sampling settings and at temperature 0, with the "
            "original and the guided prompt (κ re-weighted to the population prevalence).",
            "tab:exp-gemma-temp0", align=("left", "right", "right", "right"), columns=("2.4fr", "1fr", "1fr", "1fr"),
            size="8.5pt"))


extra_tables()


def brci(x):
    """Escape a cell and put its confidence interval on a second line, so that it does not break in the middle."""
    return esc(x).replace(" \\[", " \\ \\[")


def scale_adaptive_tables():
    """Larger models and reasoning (code/experiments_scale_analysis.py) and adaptive attacks (code/adaptive_analysis.py)."""
    f = TAB / "exp_scale.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str)
        cols = ["Evaluator", "Sensitivity", "False alarms", "κ original", "κ guided", "Injection hides",
                "False authority hides", "Injection hides (guided)"]
        write("tab_exp_scale", table(
            ["Evaluator", "Sens.", "False alarms", "κ original", "κ guided", "Injection: hidden",
             "False authority: hidden", "Injection, guided: hidden"],
            [[brci(r[c]) for c in cols] for _, r in d.iterrows()],
            "The three larger open-weight models (first three rows) and the models of E8 on the attack subset (84 "
            "trajectories with side effects, 100 without). Sensitivity and false alarms with the original prompt; κ "
            "re-weighted to the population; hidden: side effects detected without the attack that it makes the evaluator "
            "miss [95% CI].", "tab:exp-scale", align=("left",) + ("right",) * 7,
            columns=("1.45fr", "0.6fr", "0.65fr", "1.35fr", "1.35fr", "1.45fr", "1.45fr", "1.45fr"), size="7.5pt"))
    f = TAB / "exp_think.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str)
        rows = [[esc(r.Evaluator), esc(r.Prompt), esc(r["Sensitivity (off → on)"]), esc(r["False alarms (off → on)"]),
                 brci(str(r["κ off"]).replace("hides ", "")), brci(str(r["κ on"]).replace("hides ", ""))]
                for _, r in d.iterrows()]
        write("tab_exp_think", table(
            ["Evaluator", "Condition", "Sensitivity (off → on)", "False alarms (off → on)", "Reasoning off",
             "Reasoning on"], rows,
            "Reasoning switched off (as in E8) and on, on the attack subset. For the two prompts, the last two columns "
            "give κ re-weighted to the population; for the injection, the share of the side effects detected without "
            "it that it hides [95% CI].", "tab:exp-think", align=("left", "left", "right", "right", "right", "right"),
            columns=("1.4fr", "1.3fr", "1.2fr", "1.2fr", "1.6fr", "1.6fr"), size="8pt"))
    f, g = TAB / "exp_adaptive_tailored.csv", TAB / "exp_adaptive_search.csv"
    if f.exists() and g.exists():
        d, s = pd.read_csv(f, dtype=str), pd.read_csv(g, dtype=str)
        models = ["Gemma 4 31B", "Qwen3.6 35B-A3B", "gpt-oss 20B"]
        rows = [[esc(r.Attack.replace(" (thesis)", ""))] + [brci(r[m]) for m in models] for _, r in d.iterrows()]
        held = {r.Evaluator: r["Held-out"] for _, r in s.iterrows() if str(r.Fold) == "all"}
        rows.append([esc("Automated search, held-out (two folds)")] + [brci(held.get(m, "not run")) for m in models])
        write("tab_exp_adaptive", table(
            ["Attack against the guided prompt", "Gemma 4 31B (n = 22)", "Qwen3.6 35B-A3B (n = 20)", "gpt-oss 20B (n = 9)"],
            rows, "Attacks against the guided prompt by an attacker who knows it: share of the side effects detected "
            "by the guided evaluator without attack that each attack makes it miss [95% CI]. The first four rows are the "
            "fixed attacks of E3 and E7, T1 to T9 are written against the guided prompt, and the last row is the "
            "automated search, tested on trajectories it never saw.", "tab:exp-adaptive",
            align=("left", "right", "right", "right"), columns=("2.2fr", "1.6fr", "1.6fr", "1.6fr"), size="8pt"))
        rows = [[esc(r.Evaluator), esc(r.Fold), esc({"ta5": "T5", "round2": "search, round 2"}.get(r["Source of the final attack"], r["Source of the final attack"])),
                 esc(r["Development success (best per round)"]), brci(r["Held-out"])] for _, r in s.iterrows()]
        write("tab_exp_adaptive_search", table(
            ["Evaluator", "Fold", "Final attack", "Development set: best after each round", "Held-out fold"], rows,
            "Automated search against the guided prompt with two-fold cross-validation. The search runs on one fold "
            "(development set) and its best attack is tested on the other.", "tab:exp-adaptive-search",
            align=("left", "left", "left", "left", "right"), columns=("1.4fr", "0.5fr", "1.1fr", "2.2fr", "1.6fr"),
            size="8pt"))


scale_adaptive_tables()


def structured_tables():
    """E11, the structural defence (code/structured_analysis.py)."""
    f, g = TAB / "exp_structured.csv", TAB / "exp_structured_search.csv"
    if not (f.exists() and g.exists()):
        return
    d, s = pd.read_csv(f, dtype=str), pd.read_csv(g, dtype=str)
    held = {(r.Evaluator, r.Pipeline): r["Held-out"] for _, r in s.iterrows()}
    cols = ["Second run, no attack (noise)", "Injection", "False authority", "T5 instruction update",
            "T3 actions relabelled as navigation", "Dry Run text (search, E10)"]
    def short(v):    # '5/22 = 22.7% [10.1%; 43.4%]' -> '22.7% (5/22)'
        m = re.match(r"(\d+/\d+) = ([\d.]+%)", str(v))
        return f"{m.group(2)} ({m.group(1)})" if m else esc(v)

    rows = []
    for _, r in d[d.Pipeline != "Original prompt"].iterrows():
        pipe = "guided" if r.Pipeline == "Guided prompt" else "structured"
        rows.append([esc(r.Evaluator), pipe, brci(r["κ"])] + [short(r[c]) for c in cols]
                    + [short(held.get((r.Evaluator, r.Pipeline), "—"))])
    write("tab_exp_structured", table(
        ["Evaluator", "Pipeline", "κ", "Second run, no attack", "Injection", "False authority", "T5", "T3", "\"Dry Run\"",
         "Adaptive search (held-out)"], rows,
        "The guided prompt and the structured pipeline (quarantined reader and an evaluator that never sees the agent's "
        "text) on the attack subset. κ re-weighted to the population [95% CI]. The other columns give the share of the "
        "side effects detected without attack that are missed in a second run without attack (noise floor) and under "
        "each attack; for the guided prompt, T5 and T3 were run on the side effects it detects, and the \"Dry Run\" "
        "text only against the structured pipeline.", "tab:exp-structured",
        align=("left", "left") + ("right",) * 8,
        columns=("1.15fr", "1.1fr", "1.15fr", "1.1fr", "1.1fr", "1.1fr", "1.1fr", "1.1fr", "1.1fr", "1.2fr"), size="7.5pt"))


structured_tables()
