"""Turn results/ into paper tables (LaTeX), figures (PDF/PNG) and results/summary.md.

    .venv/bin/python src/make_figures.py

The paper \\input{}s the generated tables, so every number in it comes from results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import GROUPS  # noqa: E402

CV, TEST = ROOT / "results" / "cv", ROOT / "results" / "test"
FIG, TAB = ROOT / "paper" / "figures", ROOT / "paper" / "tables"
LEVEL_NAMES = ["Elementary", "Intermediate", "Advanced"]
SYSTEM_NAMES = {"majority": "Majority class", "length": "Length only", "flesch_kincaid": "Flesch-Kincaid",
                "features_lr": "Feature LR (ours)", "features_lr_plus_length": "Feature LR + word count",
                "char_ngram_svm": "Char n-gram SVM"}
RANK_NAMES = {"length": "Longer text", "flesch_kincaid": "Higher FK grade",
              "classifier": "3-class LR (expected level)", "pairwise_lr": "Pairwise LR (ours)"}

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#E69F00", "#009E73", "#8C8C8C", "#D55E00"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 8, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
COL_W = 3.1  # ACL single column, inches


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def ci(c) -> str:
    return f"[{pct(c[0])}, {pct(c[1])}]"


def signed(x: float, tex: bool = True) -> str:
    s = f"{100 * x:+.1f}"
    return s.replace("-", "$-$") if tex else s


def savefig(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def write_tex(name: str, body: str) -> None:
    (TAB / f"{name}.tex").write_text(body)


# ---------------------------------------------------------------- tables
def systems_table(md: list[str]) -> None:
    has_test = (TEST / "systems.json").exists()
    test = load(TEST / "systems.json") if has_test else {}
    rows, mrows = [], []
    for key, name in SYSTEM_NAMES.items():
        r = load(CV / f"system_{key}__grouped.json")
        fa = np.array(r["fold_accuracy"])
        cv_cells = f"{pct(r['accuracy'])} $\\pm$ {pct(fa.std())} & {pct(r['macro_f1'])}"
        t = test.get(key)
        test_cells = (f"{pct(t['accuracy'])} {ci(t['accuracy_ci'])} & {pct(t['macro_f1'])}" if t else "-- & --")
        bold = key == "features_lr"
        line = f"{name} & {cv_cells} & {test_cells} \\\\"
        rows.append(line if not bold else line.replace(name, f"\\textbf{{{name}}}"))
        mrows.append(f"| {name} | {pct(r['accuracy'])} ± {pct(fa.std())} | {pct(r['macro_f1'])} | "
                     + (f"{pct(t['accuracy'])} {ci(t['accuracy_ci'])} | {pct(t['macro_f1'])} |" if t else "– | – |"))
    write_tex("systems", "\\begin{tabular}{lcccc}\n\\toprule\n"
              " & \\multicolumn{2}{c}{Dev, 5-fold grouped CV} & \\multicolumn{2}{c}{Held-out test} \\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
              "System & Acc. & F1 & Acc. [95\\% CI] & F1 \\\\\n\\midrule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    md += ["## Systems (accuracy %, macro-F1 %)", "",
           "| System | CV acc ± fold SD | CV F1 | Test acc [95% CI] | Test F1 |", "|---|---|---|---|---|", *mrows, ""]
    if has_test:
        md += ["Paired article-level bootstrap on the test set (main system minus other):", ""]
        for key, t in test.items():
            if "diff_accuracy" in t:
                md.append(f"- vs {SYSTEM_NAMES[key]}: {signed(t['diff_accuracy'], tex=False)} points "
                          f"{ci(t['diff_accuracy_ci'])}, P(main not better) = {t['p_main_not_better']:.3f}")
        md.append("")


def ablation_table(md: list[str]) -> pd.DataFrame:
    full = load(CV / "system_features_lr__grouped.json")
    rows = []
    for g, name in GROUPS.items():
        a, w = load(CV / f"ablation_only_{g}.json"), load(CV / f"ablation_without_{g}.json")
        rows.append({"group": name, "key": g, "n": None, "alone": a["accuracy"], "alone_ci": a["accuracy_ci"],
                     "alone_f1": a["macro_f1"], "without": w["accuracy"], "drop": w["diff_accuracy"],
                     "drop_ci": w["diff_accuracy_ci"]})
    df = pd.DataFrame(rows).sort_values("alone", ascending=False)
    plus = load(CV / "ablation_all_plus_length.json")
    from src.data import load_dataset  # feature counts per group
    from src.features import feature_columns
    data = load_dataset()
    df["n"] = [len(feature_columns(data, [g])) for g in df.key]
    lines = [f"{r.group} & {r.n} & {pct(r.alone)} & {pct(r.without)} & {signed(r.drop)} {ci(r.drop_ci)} \\\\"
             for r in df.itertuples()]
    lines.append("\\midrule")
    lines.append(f"All groups & {len(feature_columns(data))} & {pct(full['accuracy'])} & -- & -- \\\\")
    lines.append(f"All groups + word count & {len(feature_columns(data)) + 1} & {pct(plus['accuracy'])} & -- & "
                 f"{signed(plus['diff_accuracy'])} {ci(plus['diff_accuracy_ci'])} \\\\")
    write_tex("ablation", "\\begin{tabular}{lrccl}\n\\toprule\n"
              "Feature group & \\# & Alone & All but & $\\Delta$ vs.\\ all [95\\% CI] \\\\\n\\midrule\n"
              + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    md += ["## Ablation (grouped CV accuracy %)", "",
           "| Group | # feats | Alone | All but this | Δ vs all [95% CI] |", "|---|---|---|---|---|"]
    md += [f"| {r.group} | {r.n} | {pct(r.alone)} | {pct(r.without)} | {signed(r.drop, tex=False)} {ci(r.drop_ci)} |"
           for r in df.itertuples()]
    md += [f"| All groups | {len(feature_columns(data))} | {pct(full['accuracy'])} | – | – |",
           f"| All + word count | | {pct(plus['accuracy'])} | – | {signed(plus['diff_accuracy'], tex=False)} "
           f"{ci(plus['diff_accuracy_ci'])} |", ""]
    return df


def split_table(md: list[str]) -> pd.DataFrame:
    rows = []
    for key in ["length", "flesch_kincaid", "features_lr", "char_ngram_svm"]:
        g, s = load(CV / f"system_{key}__grouped.json"), load(CV / f"system_{key}__stratified.json")
        rows.append({"key": key, "name": SYSTEM_NAMES[key], "grouped": g["accuracy"], "stratified": s["accuracy"],
                     "g_ci": g["accuracy_ci"], "s_ci": s["accuracy_ci"],
                     "diff": s["diff_accuracy"], "diff_ci": s["diff_accuracy_ci"]})
    df = pd.DataFrame(rows)
    write_tex("splitcheck", "\\begin{tabular}{lccl}\n\\toprule\n"
              "System & Grouped & Stratified & $\\Delta$ [95\\% CI] \\\\\n\\midrule\n"
              + "\n".join(f"{r.name} & {pct(r.grouped)} & {pct(r.stratified)} & {signed(r.diff)} {ci(r.diff_ci)} \\\\"
                          for r in df.itertuples())
              + "\n\\bottomrule\n\\end{tabular}\n")
    md += ["## Split check (dev CV accuracy %)", "", "| System | Grouped | Stratified | Δ [95% CI] |", "|---|---|---|---|"]
    md += [f"| {r.name} | {pct(r.grouped)} | {pct(r.stratified)} | {signed(r.diff, tex=False)} {ci(r.diff_ci)} |"
           for r in df.itertuples()]
    md.append("")
    return df


def ranking_table(md: list[str]) -> None:
    order = ["length", "flesch_kincaid", "classifier", "pairwise_lr"]
    keys = ["ele-int", "int-adv", "ele-adv", "overall"]
    within, cross = load(CV / "ranking.json"), load(CV / "ranking_cross_article.json")
    t_within = load(TEST / "ranking.json") if (TEST / "ranking.json").exists() else {}
    t_cross = load(TEST / "ranking_cross_article.json") if (TEST / "ranking_cross_article.json").exists() else {}
    lines, mlines = [], []
    for m in order:
        name = RANK_NAMES[m] if m != "pairwise_lr" else f"\\textbf{{{RANK_NAMES[m]}}}"
        test_cells = [pct(t[m]["overall"]) if t else "--" for t in (t_within, t_cross)]
        cells = [pct(within[m][k]) for k in keys] + [pct(cross[m][k]) for k in keys] + test_cells
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        mlines.append(f"| {RANK_NAMES[m]} | " + " | ".join(c.replace("--", "–") for c in cells) + " |")
    write_tex("ranking", "\\begin{tabular}{lcccccccccc}\n\\toprule\n"
              " & \\multicolumn{4}{c}{Same article (dev CV)} & \\multicolumn{4}{c}{Different articles (dev CV)}"
              " & \\multicolumn{2}{c}{Test, all pairs} \\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}\n"
              "Ranker & E--I & I--A & E--A & All & E--I & I--A & E--A & All & Same & Diff. \\\\\n\\midrule\n"
              + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    md += ["## Pairwise ranking accuracy (%)", "",
           "| Ranker | Same art. E–I | I–A | E–A | All | Diff. art. E–I | I–A | E–A | All | Test same | Test diff. |",
           "|---|---|---|---|---|---|---|---|---|---|---|", *mlines, ""]


def coef_table(md: list[str]) -> None:
    c = pd.read_csv(CV / "coefficients_features_lr.csv", index_col=0)
    top = c.head(8)
    def num(x: float) -> str:
        return f"{x:+.2f}".replace("-", "$-$")

    write_tex("coefficients", "\\begin{tabular}{lrrr}\n\\toprule\n"
              "Feature & Ele & Adv & Adv$-$Ele \\\\\n\\midrule\n"
              + "\n".join(f"\\texttt{{{i.replace('_', chr(92) + '_')}}} & {num(r.ele)} & {num(r.adv)} & "
                          f"{num(r.adv_minus_ele)} \\\\" for i, r in top.iterrows())
              + "\n\\bottomrule\n\\end{tabular}\n")
    md += [f"## Largest standardised coefficients (main model trained on all dev, C={c.C.iloc[0]})", "",
           "| Feature | Ele | Int | Adv | Adv−Ele |", "|---|---|---|---|---|"]
    md += [f"| {i} | {r.ele:+.2f} | {r.int:+.2f} | {r.adv:+.2f} | {r.adv_minus_ele:+.2f} |" for i, r in c.head(12).iterrows()]
    md.append("")


# ---------------------------------------------------------------- figures
def fig_ablation(df: pd.DataFrame) -> None:
    full = load(CV / "system_features_lr__grouped.json")["accuracy"]
    df = df.sort_values("alone")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 1.9), sharey=True)
    y = np.arange(len(df))
    err = np.array([[r.alone - r.alone_ci[0], r.alone_ci[1] - r.alone] for r in df.itertuples()]).T
    axes[0].barh(y, 100 * df.alone, xerr=100 * err, color=BLUE, height=0.6, error_kw={"lw": 0.8})
    axes[0].axvline(100 * full, color=GREY, ls="--", lw=0.8)
    axes[0].text(100 * full, len(df) - 0.4, " all groups", color=GREY, va="bottom", fontsize=7)
    axes[0].axvline(100 / 3, color=GREY, ls=":", lw=0.8)
    axes[0].set_xlim(0, 85)
    axes[0].set_xlabel("Accuracy with this group alone (%)")
    err2 = np.array([[r.drop - r.drop_ci[0], r.drop_ci[1] - r.drop] for r in df.itertuples()]).T
    axes[1].barh(y, 100 * df["drop"], xerr=100 * err2, color=ORANGE, height=0.6, error_kw={"lw": 0.8})
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].set_xlabel("Change when this group is removed (points)")
    axes[0].set_yticks(y, df.group)
    savefig(fig, "ablation")


def fig_splitcheck(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(df))
    w = 0.38
    for off, col, ci_col, color, label in [(-w / 2, "grouped", "g_ci", BLUE, "Article-grouped folds"),
                                           (w / 2, "stratified", "s_ci", ORANGE, "Ordinary stratified folds")]:
        err = np.array([[r[col] - r[ci_col][0], r[ci_col][1] - r[col]] for _, r in df.iterrows()]).T
        ax.bar(x + off, 100 * df[col], w, yerr=100 * err, color=color, label=label, error_kw={"lw": 0.8})
    short = {"length": "Length\nonly", "flesch_kincaid": "Flesch-\nKincaid", "features_lr": "Feature\nLR (ours)",
             "char_ngram_svm": "Char\nSVM"}
    ax.set_xticks(x, [short[k] for k in df.key])
    ax.axhline(100 / 3, color=GREY, ls=":", lw=0.8)
    ax.text(len(df) - 0.5, 100 / 3 + 1.5, "chance", color=GREY, fontsize=6.5, ha="right")
    ax.set_ylabel("Dev CV accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    savefig(fig, "splitcheck")


def fig_confusion() -> None:
    panels = [("Dev (grouped CV)", load(CV / "system_features_lr__grouped.json")["confusion"])]
    if (TEST / "systems.json").exists():
        panels.append(("Held-out test", load(TEST / "systems.json")["features_lr"]["confusion"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_W * len(panels) * 0.75, 1.9))
    axes = np.atleast_1d(axes)
    for ax, (title, m) in zip(axes, panels):
        m = np.array(m)
        rate = m / m.sum(1, keepdims=True)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{m[i, j]}\n({100 * rate[i, j]:.0f}%)", ha="center", va="center", fontsize=7,
                        color="white" if rate[i, j] > 0.55 else "black")
        ax.set_xticks(range(3), ["Ele", "Int", "Adv"])
        ax.set_yticks(range(3), ["Ele", "Int", "Adv"])
        ax.set_xlabel("Predicted")
        ax.set_title(title)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("True")
    savefig(fig, "confusion")


def fig_ranking() -> None:
    order = ["length", "flesch_kincaid", "classifier", "pairwise_lr"]
    colors = [GREY, ORANGE, GREEN, BLUE]
    types = ["ele-int", "int-adv", "ele-adv"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 1.9), sharey=True)
    for ax, (title, f) in zip(axes, [("Same article", "ranking.json"),
                                     ("Different articles", "ranking_cross_article.json")]):
        cv = load(CV / f)
        x = np.arange(len(types))
        w = 0.2
        for i, (m, c) in enumerate(zip(order, colors)):
            ax.bar(x + (i - 1.5) * w, [100 * cv[m][t] for t in types], w, color=c, label=RANK_NAMES[m])
        ax.set_xticks(x, ["Ele vs Int", "Int vs Adv", "Ele vs Adv"])
        ax.set_ylim(50, 100)
        ax.set_title(title)
    axes[0].set_ylabel("Pairs ordered correctly (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.12))
    savefig(fig, "ranking")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    md = ["# Results summary", "", "Generated by `src/make_figures.py` from `results/`. Accuracy and F1 in %.", ""]
    systems_table(md)
    abl = ablation_table(md)
    split = split_table(md)
    ranking_table(md)
    coef_table(md)
    fig_ablation(abl)
    fig_splitcheck(split)
    fig_confusion()
    fig_ranking()
    (ROOT / "results" / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
