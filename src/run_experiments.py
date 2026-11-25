"""All experiments from the proposal.

    .venv/bin/python src/run_experiments.py               # experiments 1-5 on the dev set (CV)
    .venv/bin/python src/run_experiments.py --final-test  # experiment 6: train on dev, test ONCE

Results go to results/cv/ and results/test/ as JSON/CSV; src/make_figures.py turns them into
tables and figures. The final test refuses to run twice (delete results/test/ to override,
which you should not do once numbers have been reported).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluation as E  # noqa: E402
from src import models as M  # noqa: E402
from src.data import load_dataset, load_pairs  # noqa: E402
from src.features import GROUPS, feature_columns  # noqa: E402

RESULTS = ROOT / "results"
LEVELS = ["ele", "int", "adv"]


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def systems(df):
    allf = feature_columns(df)
    return {
        "majority": (M.majority(), None),
        "length": (M.length_only(), None),
        "flesch_kincaid": (M.flesch_kincaid(), None),
        "features_lr": (M.feature_lr(allf), M.LR_GRID),
        "char_ngram_svm": (M.char_ngram_svm(), M.SVM_GRID),
        # secondary system, added after the dev ablation showed word count adds signal
        "features_lr_plus_length": (M.feature_lr(allf + ["n_words"]), M.LR_GRID),
    }


def with_ci(dev: pd.DataFrame, res: dict, other: dict | None = None) -> dict:
    d = dev[["article_id", "label"]].copy()
    d["pred"] = dev.text_id.map(res["predictions"])
    if other is not None:
        d["other"] = dev.text_id.map(other["predictions"])
    res.update(E.bootstrap(d, "pred", "other" if other is not None else None))
    res["confusion"] = E.confusion(d.label, d.pred)
    return res


# ---------------------------------------------------------------- pairwise ranking
def ranking_model():
    return LogisticRegression(fit_intercept=False, max_iter=5000)


def pair_matrix(pairs: pd.DataFrame, Z: pd.DataFrame) -> np.ndarray:
    return Z.loc[pairs.text_id_a].to_numpy() - Z.loc[pairs.text_id_b].to_numpy()


def fit_ranker(train_texts: pd.DataFrame, train_pairs: pd.DataFrame, cols, all_texts: pd.DataFrame):
    """Scale features on the training texts, fit LR on feature differences (C tuned by article)."""
    scaler = StandardScaler().fit(train_texts[cols])
    Z = pd.DataFrame(scaler.transform(all_texts[cols]), index=all_texts.text_id, columns=cols)
    inner = list(GroupKFold(n_splits=5, shuffle=True, random_state=E.SEED)
                 .split(train_pairs, train_pairs.label, groups=train_pairs.article_id))
    search = GridSearchCV(ranking_model(), {"C": [0.001, 0.01, 0.1, 1, 10]}, cv=inner, n_jobs=-1)
    search.fit(pair_matrix(train_pairs, Z), train_pairs.label)
    return search.best_estimator_, Z


def score_rankers(pairs: pd.DataFrame, texts: pd.DataFrame, expected_level: pd.Series) -> pd.DataFrame:
    """Add predictions from the simple rankers that need no training on pairs."""
    t = texts.set_index("text_id")
    out = pairs.copy()
    for name, s in [("length", t.n_words), ("flesch_kincaid", t.fk_grade), ("classifier", expected_level)]:
        a, b = s.loc[out.text_id_a].to_numpy(), s.loc[out.text_id_b].to_numpy()
        out[f"pred_{name}"] = np.where(a == b, 0.5, (a > b).astype(float))  # ties count half
    return out


def ranking_summary(scored: pd.DataFrame) -> dict:
    methods = [c[5:] for c in scored.columns if c.startswith("pred_")]
    res = {}
    for m in methods:
        correct = 1 - (scored[f"pred_{m}"] - scored.label).abs()
        res[m] = {"overall": float(correct.mean()),
                  **{pt: float(correct[scored.pair_type == pt].mean()) for pt in ["ele-int", "int-adv", "ele-adv"]}}
    return res


def cross_article_pairs(texts: pd.DataFrame) -> pd.DataFrame:
    """All pairs of texts from DIFFERENT articles and different levels, harder text first.

    Within-article pairs are near ceiling; across articles, length and topic are not held fixed.
    """
    t = texts[["text_id", "article_id", "label", "level"]]
    m = t.merge(t, how="cross", suffixes=("_a", "_b"))
    m = m[(m.article_id_a != m.article_id_b) & (m.label_a > m.label_b)]
    return pd.DataFrame({"text_id_a": m.text_id_a.values, "text_id_b": m.text_id_b.values,
                         "pair_type": (m.level_b + "-" + m.level_a).values, "label": 1})


def expected_level(proba: dict) -> pd.Series:
    return pd.Series({t: float(np.dot(p, [0, 1, 2])) for t, p in proba.items()})


# ---------------------------------------------------------------- error analysis
def error_analysis(df: pd.DataFrame, dev: pd.DataFrame, cols) -> pd.DataFrame:
    """For each misclassified dev text (grouped CV): which features pushed toward the wrong class."""
    rows = []
    for k in sorted(dev.fold_grouped.unique()):
        tr, va = dev[dev.fold_grouped != k], dev[dev.fold_grouped == k]
        est = E.fit(M.feature_lr(cols), M.LR_GRID, tr, "grouped")
        scale, clf = est.named_steps["scale"], est.named_steps["clf"]
        Z = scale.transform(va[cols].to_numpy(dtype=float))
        pred = est.predict(va)
        for z, (_, r), p in zip(Z, va.iterrows(), pred):
            if p == r.label:
                continue
            push = z * (clf.coef_[p] - clf.coef_[r.label])     # >0: pushes toward the wrong class
            order = np.argsort(-push)
            rows.append({"text_id": r.text_id, "article_id": r.article_id, "true": LEVELS[r.label],
                         "pred": LEVELS[p], "n_words": int(r.n_words),
                         "fk_grade": round(r.fk_grade, 2),
                         "top_wrong_features": "; ".join(f"{cols[i]} ({push[i]:+.2f})" for i in order[:3]),
                         "top_right_features": "; ".join(f"{cols[i]} ({push[i]:+.2f})" for i in order[::-1][:3]),
                         "excerpt": r.text[:400].replace("\n", " ")})
    return pd.DataFrame(rows)


def coefficients(dev: pd.DataFrame, cols) -> pd.DataFrame:
    est = E.fit(M.feature_lr(cols), M.LR_GRID, dev, "grouped")
    coef = est.named_steps["clf"].coef_
    out = pd.DataFrame(coef.T, index=cols, columns=LEVELS)
    out["adv_minus_ele"] = out.adv - out.ele
    out["C"] = est.named_steps["clf"].C
    return out.sort_values("adv_minus_ele", key=abs, ascending=False)


# ---------------------------------------------------------------- stages
def run_cv(df: pd.DataFrame) -> None:
    dev = df[df.split == "dev"].reset_index(drop=True)
    out = RESULTS / "cv"
    allf = feature_columns(df)

    # 1-2. baselines and main system, grouped folds
    res = {}
    for name, (model, grid) in systems(df).items():
        log(f"grouped CV: {name}")
        res[name] = E.cross_validate(model, grid, dev, "grouped")
    for name in res:
        other = res["features_lr"] if name != "features_lr" else res["flesch_kincaid"]
        save(with_ci(dev, res[name], other), out / f"system_{name}__grouped.json")

    # 3. ablation: each group alone, all minus one group, all plus raw length
    abl = {}
    for g in GROUPS:
        log(f"ablation: {g}")
        abl[f"only_{g}"] = E.cross_validate(M.feature_lr(feature_columns(df, [g])), M.LR_GRID, dev)
        rest = [x for x in GROUPS if x != g]
        abl[f"without_{g}"] = E.cross_validate(M.feature_lr(feature_columns(df, rest)), M.LR_GRID, dev)
    abl["all_plus_length"] = E.cross_validate(M.feature_lr(allf + ["n_words"]), M.LR_GRID, dev)
    for name, r in abl.items():
        save(with_ci(dev, r, res["features_lr"]), out / f"ablation_{name}.json")

    # 4. split check: same systems on ordinary stratified folds (paired with grouped results)
    for name in ["length", "flesch_kincaid", "features_lr", "char_ngram_svm"]:
        log(f"stratified CV: {name}")
        model, grid = systems(df)[name]
        r = E.cross_validate(model, grid, dev, "stratified")
        save(with_ci(dev, r, res[name]), out / f"system_{name}__stratified.json")

    # 5. pairwise ranking (grouped folds over articles)
    log("ranking")
    pairs = load_pairs()
    dev_pairs = pairs[pairs.split == "dev"].reset_index(drop=True)
    scored, cross = [], []
    for k in sorted(dev.fold_grouped.unique()):
        tr_t = dev[dev.fold_grouped != k]
        tr_p, va_p = dev_pairs[dev_pairs.fold_grouped != k], dev_pairs[dev_pairs.fold_grouped == k].copy()
        ranker, Z = fit_ranker(tr_t, tr_p, allf, df)
        va_p["pred_pairwise_lr"] = ranker.predict(pair_matrix(va_p, Z)).astype(float)
        scored.append(va_p)
        cx = cross_article_pairs(dev[dev.fold_grouped == k])
        cx["pred_pairwise_lr"] = ranker.predict(pair_matrix(cx, Z)).astype(float)
        cross.append(cx)
    level = expected_level(res["features_lr"]["proba"])
    scored = score_rankers(pd.concat(scored), df, level)
    scored.to_csv(out / "ranking_predictions.csv", index=False)
    save(ranking_summary(scored), out / "ranking.json")
    save(ranking_summary(score_rankers(pd.concat(cross), df, level)), out / "ranking_cross_article.json")

    # error analysis + coefficients of the main model
    log("error analysis + coefficients")
    error_analysis(df, dev, allf).to_csv(out / "errors_features_lr.csv", index=False)
    coefficients(dev, allf).to_csv(out / "coefficients_features_lr.csv", float_format="%.4f")
    log("CV done")


def run_test(df: pd.DataFrame) -> None:
    out = RESULTS / "test"
    if out.exists():
        raise SystemExit(f"{out} exists: the final test has already been run once.")
    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    allf = feature_columns(df)

    preds, probas = {}, {}
    for name, (model, grid) in systems(df).items():
        log(f"final fit + test: {name}")
        est = E.fit(model, grid, dev, "grouped")
        preds[name] = est.predict(test)
        if hasattr(est, "predict_proba"):
            probas[name] = dict(zip(test.text_id, est.predict_proba(test).tolist()))
        if grid:
            log(f"  chosen {[est.get_params()[p] for p in grid]}")
    summary = {}
    for name, p in preds.items():
        d = test[["article_id", "label"]].copy()
        d["pred"], d["main"] = p, preds["features_lr"]
        other = "main" if name != "features_lr" else None
        summary[name] = {**E.metrics(d.label, d.pred), **E.bootstrap(d, "pred", other),
                         "confusion": E.confusion(d.label, d.pred),
                         "predictions": dict(zip(test.text_id, map(int, p)))}
        if other:  # diff is reported as main - this system
            for key in ["diff_accuracy"]:
                summary[name][key] = -summary[name][key]
            for key in ["diff_accuracy_ci", "diff_macro_f1_ci"]:
                lo, hi = summary[name][key]
                summary[name][key] = [-hi, -lo]
            summary[name]["p_main_not_better"] = 1 - summary[name].pop("p_diff_le_0")
    save(summary, out / "systems.json")

    # ranking on the test articles
    pairs = load_pairs()
    dev_pairs = pairs[pairs.split == "dev"].reset_index(drop=True)
    test_pairs = pairs[pairs.split == "test"].reset_index(drop=True)
    ranker, Z = fit_ranker(dev, dev_pairs, allf, df)
    test_pairs["pred_pairwise_lr"] = ranker.predict(pair_matrix(test_pairs, Z)).astype(float)
    level = expected_level(probas["features_lr"])
    scored = score_rankers(test_pairs, df, level)
    scored.to_csv(out / "ranking_predictions.csv", index=False)
    save(ranking_summary(scored), out / "ranking.json")
    cx = cross_article_pairs(test)
    cx["pred_pairwise_lr"] = ranker.predict(pair_matrix(cx, Z)).astype(float)
    save(ranking_summary(score_rankers(cx, df, level)), out / "ranking_cross_article.json")
    log("test done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="train on dev, evaluate once on test")
    args = ap.parse_args()
    df = load_dataset()
    run_test(df) if args.final_test else run_cv(df)


if __name__ == "__main__":
    main()
