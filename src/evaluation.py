"""Cross-validation, metrics and article-level bootstrap shared by all experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

SEED = 42
N_BOOT = 10_000


def metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def fit(model, grid, train: pd.DataFrame, scheme: str):
    """Fit, tuning `grid` with an inner 5-fold CV of the same kind as the outer split."""
    if not grid:
        return clone(model).fit(train, train.label)
    if scheme == "grouped":
        inner = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                     .split(train, train.label, groups=train.article_id))
    else:
        inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, train.label))
    search = GridSearchCV(clone(model), grid, cv=inner, scoring="accuracy", n_jobs=-1)
    return search.fit(train, train.label).best_estimator_


def cross_validate(model, grid, dev: pd.DataFrame, scheme: str = "grouped") -> dict:
    """Out-of-fold predictions on the dev set for one system and one split scheme."""
    col = {"grouped": "fold_grouped", "stratified": "fold_stratified"}[scheme]
    pred = pd.Series(-1, index=dev.index)
    proba = pd.DataFrame(np.nan, index=dev.index, columns=[0, 1, 2])
    folds, chosen = [], []
    for k in sorted(dev[col].unique()):
        tr, va = dev[dev[col] != k], dev[dev[col] == k]
        est = fit(model, grid, tr, scheme)
        pred[va.index] = est.predict(va)
        if hasattr(est, "predict_proba"):
            proba.loc[va.index] = est.predict_proba(va)
        folds.append(metrics(va.label, pred[va.index]))
        if grid:
            chosen.append({p: est.get_params()[p] for p in grid})
    out = {"scheme": scheme, **metrics(dev.label, pred),
           "fold_accuracy": [f["accuracy"] for f in folds],
           "fold_macro_f1": [f["macro_f1"] for f in folds],
           "chosen_params": chosen,
           "predictions": dict(zip(dev.text_id, pred.astype(int).tolist()))}
    if proba.notna().all().all():
        out["proba"] = {t: p for t, p in zip(dev.text_id, proba.round(5).values.tolist())}
    return out


# ---------------------------------------------------------------- bootstrap
def _article_confusions(df: pd.DataFrame, pred_col: str) -> tuple[np.ndarray, list[str]]:
    articles = sorted(df.article_id.unique())
    idx = {a: i for i, a in enumerate(articles)}
    conf = np.zeros((len(articles), 3, 3), dtype=np.int32)
    np.add.at(conf, (df.article_id.map(idx).to_numpy(), df.label.to_numpy(), df[pred_col].to_numpy()), 1)
    return conf, articles


def _scores(conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """conf: (..., 3, 3) summed confusion matrices -> accuracy, macro-F1."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    total = conf.sum(axis=(-2, -1))
    acc = tp.sum(-1) / total
    denom = conf.sum(-1) + conf.sum(-2)                 # true count + predicted count = 2TP+FP+FN
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return acc, f1.mean(-1)


def bootstrap(df: pd.DataFrame, pred_col: str, other_col: str | None = None, n: int = N_BOOT) -> dict:
    """95% CIs by resampling whole articles (all versions of an article stay together).

    With `other_col`, also returns the CI of the paired difference pred_col - other_col.
    """
    rng = np.random.default_rng(SEED)
    conf, articles = _article_confusions(df, pred_col)
    samples = rng.integers(0, len(articles), size=(n, len(articles)))
    acc, f1 = _scores(conf[samples].sum(1))
    out = {"accuracy_ci": np.percentile(acc, [2.5, 97.5]).round(4).tolist(),
           "macro_f1_ci": np.percentile(f1, [2.5, 97.5]).round(4).tolist()}
    if other_col:
        conf2, _ = _article_confusions(df, other_col)
        acc2, f12 = _scores(conf2[samples].sum(1))
        d_acc, d_f1 = acc - acc2, f1 - f12
        out.update({"diff_accuracy": float(_scores(conf.sum(0))[0] - _scores(conf2.sum(0))[0]),
                    "diff_accuracy_ci": np.percentile(d_acc, [2.5, 97.5]).round(4).tolist(),
                    "diff_macro_f1_ci": np.percentile(d_f1, [2.5, 97.5]).round(4).tolist(),
                    "p_diff_le_0": float((d_acc <= 0).mean())})
    return out


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
