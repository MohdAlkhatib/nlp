"""Load the preprocessed corpus and iterate over the agreed splits.

    from src.data import load_corpus, cv_folds, dev_test

    df = load_corpus()
    dev, test = dev_test(df)
    for k, train, val in cv_folds(df):              # article-grouped (default)
        ...
    for k, train, val in cv_folds(df, "stratified"):  # split-check experiment only

Run `python src/preprocess.py` first to create data/processed/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
LEVELS = ["ele", "int", "adv"]


def load_corpus(path: Path | str = PROCESSED / "corpus.csv") -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def load_pairs(path: Path | str = PROCESSED / "pairs.csv") -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df.split == "dev"].copy(), df[df.split == "test"].copy()


def cv_folds(df: pd.DataFrame, scheme: str = "grouped") -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train, val) over the dev set. Test texts are never included."""
    col = {"grouped": "fold_grouped", "stratified": "fold_stratified"}[scheme]
    dev = df[df.split == "dev"]
    for k in sorted(dev[col].unique()):
        yield int(k), dev[dev[col] != k].copy(), dev[dev[col] == k].copy()
