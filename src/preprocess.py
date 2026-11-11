"""Build the cleaned OneStopEnglish corpus and the article-grouped splits.

Run from the project root:
    .venv/bin/python src/preprocess.py

Inputs:  data/raw/OneStopEnglishCorpus  (git clone of nishkalavallabhi/OneStopEnglishCorpus)
Outputs: data/processed/corpus.csv          one row per text (567), with split + fold columns
         data/processed/pairs.csv           ordered pairs for the ranking experiment (1,134)
         data/processed/splits.json         seed, test articles, per-fold article lists
         data/processed/preprocessing_report.md

Corpus problems handled here (see the report for details):
  1. Texts-SeparatedByReadingLevel/Int-Txt/*.txt have every non-ASCII character stripped
     (no curly quotes, apostrophes, £, é ...), while Ele and Adv keep them. Using those files
     would let a model spot "intermediate" from punctuation alone. The nested copy
     Int-Txt/Int-Txt/*.txt is intact, matches the authors' CSV export, and is what we use.
  2. Ele-Txt/Royal Baby-ele.txt is a byte-identical copy of Skydiver-ele.txt. We rebuild the
     real Royal Baby elementary text from Texts-Together-OneCSVperFile/Royal Baby.csv.
  3. Mixed encodings (UTF-8 with/without BOM, one cp1252 file) and PDF ligatures (ﬁ) that
     appear only in Ele/Adv. Everything is decoded and Unicode-normalised the same way.
"""

from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "OneStopEnglishCorpus"
OUT = ROOT / "data" / "processed"

SEED = 42
N_TEST_ARTICLES = 38
N_FOLDS = 5

LEVELS = ["ele", "int", "adv"]
LEVEL_NAMES = {"ele": "elementary", "int": "intermediate", "adv": "advanced"}
LABEL = {"ele": 0, "int": 1, "adv": 2}
LEVEL_DIRS = {
    "ele": Path("Texts-SeparatedByReadingLevel/Ele-Txt"),
    "int": Path("Texts-SeparatedByReadingLevel/Int-Txt/Int-Txt"),  # intact copy, see docstring
    "adv": Path("Texts-SeparatedByReadingLevel/Adv-Txt"),
}
# (article, level) -> CSV column to rebuild the text from
CSV_OVERRIDES = {("Royal Baby", "ele"): 0}

QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                           "“": '"', "”": '"', "„": '"', "‟": '"',
                           "′": "'", "″": '"'})
HEADER_RE = re.compile(r"^\s*(elementary|intermediate|advanced)\s*$", re.I)


# ---------------------------------------------------------------- reading + cleaning
def decode(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def clean_text(text: str) -> str:
    """Same normalisation for every level, so no level carries formatting artifacts."""
    text = unicodedata.normalize("NFKC", text)        # ﬁ -> fi, NBSP -> space, … -> ...
    text = text.translate(QUOTE_MAP)                  # curly quotes -> ASCII (helps CMU dict lookups)
    text = text.replace("­", "")                 # soft hyphens
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and HEADER_RE.match(lines[0]):           # level-name header line would leak the label
        lines = lines[1:]
    return "\n".join(lines)                           # one paragraph per line


def read_csv_column(article: str, col: int) -> str:
    raw, _ = decode((RAW / "Texts-Together-OneCSVperFile" / f"{article}.csv").read_bytes())
    rows = list(csv.reader(io.StringIO(raw)))
    # each CSV row is one paragraph, hard-wrapped inside the cell
    return "\n".join(" ".join(r[col].split()) for r in rows[1:] if len(r) > col and r[col].strip())


def load_raw() -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    rows = []
    for level, rel in LEVEL_DIRS.items():
        files = sorted((RAW / rel).glob("*.txt"))
        for f in files:
            suffix = f"-{level}.txt"
            assert f.name.lower().endswith(suffix), f"unexpected file name {f}"
            source_name = f.name[: -len(suffix)]
            article = source_name.strip()
            raw_bytes = f.read_bytes()
            text, enc = decode(raw_bytes)
            source = str(rel / f.name)
            if (source_name, level) in CSV_OVERRIDES:
                text = read_csv_column(source_name, CSV_OVERRIDES[(source_name, level)])
                source = f"Texts-Together-OneCSVperFile/{source_name}.csv (col {CSV_OVERRIDES[(source_name, level)]})"
                notes.append(f"Rebuilt `{article}` / {level} from the CSV export (txt file was a duplicate).")
            if enc != "utf-8":
                notes.append(f"`{source}` was {enc}-encoded.")
            first = text.lstrip("﻿").splitlines()[0] if text.strip() else ""
            if HEADER_RE.match(first):
                notes.append(f"Removed level header line from `{source}`.")
            rows.append(dict(article_id=article, level=level, source_file=source,
                             raw_text=text))
    return pd.DataFrame(rows), notes


# ---------------------------------------------------------------- checks
def non_ascii_rate(texts: pd.Series) -> float:
    chars = sum(len(t) for t in texts)
    return 1000 * sum(sum(ord(c) > 127 for c in t) for t in texts) / chars


def validate(df: pd.DataFrame) -> None:
    per_article = df.groupby("article_id")["level"].apply(set)
    bad = per_article[per_article.apply(lambda s: s != set(LEVELS))]
    assert bad.empty, f"articles missing a level: {bad.to_dict()}"
    assert (df.level.value_counts() == 189).all(), df.level.value_counts()
    assert df.article_id.nunique() == 189
    assert (df.n_words > 100).all(), df.loc[df.n_words <= 100, "text_id"].tolist()
    assert not df.text.str.contains("�").any(), "undecodable characters left"
    dup = df[df.text.duplicated(keep=False)]
    assert dup.empty, f"duplicate texts: {dup.text_id.tolist()}"
    assert not df.text.str.match(HEADER_RE.pattern, case=False).any()


# ---------------------------------------------------------------- splits
def make_splits(df: pd.DataFrame) -> dict:
    articles = np.array(sorted(df.article_id.unique()))
    rng = np.random.default_rng(SEED)
    test_articles = sorted(rng.permutation(articles)[:N_TEST_ARTICLES].tolist())
    df["split"] = np.where(df.article_id.isin(test_articles), "test", "dev")

    dev = df[df.split == "dev"]
    df["fold_grouped"] = -1
    df["fold_stratified"] = -1
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(gkf.split(dev, dev.label, groups=dev.article_id)):
        df.loc[dev.index[va], "fold_grouped"] = k
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(dev, dev.label)):
        df.loc[dev.index[va], "fold_stratified"] = k

    # sanity: grouped folds never split an article; test articles never in any fold
    assert (df[df.split == "dev"].groupby("article_id").fold_grouped.nunique() == 1).all()
    assert (df.loc[df.split == "test", ["fold_grouped", "fold_stratified"]] == -1).all().all()
    assert not set(test_articles) & set(dev.article_id)

    folds = {k: sorted(df.loc[df.fold_grouped == k, "article_id"].unique().tolist())
             for k in range(N_FOLDS)}
    return {"seed": SEED, "n_test_articles": N_TEST_ARTICLES, "n_folds": N_FOLDS,
            "test_articles": test_articles, "grouped_fold_articles": folds}


def leakage_of_stratified(df: pd.DataFrame) -> float:
    """Share of validation texts whose article also has a version in that fold's training part."""
    dev = df[df.split == "dev"]
    leaked = 0
    for k in range(N_FOLDS):
        va = dev[dev.fold_stratified == k]
        tr_articles = set(dev.loc[dev.fold_stratified != k, "article_id"])
        leaked += va.article_id.isin(tr_articles).sum()
    return leaked / len(dev)


def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for article, g in df.groupby("article_id", sort=True):
        by_level = g.set_index("level")
        for lo, hi in [("ele", "int"), ("int", "adv"), ("ele", "adv")]:
            a, b = by_level.loc[lo], by_level.loc[hi]
            for first, second in [(a, b), (b, a)]:
                rows.append(dict(
                    article_id=article, pair_type=f"{lo}-{hi}",
                    text_id_a=first.text_id, text_id_b=second.text_id,
                    level_a=first.name, level_b=second.name,
                    label=int(LABEL[first.name] > LABEL[second.name]),  # 1 = A is harder
                    split=first.split, fold_grouped=int(first.fold_grouped)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def write_report(df, splits, pairs, notes, leak, ascii_before) -> None:
    def table(frame: pd.DataFrame) -> str:
        cols = [str(c) for c in frame.columns]
        lines = ["| " + " | ".join([frame.index.name or ""] + cols) + " |",
                 "|" + "---|" * (len(cols) + 1)]
        for idx, row in frame.astype(object).iterrows():
            lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
        return "\n".join(lines)

    stats = (df.groupby("level").agg(texts=("text_id", "size"),
                                      mean_words=("n_words", "mean"),
                                      min_words=("n_words", "min"),
                                      max_words=("n_words", "max"),
                                      mean_paragraphs=("n_paragraphs", "mean"))
             .loc[LEVELS].round(1).astype({"texts": int, "min_words": int, "max_words": int}))
    split_counts = pd.crosstab(df.split, df.level)[LEVELS]
    fold_counts = pd.crosstab(df.loc[df.split == "dev", "fold_grouped"],
                              df.loc[df.split == "dev", "level"])[LEVELS]
    fold_counts["articles"] = df[df.split == "dev"].groupby("fold_grouped").article_id.nunique()
    nonascii = pd.DataFrame({"raw files": ascii_before,
                             "after cleaning": {lv: round(non_ascii_rate(df.loc[df.level == lv, "text"]), 2)
                                                for lv in LEVELS}}).loc[LEVELS]
    nonascii.index.name = "level"

    md = f"""# OneStopEnglish preprocessing report

Generated by `src/preprocess.py` (seed {SEED}).

## Corpus problems found and fixed

1. **Intermediate files had all non-ASCII characters stripped.** `Int-Txt/*.txt` contain no curly
   quotes, apostrophes, £, é, etc. (e.g. `what’s` became `whats`), while Ele and Adv keep them.
   A classifier, especially the character n-gram SVM, could have identified "intermediate" from
   punctuation alone. The nested copy `Int-Txt/Int-Txt/*.txt` keeps its punctuation and is the
   closer match to the authors' CSV export (157 of 189 articles), so we use it. The content of the
   two copies is otherwise the same.
2. **`Royal Baby-ele.txt` was a copy of `Skydiver-ele.txt`.** The elementary Royal Baby text is
   rebuilt from `Texts-Together-OneCSVperFile/Royal Baby.csv`. On 58 other articles, rebuilding
   from the CSV this way matches the .txt files with a median token similarity of 1.00.
3. **Encoding and PDF artifacts.** Files are UTF-8 (with and without BOM) and one is cp1252; Ele and
   Adv contain `ﬁ` ligatures that Int does not. All texts are decoded, NFKC-normalised, and
   curly quotes mapped to ASCII the same way.
4. **No cross-article duplicates.** The highest TF-IDF cosine between two *different* articles
   (advanced versions) is 0.35 (shared topic, not shared text), so the file name is a safe group ID.

Per-file notes:
""" + "\n".join(f"- {n}" for n in notes) + f"""

### Non-ASCII characters per 1,000 characters

This checks that no level differs in formatting. Raw uses the top-level `Int-Txt` files.

{table(nonascii)}

## Cleaned corpus

{len(df)} texts, {df.article_id.nunique()} articles. Paragraphs are kept, one per line.

{table(stats)}

## Splits

- **Test:** {N_TEST_ARTICLES} articles ({(df.split == 'test').sum()} texts), chosen at random with seed {SEED}. Untouched until the final evaluation.
- **Dev:** {df[df.split == 'dev'].article_id.nunique()} articles ({(df.split == 'dev').sum()} texts), split into {N_FOLDS} article-grouped folds (`fold_grouped`, `GroupKFold(shuffle=True)`).
- **Split-check folds:** `fold_stratified` is an ordinary `StratifiedKFold` over the same dev texts.
  Under it, **{leak:.1%}** of validation texts have another version of the same article in training.
  Under `fold_grouped`, that figure is 0%.

{table(split_counts)}

Grouped folds:

{table(fold_counts)}

## Pairs for ranking

{len(pairs)} ordered pairs (3 per article × 2 orders × 189 articles). `label = 1` means text A is
harder. Labels are balanced ({pairs.label.mean():.2f}); pair types: {dict(Counter(pairs.pair_type))}.
Each pair carries its article's `split` and `fold_grouped`, so pairs never cross a split.
"""
    (OUT / "preprocessing_report.md").write_text(md)


# ---------------------------------------------------------------- main
def main() -> None:
    if not RAW.exists():
        raise SystemExit("Corpus not found. Run:\n  git clone --depth 1 "
                         "https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git "
                         "data/raw/OneStopEnglishCorpus")
    OUT.mkdir(parents=True, exist_ok=True)

    df, notes = load_raw()

    # artifact check on the raw files, including the broken top-level Int copy for comparison
    top_int = [decode(f.read_bytes())[0] for f in
               sorted((RAW / "Texts-SeparatedByReadingLevel/Int-Txt").glob("*.txt"))]
    ascii_before = {lv: round(non_ascii_rate(df.loc[df.level == lv, "raw_text"]), 2) for lv in ("ele", "adv")}
    ascii_before["int"] = round(non_ascii_rate(pd.Series(top_int)), 2)

    df["text"] = df.raw_text.map(clean_text)
    df = df.drop(columns="raw_text")
    df["text_id"] = df.article_id + "__" + df.level
    df["level_name"] = df.level.map(LEVEL_NAMES)
    df["label"] = df.level.map(LABEL)
    df["n_words"] = df.text.str.split().str.len()
    df["n_paragraphs"] = df.text.str.count("\n") + 1
    df = df.sort_values(["article_id", "label"]).reset_index(drop=True)
    validate(df)

    splits = make_splits(df)
    leak = leakage_of_stratified(df)
    pairs = make_pairs(df)
    assert len(pairs) == 1134 and pairs.label.mean() == 0.5

    cols = ["text_id", "article_id", "level", "level_name", "label", "split",
            "fold_grouped", "fold_stratified", "n_words", "n_paragraphs", "source_file", "text"]
    df[cols].to_csv(OUT / "corpus.csv", index=False)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2))
    write_report(df, splits, pairs, notes, leak, ascii_before)

    print(f"corpus.csv: {len(df)} texts | test {(df.split == 'test').sum()} | dev {(df.split == 'dev').sum()}")
    print(f"pairs.csv: {len(pairs)} pairs | stratified-fold leakage: {leak:.1%}")
    print(f"report: {OUT / 'preprocessing_report.md'}")


if __name__ == "__main__":
    main()
