# Reading-Level Classification on OneStopEnglish

Group 23 NLP project: classifying news texts as elementary, intermediate or advanced with
hand-built linguistic features, evaluated on article-grouped splits.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone --depth 1 https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git data/raw/OneStopEnglishCorpus
.venv/bin/python src/preprocess.py   # regenerates data/processed/
```

The processed data is already committed, so the last two steps are only needed to rebuild it.

## Data

| File | Contents |
|---|---|
| `data/processed/corpus.csv` | 567 cleaned texts with `level`, `label` (0/1/2), `split` (dev/test), `fold_grouped`, `fold_stratified` |
| `data/processed/pairs.csv` | 1,134 ordered pairs for the ranking experiment (`label = 1`: text A is harder) |
| `data/processed/splits.json` | Seed, the 38 test articles, and the articles in each grouped fold |
| `data/processed/preprocessing_report.md` | Corpus problems found and fixed, statistics, split sizes |

**Read the report before using the raw corpus.** The top-level intermediate files have all
punctuation stripped, which leaks the label, and one elementary file is a duplicate of another article.

```python
from src.data import load_corpus, dev_test, cv_folds

df = load_corpus()
dev, test = dev_test(df)                       # test: touch once, at the very end
for k, train, val in cv_folds(df):             # article-grouped folds (default)
    ...
for k, train, val in cv_folds(df, "stratified"):  # split-check experiment only
    ...
```

## Licence and attribution

The OneStopEnglish corpus is by Sowmya Vajjala and Ivana Lučić (2018) and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files in `data/processed/` are
derived from it and are shared under the same licence.
