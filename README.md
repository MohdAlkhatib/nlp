# Reading-Level Classification on OneStopEnglish

Group 23 NLP project: classifying news texts as elementary, intermediate or advanced with
27 hand-built linguistic features, evaluated on article-grouped splits.

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) · **Slides:** [`slides/slides.pdf`](slides/slides.pdf) · **All results:** [`results/summary.md`](results/summary.md)

## Main results

| System | Dev CV acc. (grouped) | Test acc. [95% CI] |
|---|---|---|
| Majority class | 33.3 | 33.3 |
| Length only | 62.3 | 64.0 [57.0, 70.2] |
| Flesch-Kincaid | 51.7 | 56.1 [48.2, 64.0] |
| **Feature LR (ours, 27 features)** | **71.3** | **70.2 [62.3, 78.1]** |
| Feature LR + word count | 77.7 | 78.9 [70.2, 86.8] |
| Char n-gram SVM | 73.3 | 73.7 [68.4, 79.0] |

- **Feature groups:** sentence length (61.1% alone) and syntax (59.2%) are strongest alone; vocabulary
  is the only group whose removal clearly hurts (−3.8 points).
- **Split check:** ordinary stratified folds *lower* accuracy, because an article's other versions
  always have a different label. The char n-gram SVM falls from 73.3% to 25.8%, below chance.
- **Ranking:** ordering the 3 versions of one article is ~99% for every method, including "the
  longer text is harder"; across different articles the methods separate (83–89%).
- **Corpus problem:** the release's top-level intermediate files have all punctuation stripped,
  which leaks the label. We use the intact copy; see `data/processed/preprocessing_report.md`.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# NLTK data (on macOS python.org builds, prefix with SSL_CERT_FILE=$(.venv/bin/python -m certifi))
.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['cmudict','punkt','punkt_tab']]"
git clone --depth 1 https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git data/raw/OneStopEnglishCorpus

.venv/bin/python src/preprocess.py        # cleaned corpus + splits  -> data/processed/
.venv/bin/python src/features.py          # 27 features (~1 min)     -> data/processed/features.csv
.venv/bin/python src/run_experiments.py   # all dev CV experiments (~15 min) -> results/cv/
.venv/bin/python src/make_figures.py      # tables + figures -> paper/, results/summary.md
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

`src/run_experiments.py --final-test` trains on the 151 dev articles and evaluates once on the 38
test articles. It has already been run (`results/test/`) and refuses to run again.

## Layout

| Path | Contents |
|---|---|
| `src/preprocess.py` | Cleaning, corpus fixes, test split, grouped and stratified folds, ranking pairs |
| `src/features.py` | The 27 features (word length, sentence length, vocabulary, lexical variation, syntax, discourse) |
| `src/models.py` | Majority, length and Flesch-Kincaid threshold baselines, feature LR, char n-gram SVM |
| `src/evaluation.py` | CV with inner tuning, article-level bootstrap CIs |
| `src/run_experiments.py` | Baselines, ablation, split check, ranking, error analysis, final test |
| `src/data.py` | `load_dataset()`, `dev_test()`, `cv_folds()` |
| `data/processed/` | `corpus.csv`, `features.csv`, `pairs.csv`, `splits.json`, preprocessing report |
| `results/cv/`, `results/test/` | Raw results (JSON/CSV), incl. per-text predictions and error analysis |
| `paper/` | ACL-format paper; `tables/` and `figures/` are generated |

## Licence and attribution

The OneStopEnglish corpus is by Sowmya Vajjala and Ivana Lučić (2018) and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The files in `data/processed/` are
derived from it and are shared under the same licence.
