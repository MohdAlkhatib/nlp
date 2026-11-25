"""Hand-built linguistic features, one row per text.

Run from the project root (after src/preprocess.py):
    .venv/bin/python src/features.py

Output: data/processed/features.csv  (text_id + one column per feature)

Every feature column starts with its group prefix, so experiments can select groups by name:
    wl_   word length          NLTK tokens, CMU Pronouncing Dictionary
    sl_   sentence length      NLTK sentence splitter
    voc_  vocabulary           wordfreq
    lex_  lexical variation    type-token ratios, spaCy POS tags
    syn_  syntax               spaCy dependency parse
    disc_ discourse            connective word list

No feature depends on text length by construction (all are averages, rates or ratios),
except sl_max_words, which is listed in the proposal; see the paper for discussion.
The Flesch-Kincaid grade (fk_grade) and word count (n_words) are saved too, for the baselines,
but they carry no group prefix and are not part of the feature model.
"""

from __future__ import annotations

import re
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
from wordfreq import top_n_list, zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import PROCESSED, load_corpus  # noqa: E402

GROUPS = {
    "wl": "Word length",
    "sl": "Sentence length",
    "voc": "Vocabulary",
    "lex": "Lexical variation",
    "syn": "Syntax",
    "disc": "Discourse",
}

WORD_RE = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
CMU = cmudict.dict()
TOP3000 = set(top_n_list("en", 3000))

# Hand-compiled list of frequent English explicit connectives,
# grouped by relation type. Multi-word items are matched as phrases.
CONNECTIVES = {
    "contrast": ["however", "although", "though", "even though", "whereas", "while",
                 "nevertheless", "nonetheless", "despite", "in spite of", "on the other hand",
                 "instead", "yet", "but", "in contrast", "on the contrary", "still"],
    "cause": ["because", "since", "therefore", "thus", "hence", "consequently",
              "as a result", "so that", "due to", "accordingly", "for this reason"],
    "addition": ["moreover", "furthermore", "in addition", "also", "besides",
                 "additionally", "as well as", "likewise", "similarly"],
    "time": ["meanwhile", "afterwards", "subsequently", "until", "before", "after",
             "when", "once", "eventually", "previously"],
}
CONN_RE = {k: re.compile(r"\b(" + "|".join(re.escape(c) for c in sorted(v, key=len, reverse=True)) + r")\b")
           for k, v in CONNECTIVES.items()}

SUBORDINATE_DEPS = {"advcl", "ccomp", "csubj", "csubjpass", "acl", "relcl"}
PASSIVE_DEPS = {"nsubjpass", "auxpass", "csubjpass"}


# ---------------------------------------------------------------- word helpers
def words_of(tokens: list[str]) -> list[str]:
    return [t for t in tokens if WORD_RE.match(t) and t.lower() != "n't"]


def _vowel_groups(word: str) -> int:
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1  # silent final e: "make", "home"
    return max(n, 1)


@lru_cache(maxsize=None)
def syllables(word: str) -> int:
    """CMU dictionary first (first pronunciation), vowel-group fallback for unknown words."""
    total = 0
    for part in word.lower().split("-"):
        if not part:
            continue
        prons = CMU.get(part) or CMU.get(part.replace("'", ""))
        total += sum(ph[-1].isdigit() for ph in prons[0]) if prons else _vowel_groups(part)
    return max(total, 1)


def cmu_coverage(words: list[str]) -> float:
    return np.mean([w.lower() in CMU for w in words])


# ---------------------------------------------------------------- spaCy helpers
def tree_depth(token) -> int:
    depth = 0
    while token.head.i != token.i:
        token = token.head
        depth += 1
    return depth


# ---------------------------------------------------------------- feature extraction
def extract(text: str, doc) -> dict:
    paragraphs = text.split("\n")
    sentences = [s for p in paragraphs for s in sent_tokenize(p)]
    sent_words = [words_of(word_tokenize(s)) for s in sentences]
    sent_words = [w for w in sent_words if w]  # drop sentences with no words
    words = [w for s in sent_words for w in s]
    lower = [w.lower() for w in words]
    syl = np.array([syllables(w) for w in words])
    sent_lens = np.array([len(s) for s in sent_words])
    n_sent = len(sent_words)

    f = {}
    # word length
    f["wl_syllables_per_word"] = syl.mean()
    f["wl_pct_3plus_syllables"] = 100 * (syl >= 3).mean()
    f["wl_chars_per_word"] = np.mean([len(w) for w in words])
    # sentence length
    f["sl_words_per_sentence"] = sent_lens.mean()
    f["sl_max_words"] = sent_lens.max()
    f["sl_sd_words"] = sent_lens.std()
    # vocabulary
    zipf = np.array([zipf_frequency(w, "en") for w in lower])
    f["voc_pct_outside_top3000"] = 100 * np.mean([w not in TOP3000 for w in lower])
    f["voc_mean_zipf"] = zipf.mean()
    f["voc_pct_rare"] = 100 * (zipf < 3).mean()  # rarer than ~1 per million words
    # lexical variation
    f["lex_ttr_first100"] = len(set(lower[:100])) / len(lower[:100])
    windows = [lower[i:i + 100] for i in range(0, len(lower) - 99, 100)]
    f["lex_msttr100"] = np.mean([len(set(w)) / 100 for w in windows])
    content = [t for t in doc if not (t.is_punct or t.is_space)]
    pos = pd.Series([t.pos_ for t in content]).value_counts(normalize=True)
    for tag, name in [("NOUN", "noun"), ("VERB", "verb"), ("ADJ", "adj"), ("ADV", "adv"), ("PROPN", "propn")]:
        f[f"lex_{name}_ratio"] = pos.get(tag, 0.0)
    f["lex_lexical_density"] = sum(pos.get(t, 0.0) for t in ("NOUN", "VERB", "ADJ", "ADV"))
    # syntax (spaCy sentences)
    sents = [s for s in doc.sents if any(not t.is_punct and not t.is_space for t in s)]
    f["syn_parse_depth"] = np.mean([max(tree_depth(t) for t in s) for s in sents])
    f["syn_subord_per_sentence"] = np.mean([sum(t.dep_ in SUBORDINATE_DEPS for t in s) for s in sents])
    f["syn_passive_rate"] = np.mean([any(t.dep_ in PASSIVE_DEPS for t in s) for s in sents])
    f["syn_dep_distance"] = np.mean([abs(t.i - t.head.i) for t in content if t.head.i != t.i])
    f["syn_np_premods"] = np.mean([sum(c.dep_ in ("amod", "compound", "nummod") for c in t.children)
                                   for t in content if t.pos_ in ("NOUN", "PROPN")] or [0])
    # discourse
    low_text = text.lower()
    counts = {k: len(r.findall(low_text)) for k, r in CONN_RE.items()}
    f["disc_connectives_per_sentence"] = sum(counts.values()) / n_sent
    for k, v in counts.items():
        f[f"disc_{k}_per_sentence"] = v / n_sent

    # baseline inputs (no group prefix)
    f["n_words"] = len(words)
    f["n_sentences"] = n_sent
    f["fk_grade"] = 0.39 * len(words) / n_sent + 11.8 * syl.sum() / len(words) - 15.59
    f["cmu_coverage"] = cmu_coverage(words)
    return f


def feature_columns(df: pd.DataFrame, groups: list[str] | None = None) -> list[str]:
    groups = groups or list(GROUPS)
    return [c for c in df.columns if c.split("_", 1)[0] in groups]


def main() -> None:
    corpus = load_corpus()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    rows = []
    for i, (row, doc) in enumerate(zip(corpus.itertuples(), nlp.pipe(corpus.text, batch_size=16))):
        rows.append({"text_id": row.text_id, **extract(row.text, doc)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(corpus)}", flush=True)
    feats = pd.DataFrame(rows)
    assert feats.notna().all().all(), feats.columns[feats.isna().any()].tolist()
    feats.to_csv(PROCESSED / "features.csv", index=False, float_format="%.6g")
    print(f"features.csv: {feats.shape[0]} texts x {len(feature_columns(feats))} features "
          f"| mean CMU coverage {feats.cmu_coverage.mean():.1%}")


if __name__ == "__main__":
    main()
