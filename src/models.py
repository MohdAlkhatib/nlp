"""The five systems, as scikit-learn estimators that take the merged corpus+features DataFrame.

Every model gets the same input (one row per text with `text`, `n_words`, `fk_grade` and all
feature columns) and picks the columns it needs, so the same CV code runs all of them.
Models with a hyper-parameter are wrapped in GridSearchCV; the inner CV is built per outer fold
by `src.evaluation` so it follows the same split scheme (grouped or stratified).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

LR_GRID = {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10]}
SVM_GRID = {"clf__C": [0.1, 1, 10, 100]}


def select(cols):
    cols = list(cols)
    return FunctionTransformer(lambda X: X[cols].to_numpy(dtype=float), validate=False)


def select_text():
    return FunctionTransformer(lambda X: X["text"].tolist(), validate=False)


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Maps one score to 3 ordered levels with two cut-offs learned on the training data.

    Chooses t1 <= t2 maximising training accuracy for: x < t1 -> 0, t1 <= x < t2 -> 1, x >= t2 -> 2.
    Exhaustive over all midpoints between observed values, in O(n) with prefix counts.
    """

    def __init__(self, column: str):
        self.column = column

    def fit(self, X, y):
        x = X[self.column].to_numpy(dtype=float)
        y = np.asarray(y)
        u = np.unique(x)
        cand = np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]])
        # c[k][j] = number of class-k texts with x < cand[j]
        c = [np.searchsorted(np.sort(x[y == k]), cand, side="left") for k in range(3)]
        a = c[0] - c[1]                      # gain of placing t1 at cand[j]
        b = c[1] - c[2]                      # gain of placing t2 at cand[j]
        # best t1 at or before each candidate t2 (running max of a, with its position)
        run_max, run_arg = np.empty_like(a), np.empty_like(a)
        m, arg = -np.inf, 0
        for j, v in enumerate(a):
            if v > m:
                m, arg = v, j
            run_max[j], run_arg[j] = m, arg
        j2 = int(np.argmax(run_max + b))
        self.t1_, self.t2_ = cand[run_arg[j2]], cand[j2]
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict(self, X):
        x = X[self.column].to_numpy(dtype=float)
        return np.where(x < self.t1_, 0, np.where(x < self.t2_, 1, 2))


def majority():
    return Pipeline([("sel", select(["n_words"])), ("clf", DummyClassifier(strategy="most_frequent"))])


def length_only():
    return ThresholdClassifier("n_words")


def flesch_kincaid():
    return ThresholdClassifier("fk_grade")


def feature_lr(cols):
    """Main system: standardised features -> multinomial logistic regression (C tuned)."""
    return Pipeline([("sel", select(cols)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000))])


def char_ngram_svm():
    return Pipeline([("sel", select_text()),
                     ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True,
                                               min_df=2, lowercase=False)),
                     ("clf", LinearSVC(random_state=0, max_iter=20000))])
