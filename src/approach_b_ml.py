"""Approach B: traditional machine learning (scikit-learn).

Learns intent and sentiment from labelled data rather than hand written rules.
Two models are included:
  * Logistic regression over TF-IDF features, the strong and inexpensive
    baseline.
  * An MLP neural network, the same pipeline with a deeper model, for direct
    comparison with the linear baseline.

It still uses the regular expression receipt parser for document fields, which
is a fair point in the comparison: classic ML classifies text well but does not
extract fields from unseen documents or read images without substantially more
labelled data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from schema import INTENT_TO_ROUTE, ExtractedFields, TriageResult, decide_action, decide_priority
from approach_a_rules import extract_document, classify_image

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tickets.csv")


def _make_pipeline(model: str) -> Pipeline:
    """TF-IDF features into a classifier. Swapping the final estimator is the
    only difference between the baseline and the neural network, which shows how
    scikit-learn pipelines compose."""
    if model == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif model == "mlp":
        # A small feed forward neural network. One hidden layer keeps it fast
        # and is plenty for this dataset.
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=800, random_state=42)
    else:
        raise ValueError(f"unknown model {model!r}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", clf),
    ])


@dataclass
class MLTriager:
    """Trains an intent model and a sentiment model, then triages tickets."""

    intent_model: str = "logreg"   # "logreg" or "mlp"
    sentiment_model: str = "logreg"

    def __post_init__(self):
        self._intent_pipe = _make_pipeline(self.intent_model)
        self._sentiment_pipe = _make_pipeline(self.sentiment_model)
        self._trained = False

    def fit(self, df: pd.DataFrame) -> "MLTriager":
        self._intent_pipe.fit(df["text"], df["intent"])
        self._sentiment_pipe.fit(df["text"], df["sentiment"])
        self._trained = True
        return self

    def triage(self, text: str, attachment_type: str = "none", attachment_content: str = "") -> TriageResult:
        if not self._trained:
            raise RuntimeError("call .fit() before .triage()")
        intent = self._intent_pipe.predict([text])[0]
        sentiment = self._sentiment_pipe.predict([text])[0]
        has_document = attachment_type == "document"
        has_image = attachment_type == "image"
        return TriageResult(
            intent=intent,
            sentiment=sentiment,
            priority=decide_priority(intent, sentiment),
            route=INTENT_TO_ROUTE[intent],
            action=decide_action(intent, has_document, has_image),
            extracted=extract_document(attachment_content) if has_document else ExtractedFields(),
            image_label=classify_image(attachment_content) if has_image else None,
        )


def load_data(path: str = DATA) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = load_data()
    train, test = train_test_split(df, test_size=0.25, random_state=42, stratify=df["intent"])
    for model in ("logreg", "mlp"):
        t = MLTriager(intent_model=model, sentiment_model=model).fit(train)
        correct = sum(t.triage(r.text).intent == r.intent for r in test.itertuples())
        print(f"{model:7s} intent accuracy on held out set: {correct / len(test):.1%}")
