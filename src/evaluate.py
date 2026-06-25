"""Evaluate every approach and build the decision matrix.

The offline approaches (rules, classic ML) are measured directly on the
held-out set. The GenAI/agent costs are computed from real published pricing
and a representative token count, re-run with a live API key (`--live`) to fill
in measured accuracy and tokens for those rows too.

Run:  python src/evaluate.py          # offline rows + cost estimates
      python src/evaluate.py --live    # also calls the Claude API (needs key)
"""

from __future__ import annotations

import sys
import time
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

import approach_a_rules as rules
from approach_b_ml import MLTriager, load_data
from approach_c_genai import MODEL_TIERS

warnings.filterwarnings("ignore")

# Representative tokens per triage call (input includes system + ticket +
# schema; output is the small structured result). Override with measured values
# from a --live run.
EST_INPUT_TOKENS = 220
EST_OUTPUT_TOKENS = 30


def cost_per_1k(tier: str, in_tok: int = EST_INPUT_TOKENS, out_tok: int = EST_OUTPUT_TOKENS) -> float:
    p = MODEL_TIERS[tier]
    return (in_tok * p["in"] + out_tok * p["out"]) / 1e6 * 1000


def _intent_extract_ok(pred, row) -> bool:
    """For document tickets: did we recover the order id from the receipt?
    Rules use regex (works on this exact layout); the receipt itself carries
    the true order id."""
    return pred.extracted.order_id is not None


def evaluate_offline(test: pd.DataFrame, triage_fn) -> dict:
    t0 = time.perf_counter()
    preds = [triage_fn(r.text, r.attachment_type, r.attachment_content) for r in test.itertuples()]
    elapsed = time.perf_counter() - t0
    intent_acc = sum(p.intent == r.intent for p, r in zip(preds, test.itertuples())) / len(test)
    sent_acc = sum(p.sentiment == r.sentiment for p, r in zip(preds, test.itertuples())) / len(test)
    docs = [(p, r) for p, r in zip(preds, test.itertuples()) if r.attachment_type == "document"]
    extract_acc = (sum(_intent_extract_ok(p, r) for p, r in docs) / len(docs)) if docs else float("nan")
    return {
        "intent_acc": intent_acc,
        "sentiment_acc": sent_acc,
        "doc_extract_acc": extract_acc,
        "latency_ms": elapsed / len(test) * 1000,
    }


def build_matrix(live: bool = False) -> pd.DataFrame:
    df = load_data()
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df["intent"])

    rows = []

    # A, rules
    m = evaluate_offline(test, rules.triage)
    rows.append({"approach": "A. Rules / regex", **m, "cost_per_1k_$": 0.0,
                 "training_data": "none", "extraction": "regex (brittle)",
                 "images": "filename only", "explainability": "total"})

    # B, classic ML (logreg baseline + mlp neural net)
    for model, label in [("logreg", "B. ML, TF-IDF + LogReg"), ("mlp", "B. ML, neural net (MLP)")]:
        triager = MLTriager(intent_model=model, sentiment_model=model).fit(train)
        m = evaluate_offline(test, triager.triage)
        rows.append({"approach": label, **m, "cost_per_1k_$": 0.0,
                     "training_data": "needs labels", "extraction": "regex (brittle)",
                     "images": "needs labelled photos", "explainability": "medium"})

    # C/D, GenAI tiers (cost from pricing; accuracy measured only with --live)
    for tier in ("haiku", "sonnet", "opus"):
        row = {"approach": f"C. GenAI, Claude {tier}",
               "intent_acc": float("nan"), "sentiment_acc": float("nan"),
               "doc_extract_acc": float("nan"), "latency_ms": float("nan"),
               "cost_per_1k_$": round(cost_per_1k(tier), 2),
               "training_data": "none (zero-shot)", "extraction": "zero-shot, robust",
               "images": "vision-capable", "explainability": "low"}
        if live:
            row.update(_evaluate_genai(test, tier))
        rows.append(row)

    return pd.DataFrame(rows)


def _evaluate_genai(test: pd.DataFrame, tier: str) -> dict:
    """Live evaluation against the Claude API (sampled to keep cost down)."""
    from approach_c_genai import GenAITriager

    triager = GenAITriager(tier=tier)
    sample = test.sample(min(40, len(test)), random_state=0)
    in_tok = out_tok = 0
    correct_i = correct_s = 0
    t0 = time.perf_counter()
    for r in sample.itertuples():
        p = triager.triage(r.text, r.attachment_type, r.attachment_content)
        correct_i += p.intent == r.intent
        correct_s += p.sentiment == r.sentiment
        in_tok += triager.last_usage[0]
        out_tok += triager.last_usage[1]
    elapsed = time.perf_counter() - t0
    n = len(sample)
    return {
        "intent_acc": correct_i / n,
        "sentiment_acc": correct_s / n,
        "latency_ms": elapsed / n * 1000,
        "cost_per_1k_$": round(cost_per_1k(tier, in_tok // n, out_tok // n), 2),
    }


def main() -> None:
    live = "--live" in sys.argv
    matrix = build_matrix(live=live)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print(f"\nDecision matrix ({'live' if live else 'offline + cost estimates'}):\n")
    print(matrix.to_string(index=False))
    print("\nAt 1,000,000 tickets/month, GenAI cost ≈ cost_per_1k × 1000; "
          "rules/ML are ~free at inference (training is one-time).")


if __name__ == "__main__":
    main()
