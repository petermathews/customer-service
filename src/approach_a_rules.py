"""Approach A, Rules / heuristics (the 'before' state).

No ML. Keyword lexicons for intent, a sentiment word list, and regex to pull
fields out of a receipt. This is what most support teams actually start with,
and it's the honest baseline every other approach has to beat. Its failure
modes are the teaching point: it can't read free-text it wasn't told about, and
every new phrasing means a new keyword.
"""

from __future__ import annotations

import re

from schema import (
    ExtractedFields,
    TriageResult,
    decide_action,
    decide_priority,
)

# Ordered most-specific first, the first lexicon that matches wins.
INTENT_KEYWORDS = [
    ("account_access", ["locked out", "log in", "login", "password", "hacked", "two-factor", "2fa", "reset"]),
    ("damaged_product", ["damaged", "cracked", "broken", "dent", "scratched", "torn"]),
    ("refund_request", ["refund", "money back", "double charged"]),
    ("billing_question", ["bill", "charge", "invoice", "fee", "subscription price", "statement"]),
    ("order_status", ["where is my order", "tracking", "shipping status", "delivered", "shipped"]),
    ("technical_issue", ["crash", "error", "not work", "doesn't work", "spins", "sync", "buttons don't"]),
    ("complaint", ["unacceptable", "frustrated", "terrible", "worst", "disappointed", "fed up"]),
    ("general_inquiry", ["discount", "hours", "mobile app", "ship internationally", "change my plan"]),
]

NEGATIVE_WORDS = [
    "unacceptable", "frustrated", "terrible", "worst", "disappointed",
    "fed up", "broken", "damaged", "never", "hacked", "wrong", "too late",
]
POSITIVE_WORDS = ["thanks", "great", "love", "appreciate"]

RECEIPT_ORDER = re.compile(r"Order\s*#:\s*([A-Z0-9]+)", re.IGNORECASE)
RECEIPT_TOTAL = re.compile(r"Total:\s*\$([0-9]+\.[0-9]{2})")
RECEIPT_DATE = re.compile(r"Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")


def classify_intent(text: str) -> str:
    low = text.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(k in low for k in keywords):
            return intent
    return "general_inquiry"  # the catch-all when nothing matches


def classify_sentiment(text: str) -> str:
    low = text.lower()
    if any(w in low for w in NEGATIVE_WORDS):
        return "negative"
    if any(w in low for w in POSITIVE_WORDS):
        return "positive"
    return "neutral"


def extract_document(content: str) -> ExtractedFields:
    """Regex extraction, works only because we know the receipt's exact
    layout. Change the template and this silently returns nothing."""
    fields = ExtractedFields()
    if (m := RECEIPT_ORDER.search(content)):
        fields.order_id = m.group(1)
    if (m := RECEIPT_TOTAL.search(content)):
        fields.amount = float(m.group(1))
    if (m := RECEIPT_DATE.search(content)):
        fields.order_date = m.group(1)
    return fields


def classify_image(content: str) -> str:
    """The rules approach can't actually look at an image, it can only read
    the filename. That's the limitation the ML/agent approaches remove."""
    if "damaged" in content:
        return "damaged"
    if "intact" in content:
        return "intact"
    return "unknown"


def triage(text: str, attachment_type: str = "none", attachment_content: str = "") -> TriageResult:
    intent = classify_intent(text)
    sentiment = classify_sentiment(text)
    has_document = attachment_type == "document"
    has_image = attachment_type == "image"

    extracted = extract_document(attachment_content) if has_document else ExtractedFields()
    image_label = classify_image(attachment_content) if has_image else None

    from schema import INTENT_TO_ROUTE

    return TriageResult(
        intent=intent,
        sentiment=sentiment,
        priority=decide_priority(intent, sentiment),
        route=INTENT_TO_ROUTE[intent],
        action=decide_action(intent, has_document, has_image),
        extracted=extracted,
        image_label=image_label,
    )


if __name__ == "__main__":
    demo = triage(
        "My order A12345 came damaged, there's a dent on the side.",
        attachment_type="image",
        attachment_content="photo_damaged",
    )
    print(demo.to_dict())
