"""Shared data structures for the triage pipeline.

Every approach in this repo — rules, classic ML, GenAI, and the LangGraph
agent — produces the *same* `TriageResult` for an incoming ticket. That's the
whole point: one business process, four implementations, directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# The set of intents a ticket can have. In a real deployment these map to the
# queues your support org already runs.
INTENTS = [
    "billing_question",
    "refund_request",
    "technical_issue",
    "damaged_product",
    "order_status",
    "account_access",
    "complaint",
    "general_inquiry",
]

# Each intent routes to a team. Routing is part of what we're comparing, so it
# lives here as the single source of truth.
INTENT_TO_ROUTE = {
    "billing_question": "Billing",
    "refund_request": "Refunds",
    "technical_issue": "Tech Support",
    "damaged_product": "Returns & Quality",
    "order_status": "Fulfillment",
    "account_access": "Account Security",
    "complaint": "Escalations",
    "general_inquiry": "General Support",
}

SENTIMENTS = ["positive", "neutral", "negative"]
PRIORITIES = ["low", "normal", "high", "urgent"]


@dataclass
class ExtractedFields:
    """Structured fields pulled from an attachment (a receipt, an order
    confirmation). Empty when the ticket has no document attachment."""

    order_id: Optional[str] = None
    amount: Optional[float] = None
    order_date: Optional[str] = None


@dataclass
class TriageResult:
    """The decision the intake system makes for one ticket."""

    intent: str
    sentiment: str
    priority: str
    route: str
    # The action a human (or downstream automation) should take next.
    action: str
    # Populated only when a document attachment was processed.
    extracted: ExtractedFields = field(default_factory=ExtractedFields)
    # For image attachments: what the image classifier saw.
    image_label: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def decide_priority(intent: str, sentiment: str) -> str:
    """Priority policy shared by every approach so the comparison is apples to
    apples. Angry customers and account-security issues jump the queue."""
    if intent == "account_access":
        return "urgent"
    if sentiment == "negative" and intent in ("complaint", "damaged_product", "refund_request"):
        return "high"
    if intent == "complaint":
        return "high"
    if sentiment == "negative":
        return "normal"
    return "normal" if intent != "general_inquiry" else "low"


def decide_action(intent: str, has_document: bool, has_image: bool) -> str:
    """The 'varying actions' branch: what to do depends on intent and whether
    something is attached."""
    if intent == "refund_request" and has_document:
        return "Verify order from receipt, then issue refund"
    if intent == "refund_request":
        return "Request receipt, then route to Refunds"
    if intent == "damaged_product" and has_image:
        return "Inspect photo, open return, send replacement"
    if intent == "damaged_product":
        return "Request a photo of the damage"
    if intent == "account_access":
        return "Trigger identity verification flow"
    if intent == "complaint":
        return "Escalate to a manager for personal follow-up"
    if intent == "order_status":
        return "Look up tracking and reply with status"
    if intent == "technical_issue":
        return "Run troubleshooting playbook"
    if intent == "billing_question":
        return "Answer from billing FAQ or pull the invoice"
    return "Answer the general inquiry"
