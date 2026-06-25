"""Approach C, GenAI with the Claude API (structured output).

One model call does the whole intake: classify intent + sentiment, and, when a
receipt is attached, extract the order fields in the *same* call, zero-shot, no
labelled training data. This is where the LLM earns its cost: it reads
free-text it was never trained on and pulls structured data out of a document a
regex would choke on.

Two things a coach would point out to a team:
  * `messages.parse()` with a Pydantic schema *guarantees* valid structured
    output, no brittle JSON parsing.
  * The model tier is a dial. Haiku is cheap and fast and fine for triage;
    Sonnet/Opus cost more but reason better. `MODEL_TIERS` makes that explicit
    so you can run the same code across tiers and compare (see evaluate.py).

Needs an API key:  export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field

from schema import (
    INTENT_TO_ROUTE,
    INTENTS,
    SENTIMENTS,
    ExtractedFields,
    TriageResult,
    decide_action,
    decide_priority,
)

# Pricing as published (USD per 1M tokens), used by the cost comparison.
# Haiku is the default for high-volume triage; the others show the tradeoff.
MODEL_TIERS = {
    "haiku": {"id": "claude-haiku-4-5", "in": 1.00, "out": 5.00},
    "sonnet": {"id": "claude-sonnet-4-6", "in": 3.00, "out": 15.00},
    "opus": {"id": "claude-opus-4-8", "in": 5.00, "out": 25.00},
}
DEFAULT_TIER = "haiku"


class TriageSchema(BaseModel):
    """The structure we force the model to return. Field descriptions are part
    of the prompt, the model reads them."""

    intent: str = Field(description=f"One of: {', '.join(INTENTS)}")
    sentiment: str = Field(description=f"One of: {', '.join(SENTIMENTS)}")
    order_id: Optional[str] = Field(default=None, description="Order number if a receipt/document is attached, else null")
    amount: Optional[float] = Field(default=None, description="Total amount from the receipt if present, else null")
    order_date: Optional[str] = Field(default=None, description="Order date (YYYY-MM-DD) from the receipt if present, else null")


SYSTEM = (
    "You are a customer-service intake assistant. Classify the ticket and, if a "
    "document is attached, extract its order fields. Return only the structured "
    "result. Use exactly the allowed values for intent and sentiment."
)


def _build_user_content(text: str, attachment_type: str, attachment_content: str) -> str:
    parts = [f"TICKET:\n{text}"]
    if attachment_type == "document":
        parts.append(f"\nATTACHED DOCUMENT:\n{attachment_content}")
    elif attachment_type == "image":
        parts.append("\n(An image is attached; it is handled by the image classifier, not here.)")
    return "\n".join(parts)


class GenAITriager:
    def __init__(self, tier: str = DEFAULT_TIER, provider: str = "anthropic", client=None):
        if tier not in MODEL_TIERS:
            raise ValueError(f"tier must be one of {list(MODEL_TIERS)}")
        self.tier = tier
        self.provider = provider
        # The same triage code runs on Anthropic, Bedrock, or Vertex, only the
        # client and the model-id string change. See providers.py.
        from providers import make_client, resolve_model

        self.model = resolve_model(provider, tier)
        self.client = client if client is not None else make_client(provider)
        self.last_usage = None  # (input_tokens, output_tokens) of the last call

    def triage(self, text: str, attachment_type: str = "none", attachment_content: str = "",
               image_label: Optional[str] = None) -> TriageResult:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=512,
            system=SYSTEM,
            messages=[{"role": "user", "content": _build_user_content(text, attachment_type, attachment_content)}],
            output_format=TriageSchema,
        )
        parsed: TriageSchema = response.parsed_output
        self.last_usage = (response.usage.input_tokens, response.usage.output_tokens)

        intent = parsed.intent if parsed.intent in INTENTS else "general_inquiry"
        sentiment = parsed.sentiment if parsed.sentiment in SENTIMENTS else "neutral"
        has_document = attachment_type == "document"
        has_image = attachment_type == "image"

        return TriageResult(
            intent=intent,
            sentiment=sentiment,
            priority=decide_priority(intent, sentiment),
            route=INTENT_TO_ROUTE[intent],
            action=decide_action(intent, has_document, has_image),
            extracted=ExtractedFields(
                order_id=parsed.order_id, amount=parsed.amount, order_date=parsed.order_date
            ) if has_document else ExtractedFields(),
            image_label=image_label if has_image else None,
        )


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the GenAI approach.")
    t = GenAITriager(tier="haiku")
    receipt = "ORDER CONFIRMATION\nOrder #: A55512\nDate: 2026-03-14\nItem: Wireless Headphones\nTotal: $129.99\nThank you!"
    result = t.triage("I want my money back, the headphones broke.", "document", receipt)
    print(result.to_dict())
    print("tokens:", t.last_usage)
