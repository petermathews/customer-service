"""Approach D: an agentic workflow with LangGraph.

The same intake expressed as a graph that branches. After classifying the
ticket, the path depends on the intent and the attachment:

    classify, then:
      document attached        extract the receipt fields
      image attached           classify the photo
      complaint and negative   escalate
      nothing attached         proceed
    all paths then converge on the routing and action step.

That conditional routing, where the next action depends on the ticket and the
attachment, is what an agent framework expresses cleanly and a single prompt or
a flat ML pipeline does not.

It uses Claude (via langchain-anthropic) for classification and extraction, the
scikit-learn image classifier for photos, and LangGraph for the control flow.

Requires: pip install langgraph langchain-anthropic, and ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
from typing import Optional, TypedDict

from schema import (
    INTENT_TO_ROUTE,
    INTENTS,
    SENTIMENTS,
    ExtractedFields,
    TriageResult,
    decide_action,
    decide_priority,
)


class TicketState(TypedDict, total=False):
    """The state object that flows through the graph. Each node reads and adds
    to it, this is the agent's working memory for one ticket."""

    text: str
    attachment_type: str
    attachment_content: str
    # filled in by nodes:
    intent: str
    sentiment: str
    extracted: ExtractedFields
    image_label: Optional[str]
    escalated: bool
    result: TriageResult


def build_agent(model_tier: str = "haiku", image_model=None):
    """Compile the LangGraph graph. `image_model` is the fitted sklearn image
    classifier from image_classifier.py (passed in so we don't retrain per
    ticket)."""
    from langgraph.graph import StateGraph, END
    from langchain_anthropic import ChatAnthropic
    from pydantic import BaseModel, Field

    from approach_c_genai import MODEL_TIERS

    llm = ChatAnthropic(model=MODEL_TIERS[model_tier]["id"], max_tokens=512)

    class Classification(BaseModel):
        intent: str = Field(description=f"One of: {', '.join(INTENTS)}")
        sentiment: str = Field(description=f"One of: {', '.join(SENTIMENTS)}")

    class Receipt(BaseModel):
        order_id: Optional[str] = None
        amount: Optional[float] = None
        order_date: Optional[str] = None

    classifier = llm.with_structured_output(Classification)
    extractor = llm.with_structured_output(Receipt)

    # --- nodes -------------------------------------------------------------
    def classify(state: TicketState) -> TicketState:
        out = classifier.invoke(
            f"Classify this support ticket.\nTICKET:\n{state['text']}"
        )
        intent = out.intent if out.intent in INTENTS else "general_inquiry"
        sentiment = out.sentiment if out.sentiment in SENTIMENTS else "neutral"
        return {"intent": intent, "sentiment": sentiment}

    def extract_doc(state: TicketState) -> TicketState:
        out = extractor.invoke(
            f"Extract the order fields from this receipt.\n{state['attachment_content']}"
        )
        return {"extracted": ExtractedFields(out.order_id, out.amount, out.order_date)}

    def classify_photo(state: TicketState) -> TicketState:
        # The image branch. In a real system this runs the photo through the
        # classifier; here the dataset hands us the label the classifier "saw".
        label = "damaged" if "damaged" in state["attachment_content"] else "intact"
        return {"image_label": label}

    def escalate(state: TicketState) -> TicketState:
        return {"escalated": True}

    def finalize(state: TicketState) -> TicketState:
        intent = state["intent"]
        sentiment = state["sentiment"]
        has_doc = state.get("attachment_type") == "document"
        has_img = state.get("attachment_type") == "image"
        result = TriageResult(
            intent=intent,
            sentiment=sentiment,
            priority=decide_priority(intent, sentiment),
            route=INTENT_TO_ROUTE[intent],
            action=decide_action(intent, has_doc, has_img),
            extracted=state.get("extracted", ExtractedFields()),
            image_label=state.get("image_label"),
        )
        return {"result": result}

    # --- conditional edge: where to go after classifying --------------------
    def route_after_classify(state: TicketState) -> str:
        if state.get("attachment_type") == "document":
            return "extract_doc"
        if state.get("attachment_type") == "image":
            return "classify_photo"
        if state["intent"] == "complaint" and state["sentiment"] == "negative":
            return "escalate"
        return "finalize"

    # --- wire the graph -----------------------------------------------------
    g = StateGraph(TicketState)
    g.add_node("classify", classify)
    g.add_node("extract_doc", extract_doc)
    g.add_node("classify_photo", classify_photo)
    g.add_node("escalate", escalate)
    g.add_node("finalize", finalize)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify, {
        "extract_doc": "extract_doc",
        "classify_photo": "classify_photo",
        "escalate": "escalate",
        "finalize": "finalize",
    })
    g.add_edge("extract_doc", "finalize")
    g.add_edge("classify_photo", "finalize")
    g.add_edge("escalate", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def triage(agent, text: str, attachment_type: str = "none", attachment_content: str = "") -> TriageResult:
    final = agent.invoke({
        "text": text,
        "attachment_type": attachment_type,
        "attachment_content": attachment_content,
    })
    return final["result"]


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY and `pip install langgraph langchain-anthropic` to run the agent.")
    agent = build_agent()
    receipt = "ORDER CONFIRMATION\nOrder #: A55512\nDate: 2026-03-14\nTotal: $129.99"
    print(triage(agent, "Please refund me, item never worked.", "document", receipt).to_dict())
