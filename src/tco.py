"""Total cost of ownership — and the fixed-vs-variable crossover.

"What does it cost" has no answer without "where does it run." Each approach
maps to a different cloud cost *model*, and the models cross over:

  * A trained model on a **provisioned endpoint** (SageMaker / Vertex Endpoint /
    Azure ML online endpoint) is a **fixed** monthly cost — you pay for the
    instance whether 1 or 10M tickets flow through it. Cheap per ticket at high
    volume, wasteful at low volume.
  * A **per-token LLM** (Claude on Bedrock / Vertex / Anthropic) is **variable**
    — $0 when idle, scales linearly with volume. Cheap at low volume, expensive
    at high volume.
  * **Rules on serverless** (Lambda / Cloud Functions / Azure Functions) are the
    near-free variable floor.

The crossover volume — where the fixed endpoint becomes cheaper than calling the
LLM — is the single most useful number in this whole analysis, and the kind of
thing a coach should make a team compute before they pick an architecture.

All prices are **illustrative on-demand list prices** for shaping the decision —
confirm current pricing for your region/provider. Run:  python src/tco.py
"""

from __future__ import annotations

import pandas as pd

# --- cost assumptions (edit these; they're list-price estimates) -------------
# Per-1M-token prices mirror approach_c.MODEL_TIERS.
PRICE_PER_MTOK = {
    "haiku": {"in": 1.00, "out": 5.00},
    "sonnet": {"in": 3.00, "out": 15.00},
    "opus": {"in": 5.00, "out": 25.00},
}
EST_INPUT_TOKENS = 220
EST_OUTPUT_TOKENS = 30

# A small always-on real-time inference endpoint, ~$0.12/hr × 730 hr.
ML_ENDPOINT_MONTHLY = 88.0
# Serverless per-ticket costs (compute only; model is tiny).
RULES_SERVERLESS_PER_TICKET = 0.0000004   # ~$0.40 / 1M tickets
ML_SERVERLESS_PER_TICKET = 0.00005        # ~$50 / 1M tickets (cold-start + compute)


def llm_cost_per_ticket(tier: str, batch: bool = False) -> float:
    p = PRICE_PER_MTOK[tier]
    cost = (EST_INPUT_TOKENS * p["in"] + EST_OUTPUT_TOKENS * p["out"]) / 1e6
    return cost * 0.5 if batch else cost   # Batch APIs are ~50% off


def monthly_cost(approach: str, volume: int) -> float:
    """Modelled monthly cost for a given approach at `volume` tickets/month."""
    if approach == "rules_serverless":
        return volume * RULES_SERVERLESS_PER_TICKET
    if approach == "ml_endpoint":
        return ML_ENDPOINT_MONTHLY                      # fixed, volume-independent
    if approach == "ml_serverless":
        return volume * ML_SERVERLESS_PER_TICKET
    if approach.startswith("genai_"):
        tier = approach.split("_", 1)[1]
        return volume * llm_cost_per_ticket(tier)
    raise ValueError(f"unknown approach {approach!r}")


def crossover_volume(fixed_monthly: float, variable_per_ticket: float) -> float:
    """Volume at which the fixed-cost option overtakes the variable one."""
    return fixed_monthly / variable_per_ticket


def build_tco_table(volumes=(10_000, 100_000, 1_000_000, 10_000_000)) -> pd.DataFrame:
    approaches = [
        ("Rules → serverless (Lambda/Functions)", "rules_serverless"),
        ("Classic ML → provisioned endpoint", "ml_endpoint"),
        ("Classic ML → serverless inference", "ml_serverless"),
        ("Claude Haiku (per-token)", "genai_haiku"),
        ("Claude Sonnet (per-token)", "genai_sonnet"),
        ("Claude Opus (per-token)", "genai_opus"),
    ]
    rows = []
    for label, key in approaches:
        row = {"approach": label}
        for v in volumes:
            row[f"${v:,}/mo"] = round(monthly_cost(key, v), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\nModelled monthly cost (USD) by volume:\n")
    print(build_tco_table().to_string(index=False))

    xo = crossover_volume(ML_ENDPOINT_MONTHLY, llm_cost_per_ticket("haiku"))
    print(
        f"\nCrossover: a provisioned ML endpoint (${ML_ENDPOINT_MONTHLY:.0f}/mo fixed) "
        f"beats per-token Claude Haiku above ~{xo:,.0f} tickets/month."
    )
    print("Below that, just call the LLM — you're not paying for an idle endpoint.")
    print("Above it, host the trained model. (Quality is a separate axis — the")
    print("LLM may be the only option that handles new formats and attachments.)")


if __name__ == "__main__":
    main()
