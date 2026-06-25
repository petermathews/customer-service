"""Provider portability — the same Claude code on Anthropic, AWS, or GCP.

A coach should be fluent in *where* a model runs, not just which model. Claude
is reachable three ways that share the exact same Messages API — only the client
constructor and the model-id string change:

    Anthropic API   anthropic.Anthropic()              model="claude-haiku-4-5"
    Amazon Bedrock  anthropic.AnthropicBedrock()       model="global.anthropic.claude-haiku-4-5-20251001-v1:0"
    Google Vertex   anthropic.AnthropicVertex()        model="claude-haiku-4-5@<version>"

Everything downstream — `messages.parse()`, the Pydantic schema, the response
handling in approach_c — is identical. That portability is the point: you pick
the platform on procurement, security, and billing grounds, not by rewriting
code.

Install extras as needed:
    pip install "anthropic[bedrock]"     # AWS
    pip install "anthropic[vertex]"      # GCP

NOTE: exact Bedrock/Vertex model strings are region- and version-specific and
change over time. The maps below are illustrative — confirm the current id in
the AWS Bedrock / GCP Model Garden console for your region before relying on it.
"""

from __future__ import annotations

# Anthropic-direct model ids (authoritative — these are what approach_c uses).
ANTHROPIC_IDS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

# Amazon Bedrock ids use an `anthropic.` provider prefix plus a routing prefix
# (`global.` for cross-region, `us.`/`eu.` for data-residency). Illustrative.
BEDROCK_IDS = {
    "haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "global.anthropic.claude-sonnet-4-6",
    "opus": "global.anthropic.claude-opus-4-8",  # confirm current id in console
}

# Google Vertex ids use an `@version` suffix. Illustrative — confirm in Model
# Garden for your region.
VERTEX_IDS = {
    "haiku": "claude-haiku-4-5@latest",
    "sonnet": "claude-sonnet-4-6@latest",
    "opus": "claude-opus-4-8@latest",
}

PROVIDERS = {
    "anthropic": ANTHROPIC_IDS,
    "bedrock": BEDROCK_IDS,
    "vertex": VERTEX_IDS,
}


def make_client(provider: str = "anthropic"):
    """Construct the right SDK client for the provider. Same `client.messages.*`
    surface in every case."""
    import anthropic

    if provider == "anthropic":
        return anthropic.Anthropic()           # reads ANTHROPIC_API_KEY
    if provider == "bedrock":
        return anthropic.AnthropicBedrock()    # reads AWS creds / AWS_REGION
    if provider == "vertex":
        return anthropic.AnthropicVertex()     # reads GOOGLE_* / gcloud creds
    raise ValueError(f"unknown provider {provider!r}; choose one of {list(PROVIDERS)}")


def resolve_model(provider: str, tier: str) -> str:
    """Map a tier (haiku/sonnet/opus) to the provider's model-id string."""
    try:
        return PROVIDERS[provider][tier]
    except KeyError as exc:
        raise ValueError(f"no model id for provider={provider!r} tier={tier!r}") from exc
