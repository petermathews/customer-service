# Deployment & cost — thinking in AWS / GCP / Azure terms

A model's accuracy is half the decision. The other half is **where it runs, how
it's billed, and what it costs to own** — and that's a cloud question. This is
the framework I'd have a Fellow team apply before committing to an architecture.

## 1 · Map each approach to a managed service

The same four approaches, mapped to the equivalent managed service on each major
cloud. Nobody hand-rolls this infrastructure anymore — you pick a tier and a
provider.

| Approach | AWS | Google Cloud | Azure | Cost model |
| --- | --- | --- | --- | --- |
| **A · Rules** | Lambda | Cloud Functions / Cloud Run | Azure Functions | Per-invocation (≈ free) |
| **B · Classic ML** | SageMaker (training + endpoints) | Vertex AI (Training + Endpoints) | Azure Machine Learning | Per instance-hour (provisioned) **or** per-request (serverless) |
| **B′ · Image model** | SageMaker / Rekognition | Vertex AI / Vision AI | Azure ML / AI Vision | Same as B, or per-image for the prebuilt vision API |
| **C · GenAI (Claude)** | **Amazon Bedrock** · Claude Platform on AWS | **Vertex AI Model Garden** | **Azure AI Foundry** | Per-token (in/out), Batch ≈ 50% off, prompt caching |
| **D · Agent** | Bedrock AgentCore / API + tool use | Vertex AI Agent Builder | Azure AI Agent Service | LLM tokens + orchestration compute |

> Claude itself is available on **Amazon Bedrock**, **Google Vertex AI**,
> **Microsoft Foundry**, and **Claude Platform on AWS**, in addition to the
> Anthropic API — so the GenAI/agent rows aren't locked to one vendor.

## 2 · Same Claude code, three platforms

Provider choice is usually a procurement / security / data-residency decision,
not an engineering one — because the code barely changes. Only the client
constructor and the model-id string differ (see [`src/providers.py`](../src/providers.py)):

```python
import anthropic
# Anthropic API
client = anthropic.Anthropic();        model = "claude-haiku-4-5"
# Amazon Bedrock           pip install "anthropic[bedrock]"
client = anthropic.AnthropicBedrock(); model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
# Google Vertex AI         pip install "anthropic[vertex]"
client = anthropic.AnthropicVertex();  model = "claude-haiku-4-5@<version>"

# everything downstream is identical:
client.messages.parse(model=model, max_tokens=512, messages=[...], output_format=TriageSchema)
```

**Caveat worth teaching:** feature parity isn't perfect. Managed Agents,
server-side tools, and the Batches API are **not** on Bedrock or Vertex — there
you build agents with "API + tool use." Knowing those edges is the difference
between a slide and a shippable design.

## 3 · The cost models cross over (the key lesson)

Run [`src/tco.py`](../src/tco.py). It models monthly cost three ways:

- **Fixed** — a provisioned ML endpoint costs the same whether 1 or 10M tickets
  flow through it (you rent the instance 24/7).
- **Variable** — per-token LLM calls cost \$0 when idle and scale linearly.
- **Near-free variable** — rules on serverless.

| | 10k/mo | 100k/mo | 1M/mo | 10M/mo |
| --- | ---: | ---: | ---: | ---: |
| Rules → serverless | \$0.00 | \$0.04 | \$0.40 | \$4 |
| ML → provisioned endpoint | \$88 | \$88 | \$88 | \$88 |
| ML → serverless inference | \$0.50 | \$5 | \$50 | \$500 |
| Claude Haiku (per-token) | \$3.70 | \$37 | \$370 | \$3,700 |
| Claude Sonnet (per-token) | \$11 | \$111 | \$1,110 | \$11,100 |
| Claude Opus (per-token) | \$18.50 | \$185 | \$1,850 | \$18,500 |

**Crossover:** a provisioned ML endpoint (\$88/mo fixed) beats per-token Claude
Haiku above **~238,000 tickets/month**. Below that, just call the LLM — you're
not paying for an idle box. Above it, host the trained model.

> Quality is a *separate* axis. Even where the LLM costs more, it may be the only
> option that reads a new receipt format or a photo zero-shot — so the real
> design routes the cheap, stable bulk to ML and the messy tail to the LLM.

## 4 · The TCO checklist (build-vs-buy)

Per-call price is the most visible cost and rarely the biggest. The framework:

- **Inference** — per-token or per-instance-hour (above).
- **Training / fine-tuning** — one-time compute for the ML path; \$0 for zero-shot LLM.
- **Data & labelling** — often the largest hidden cost of the ML path; \$0 for the LLM.
- **Engineering & maintenance** — retraining, drift monitoring, prompt upkeep.
- **Ops** — endpoints to keep up, autoscaling, on-call.
- **Risk / compliance** — data residency (regional endpoints carry a premium),
  logging, PII handling, vendor lock-in.

The cheapest *per-call* option is frequently the most expensive to *own*. Making
a team put a number on each row — not just the API price — is the coaching.
