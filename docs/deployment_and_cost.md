# Deployment and cost: thinking in AWS, GCP, and Azure terms

Accuracy is half the decision. The other half is where the thing runs, how it gets billed, and what it costs to own. That is a cloud question, and it usually decides the architecture more than the model does.

## Map each version to a managed service

Nobody hand rolls this infrastructure anymore. You pick a tier and a provider. Here is the same set of versions mapped to the equivalent managed service on each major cloud.

| Version | AWS | Google Cloud | Azure | How it bills |
| --- | --- | --- | --- | --- |
| Rules | Lambda | Cloud Functions or Cloud Run | Azure Functions | Per invocation, basically free |
| Classic ML | SageMaker | Vertex AI | Azure Machine Learning | Per instance hour if provisioned, or per request if serverless |
| Image model | SageMaker or Rekognition | Vertex AI or Vision AI | Azure ML or AI Vision | Same as ML, or per image on the prebuilt vision API |
| GenAI (Claude) | Amazon Bedrock, or Claude Platform on AWS | Vertex AI Model Garden | Azure AI Foundry | Per token in and out, Batch is about half price, caching helps |
| Agent | Bedrock AgentCore, or the API plus tools | Vertex AI Agent Builder | Azure AI Agent Service | LLM tokens plus orchestration compute |

Claude itself runs on Amazon Bedrock, Google Vertex AI, Microsoft Foundry, and the Anthropic API, so the GenAI and agent rows are not locked to one vendor.

## Same Claude code, three platforms

Which provider you use is usually a procurement, security, or data residency call, not an engineering one, because the code barely changes. Only the client and the model id string differ. See [`src/providers.py`](../src/providers.py).

```python
import anthropic
# Anthropic API
client = anthropic.Anthropic();        model = "claude-haiku-4-5"
# Amazon Bedrock           pip install "anthropic[bedrock]"
client = anthropic.AnthropicBedrock(); model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
# Google Vertex AI         pip install "anthropic[vertex]"
client = anthropic.AnthropicVertex();  model = "claude-haiku-4-5@<version>"

# everything after this is identical:
client.messages.parse(model=model, max_tokens=512, messages=[...], output_format=TriageSchema)
```

One thing worth knowing: parity is not perfect. Managed Agents, server side tools, and the Batches API are not on Bedrock or Vertex. There you build agents with the API plus tool use. Knowing those edges is the difference between a slide and something you can actually ship.

## The cost models cross over

This is the part most people skip. Run [`src/tco.py`](../src/tco.py). It models monthly cost three ways.

A trained model on a provisioned endpoint is a fixed cost. You rent the instance around the clock, so it costs the same whether one ticket or ten million flow through it. Cheap per ticket at high volume, wasteful at low volume.

Claude calls are billed by the token. Nothing when idle, and the bill scales straight up with volume. Cheap at low volume, expensive at high volume.

Rules on serverless are the near zero floor.

| | 10k a month | 100k a month | 1M a month | 10M a month |
| --- | ---: | ---: | ---: | ---: |
| Rules on serverless | $0.00 | $0.04 | $0.40 | $4 |
| ML on a provisioned endpoint | $88 | $88 | $88 | $88 |
| ML on serverless inference | $0.50 | $5 | $50 | $500 |
| Claude Haiku | $3.70 | $37 | $370 | $3,700 |
| Claude Sonnet | $11 | $111 | $1,110 | $11,100 |
| Claude Opus | $18.50 | $185 | $1,850 | $18,500 |

The endpoint at $88 a month beats Haiku once you pass roughly 238,000 tickets a month. Below that, just call the LLM, because you are not paying for an idle box. Above it, host the trained model.

Quality is a separate axis. Even where the LLM costs more, it may be the only option that reads a new receipt format or a photo without retraining, so the real design sends the cheap stable bulk to ML and the messy tail to the LLM.

## The cost of ownership checklist

The per call price is the most visible cost and rarely the biggest. Before picking, put a number on each of these:

- Inference, either per token or per instance hour.
- Training or fine tuning, a one time compute cost for the ML path and zero for the zero shot LLM.
- Data and labelling, often the largest hidden cost of the ML path and zero for the LLM.
- Engineering and maintenance, meaning retraining, drift monitoring, prompt upkeep.
- Ops, meaning endpoints to keep alive, autoscaling, on call.
- Risk and compliance, meaning data residency (regional endpoints cost a premium), logging, PII, and vendor lock in.

The cheapest option per call is often the most expensive to own. Making a team put a number on every row, and name the actual service they would deploy on, is the work.
