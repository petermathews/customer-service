# Deployment and cost

Accuracy is only part of the decision. Where a workload runs, how it is billed, and what it costs to own often determines the architecture more than the model does, and that is a cloud question.

## Each implementation maps to a managed service

None of this infrastructure is built by hand. Each implementation maps to an equivalent managed service on the major clouds.

| Implementation | AWS | Google Cloud | Azure | Billing model |
| --- | --- | --- | --- | --- |
| Rules | Lambda | Cloud Functions or Cloud Run | Azure Functions | Per invocation, effectively free |
| Classic ML | SageMaker (training and endpoints) | Vertex AI | Azure Machine Learning | Per instance hour if provisioned, or per request if serverless |
| Image model | SageMaker or Rekognition | Vertex AI or Vision AI | Azure ML or AI Vision | As above, or per image on the prebuilt vision API |
| GenAI (Claude) | Amazon Bedrock, or Claude Platform on AWS | Vertex AI Model Garden | Azure AI Foundry | Per token in and out; Batch is about half price; caching reduces repeat cost |
| Agent | Bedrock AgentCore, or the API plus tool use | Vertex AI Agent Builder | Azure AI Agent Service | Language model tokens plus orchestration compute |

Claude is available on Amazon Bedrock, Google Vertex AI, Microsoft Foundry, and the Anthropic API, so the GenAI and agent rows are not tied to one vendor.

## The same Claude code on three platforms

Provider choice is usually a procurement, security, or data residency decision rather than an engineering one, because the code barely changes. Only the client and the model identifier differ. See [`src/providers.py`](../src/providers.py).

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

Parity is not complete. Managed Agents, server side tools, and the Batches API are not available on Bedrock or Vertex; on those platforms, agents are built with the API plus tool use. Accounting for those differences is the distinction between a diagram and a deployable design.

## The cost models cross over

Run [`src/tco.py`](../src/tco.py). It models monthly cost three ways.

A trained model on a provisioned endpoint is a fixed cost. The instance runs continuously, so it costs the same whether one ticket or ten million flow through it. That is inexpensive per ticket at high volume and wasteful at low volume.

Claude calls are billed by the token. They cost nothing when idle and the bill scales linearly with volume. That is inexpensive at low volume and expensive at high volume.

Rules on serverless are the near zero floor.

| | 10k / month | 100k / month | 1M / month | 10M / month |
| --- | ---: | ---: | ---: | ---: |
| Rules on serverless | $0.00 | $0.04 | $0.40 | $4 |
| ML on a provisioned endpoint | $88 | $88 | $88 | $88 |
| ML on serverless inference | $0.50 | $5 | $50 | $500 |
| Claude Haiku | $3.70 | $37 | $370 | $3,700 |
| Claude Sonnet | $11 | $111 | $1,110 | $11,100 |
| Claude Opus | $18.50 | $185 | $1,850 | $18,500 |

The endpoint at $88 a month becomes cheaper than Haiku above roughly 238,000 tickets per month. Below that, calling the model is less expensive, since there is no idle instance to pay for. Above it, hosting the trained model is more economical.

Quality is a separate axis. Even where the model costs more, it may be the only option that reads a new receipt format or a photo without retraining, so the strongest design routes the stable, high volume bulk to ML and the messy remainder to the model.

## The cost of ownership checklist

The per call price is the most visible cost and rarely the largest. Before selecting an approach, put a figure on each of the following:

- Inference, either per token or per instance hour.
- Training or fine tuning, a one time compute cost for the ML path and none for the GenAI path.
- Data and labelling, often the largest hidden cost of the ML path and none for the GenAI path.
- Engineering and maintenance: retraining, drift monitoring, and prompt upkeep.
- Operations: endpoints to keep available, autoscaling, and on call coverage.
- Risk and compliance: data residency (regional endpoints carry a premium), logging, handling of personal data, and vendor lock in.

The option that is cheapest per call is frequently the most expensive to own. The useful analysis is a number for every row, against the actual service the workload would run on.
