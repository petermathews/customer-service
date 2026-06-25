# Customer Service Triage: Rules vs ML vs GenAI vs Agents

A practical comparison of four ways to automate one customer service workflow, measured on accuracy, cost, latency, and capability.

## What this is

A common failure mode in applied AI is reaching for an agent or a large language model before understanding the workflow. This project takes the opposite approach. It implements a single, familiar support workflow four ways and measures the tradeoffs rather than assuming them.

The workflow is ticket intake. An inbound support ticket arrives, sometimes with an attachment. The system classifies the request, assesses its urgency, extracts any structured data from the attachment, and routes the ticket to the appropriate team. The same workflow is implemented as:

1. Rules and regular expressions
2. Classic machine learning
3. A single Claude call with structured output
4. A LangGraph agent

Every implementation returns the same `TriageResult`, so the comparison is direct rather than theoretical. The recurring conclusion is that the right design is usually a hybrid: rules and classic ML for the stable, high volume majority, and GenAI or an agent for the messy, branching minority.

## What this demonstrates

- Why rules remain the right starting point for stable, well defined logic.
- When classic ML outperforms rules for intent classification, given labelled data.
- Where GenAI justifies its cost: messy text, unseen document formats, and field extraction without training data.
- When an agent is warranted: workflows that branch into different actions or tools.
- How cost behaves at different ticket volumes, mapped to real cloud pricing.

## Decision rule: when to use each

| Situation | Recommended starting point |
| --- | --- |
| The logic is stable and easy to specify | Rules |
| Labelled historical tickets are available | Classic ML |
| The input is messy or document formats vary | GenAI |
| The workflow branches into different actions or tools | Agent |
| Low cost at high volume is the priority | Rules or classic ML |
| Flexibility at low volume is the priority | GenAI |

## The four implementations

| Implementation | Stack | Demonstrates |
| --- | --- | --- |
| A. Rules | Python, regex | Keyword classification and regular expression field extraction |
| B. Classic ML | pandas, scikit-learn | TF-IDF with logistic regression, and an MLP neural network |
| Image model | scikit-learn | Classification on raw image pixels for photo attachments |
| C. GenAI | Claude API, structured output | Classification and field extraction in one call, no training data |
| D. Agent | LangGraph, Claude | Conditional routing based on intent and attachment type |

`src/evaluate.py` runs all of them and produces a single comparison table covering accuracy, cost per thousand tickets at current Claude pricing, latency, and capability.

## The workflow

Effective automation starts with the existing process, not the model. The table below describes the current state of a typical support operation.

| Layer | Current state |
| --- | --- |
| People | Frontline agents read each ticket, infer the correct queue, open attachments manually, and escalate dissatisfied customers at their discretion. |
| Process | A ticket is received, read, classified by request type and urgency, enriched with any attachment data, then routed and actioned. |
| Technology | A shared inbox and a ticketing system. Routing depends on individual judgment and is not measured. |

The governing question is which of those steps warrant a rule, a trained model, a language model call, or an agent, and what each option costs to operate.

## About the data

The dataset is synthetic by design. Real customer service queues contain private customer data, proprietary categories, and attachments that cannot be published. The value here is the architecture and the evaluation, not the specific tickets:

```
ticket in -> classify -> assess sentiment and priority -> inspect attachment -> extract fields -> route -> compare cost and accuracy
```

A real team can substitute its own ticket history and keep the same evaluation structure. `data/generate_dataset.py` builds the dataset deterministically.

## The notebook

[`notebooks/customer_service_triage.ipynb`](notebooks/customer_service_triage.ipynb) presents the full analysis end to end. It runs in Google Colab (open it from the GitHub tab within Colab and paste the repository URL) or locally.

## Running locally

```bash
pip install -r requirements.txt
python data/generate_dataset.py      # builds data/tickets.csv
python src/evaluate.py               # prints the comparison table
```

To run the Claude and agent implementations against the live API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install langgraph langchain-anthropic
python src/evaluate.py --live        # adds measured GenAI accuracy and token counts
python src/approach_c_genai.py       # a single live Claude triage call
python src/approach_d_agent.py       # a single agent run
```

## What the comparison measures

For each implementation, the table reports intent accuracy, sentiment accuracy, document extraction success, latency, cost per thousand tickets, training data requirements, extraction robustness, image handling, and explainability. The offline rows (rules and classic ML) are measured on every run. The GenAI rows show estimated cost by default; accuracy and latency are populated when the notebook or `evaluate.py` is rerun with `ANTHROPIC_API_KEY` and `--live`.

## Cost and deployment

Accuracy is only part of the decision. Where a workload runs and how it is billed frequently matters more, and that is a cloud question. [`docs/deployment_and_cost.md`](docs/deployment_and_cost.md) maps each implementation to its equivalent service on AWS, GCP, and Azure, shows that the same Claude code runs on Amazon Bedrock, Google Vertex AI, or the Anthropic API with only the client and model identifier changing ([`src/providers.py`](src/providers.py)), and [`src/tco.py`](src/tco.py) models monthly cost at scale.

A provisioned machine learning endpoint carries a fixed monthly cost regardless of volume. Claude calls are billed by the token, so they cost nothing when idle and scale linearly with volume. The two cross at approximately 238,000 tickets per month. Below that threshold, calling Claude Haiku is less expensive than operating a dedicated endpoint; above it, hosting the trained model is more economical.

## Walkthrough

1. The problem: support tickets must be classified, prioritised, enriched from attachments, and routed.
2. The structure: the same workflow is implemented as rules, classic ML, GenAI, and an agent, each returning the same `TriageResult`.
3. The evaluation: `src/evaluate.py` measures accuracy, latency, cost, and capability across all four.
4. The cost model: `src/tco.py` adds the fixed versus variable cost crossover and the cloud service mapping.
5. The conclusion: the strongest design is usually hybrid, with rules and classic ML handling the stable bulk and GenAI or an agent handling the messy, branching cases.

## Repository layout

```
data/generate_dataset.py        reproducible synthetic ticket dataset
src/schema.py                   shared result object, routing and priority rules
src/approach_a_rules.py         A. keyword and regex baseline
src/approach_b_ml.py            B. TF-IDF with logistic regression and an MLP neural network
src/image_classifier.py         image classification
src/approach_c_genai.py         C. Claude structured output, Haiku, Sonnet, and Opus tiers
src/approach_d_agent.py         D. LangGraph branching agent
src/providers.py                the same Claude code on Anthropic, Bedrock, or Vertex
src/evaluate.py                 runs every implementation and prints the comparison
src/tco.py                      cost model and the fixed versus variable crossover
docs/deployment_and_cost.md     cloud service mapping and total cost of ownership
notebooks/                      the full walkthrough
```

## Stack

Python, Jupyter and Google Colab, pandas, scikit-learn, LangGraph, and the Claude API via the `anthropic` and `langchain-anthropic` libraries.

Author: Peter Mathews
