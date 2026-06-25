# Customer Service Triage: Rules vs ML vs GenAI vs Agents

This project compares four approaches to automating the same customer service workflow: rules, classic machine learning, GenAI, and an agent-based workflow.

The goal is to show practical AI implementation judgment. Rather than starting with a model or framework, the project starts with the business process: an incoming support ticket needs to be classified, prioritized, enriched from any attachment, and routed to the right team.

Each implementation solves the same problem and returns the same `TriageResult`, which makes the approaches directly comparable across accuracy, latency, cost, training-data requirements, extraction capability, image handling, and explainability.

## Why I built this

I built this as a portfolio project to demonstrate how I approach applied AI work: define the workflow first, establish a simple baseline, then evaluate whether ML, GenAI, or an agent actually improves the solution.

This is a common decision point for product and engineering teams. Many use cases do not require an agent. Some are better served by deterministic rules. Others benefit from a trained model. GenAI becomes valuable when the input is less structured, such as free-form customer messages or variable document formats. An agent becomes useful when the workflow needs to branch across multiple steps or tools.

The project is intentionally small enough to review in a notebook, but structured like a real evaluation: same task, same output schema, multiple implementation paths, and a cost/quality comparison.

## Workflow being modeled

The example workflow is customer service ticket triage.

For each ticket, the system needs to:

- identify the customer intent
- assess sentiment and priority
- inspect an attachment when present
- extract structured fields such as order number, amount, and date
- route the ticket to the correct support team
- recommend the next action

This creates a useful test case because it includes both predictable logic and messy inputs.

## Implementations

| Version | Stack | Purpose |
| --- | --- | --- |
| A. Rules | Python, regex | Establishes a fast, explainable baseline using keyword matching and receipt parsing |
| B. Classic ML | pandas, scikit-learn | Trains intent and sentiment classifiers using TF-IDF, logistic regression, and an MLP classifier |
| Image model | scikit-learn | Demonstrates the image-classification path for photo attachments |
| C. GenAI | Claude API, structured output | Uses one model call to classify the ticket and extract fields from unstructured text |
| D. Agent | LangGraph + Claude | Routes the workflow through different steps based on intent and attachment type |

## Evaluation approach

`src/evaluate.py` runs the approaches and produces a comparison table.

The evaluation includes:

- intent accuracy
- sentiment accuracy
- document extraction success
- latency
- estimated cost per 1,000 tickets
- training-data requirements
- extraction robustness
- image-handling capability
- explainability

The offline rows, including rules and classic ML, run without an API key. The GenAI and agent paths can be run live by setting `ANTHROPIC_API_KEY`.

## Key takeaway

The strongest architecture is usually not one approach everywhere. It is a hybrid design.

| Situation | Practical starting point |
| --- | --- |
| Logic is stable and easy to define | Rules |
| Labeled historical data is available | Classic ML |
| Inputs are messy or document formats vary | GenAI |
| The workflow branches into different steps or tools | Agent |
| High-volume, low-cost processing is required | Rules or ML |
| Flexibility is needed before much data exists | GenAI |

In this example, rules and ML are strong candidates for the predictable, high-volume parts of the workflow. GenAI and agents are better suited for the less predictable cases, such as variable attachments, extraction from unfamiliar formats, or multi-step routing decisions.

## Data

The dataset is synthetic by design. Real customer service data often contains private customer information, proprietary routing categories, and attachments that should not be published.

The value of the project is the evaluation pattern:

```text
ticket in -> classify -> assess priority -> inspect attachment -> extract fields -> route -> compare approaches
```

A real organization could replace the synthetic data with historical ticket data while keeping the same evaluation structure.

## Notebook

The full walkthrough is available here:

[`notebooks/customer_service_triage.ipynb`](notebooks/customer_service_triage.ipynb)

The notebook walks through the workflow, data, rules baseline, ML model, GenAI approach, agent workflow, and final comparison.

## Run locally

```bash
pip install -r requirements.txt
python data/generate_dataset.py
python src/evaluate.py
```

To run the Claude and agent implementations live:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install langgraph langchain-anthropic
python src/evaluate.py --live
python src/approach_c_genai.py
python src/approach_d_agent.py
```

## Cost and deployment

The project also includes a cost and deployment comparison.

A provisioned ML endpoint has a fixed monthly cost. GenAI calls are typically billed per token. That creates a tradeoff: GenAI can be cost-effective at lower volumes or for variable inputs, while a hosted model may become more economical at higher volumes.

The deployment and cost notes are here:

[`docs/deployment_and_cost.md`](docs/deployment_and_cost.md)

That document maps each approach to comparable services across AWS, GCP, and Azure.

## Repository layout

```text
data/generate_dataset.py        reproducible synthetic ticket dataset
src/schema.py                   shared result object, routing, and priority rules
src/approach_a_rules.py         rules and regex baseline
src/approach_b_ml.py            TF-IDF, logistic regression, and MLP classifier
src/image_classifier.py         image classification example
src/approach_c_genai.py         Claude structured-output implementation
src/approach_d_agent.py         LangGraph branching workflow
src/providers.py                Claude provider examples for Anthropic, Bedrock, and Vertex
src/evaluate.py                 comparison runner
src/tco.py                      cost model
docs/deployment_and_cost.md     cloud mapping and cost discussion
notebooks/                      project walkthrough
```

## Stack

Python, Jupyter, Google Colab, pandas, scikit-learn, LangGraph, and Claude.

Built by Peter Mathews.
