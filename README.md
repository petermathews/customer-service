# Customer Service Triage: Rules vs ML vs GenAI vs Agents

I built this to show how I think through AI implementation.

A lot of AI projects jump straight to the newest thing: an LLM, an agent, a framework, a demo that looks good for five minutes. But in real work, the harder question is usually simpler:

**What is the job, and what is the smallest reliable way to automate each part of it?**

So I took one familiar workflow — customer service ticket triage — and built it four ways:

1. Rules and regex
2. Classic machine learning
3. A single Claude call with structured output
4. A LangGraph agent

Each version handles the same task: read an incoming ticket, figure out what it is about, judge how urgent it is, pull data from an attachment when there is one, and route it to the right team.

The point is not that one approach wins every time. The point is to make the tradeoffs visible.

Sometimes rules are enough. Sometimes a small ML model is better. Sometimes an LLM is worth paying for because the input is messy. And sometimes an agent makes sense because the workflow branches.

## Why this project

I wanted something practical enough to run, but simple enough to explain.

Customer service is a good example because most companies understand the problem immediately. Tickets come in. People read them. Someone guesses the queue. Someone opens attachments. Someone decides whether the customer is upset enough to escalate.

That process can be automated in different ways. The mistake is picking the tool before mapping the process.

This repo starts with the workflow, then compares the options.

## The workflow

A ticket comes in.

The system needs to:

- classify the request
- detect sentiment and priority
- inspect an attachment if there is one
- extract fields like order number, amount, and date
- route the ticket
- suggest the next action

Every approach returns the same `TriageResult`, so the comparison is apples to apples instead of a bunch of disconnected demos.

## The four versions

| Version | Stack | What it shows |
| --- | --- | --- |
| A. Rules | Python, regex | Fast, cheap, easy to explain, but brittle |
| B. Classic ML | pandas, scikit-learn | Better intent classification when labelled data exists |
| Image model | scikit-learn | A simple stand-in for classifying photo attachments |
| C. GenAI | Claude API, structured output | Good for messy text and document extraction without training data |
| D. Agent | LangGraph + Claude | Useful when the path changes based on the ticket and attachment |

`src/evaluate.py` runs the approaches and prints a comparison table.

The table looks at:

- intent accuracy
- sentiment accuracy
- document extraction
- latency
- cost per 1,000 tickets
- training data needs
- image handling
- explainability

## The basic decision rule

This is the main takeaway:

| Situation | Start here |
| --- | --- |
| The logic is stable and obvious | Rules |
| You have labelled historical tickets | Classic ML |
| The text or documents are messy | GenAI |
| The workflow branches into different paths or tools | Agent |
| You need very low cost at high volume | Rules or ML |
| You need flexibility before you have much data | GenAI |

In practice, I would usually design this as a hybrid. Use rules and ML for the predictable bulk. Use GenAI or an agent for the messy edge cases.

## About the data

The data is synthetic on purpose.

Real support queues have private customer information, company-specific categories, and attachments that should not be published. So this repo generates its own ticket set and keeps the focus on the architecture and evaluation.

The pattern is what matters:

```text
ticket in -> classify -> assess priority -> inspect attachment -> extract fields -> route -> compare options
```

A real company could swap in its own ticket history and keep the same structure.

## Start with the notebook

The full walkthrough is here:

[`notebooks/customer_service_triage.ipynb`](notebooks/customer_service_triage.ipynb)

It runs in Google Colab or locally. The notebook is the easiest way to see the thinking end to end.

## Run it locally

```bash
pip install -r requirements.txt
python data/generate_dataset.py
python src/evaluate.py
```

To run the Claude and agent versions live:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install langgraph langchain-anthropic
python src/evaluate.py --live
python src/approach_c_genai.py
python src/approach_d_agent.py
```

Without an API key, the repo still runs the rules, ML, image classifier, and cost estimates.

## Cost and deployment

Accuracy is only part of the decision.

Cost changes a lot depending on volume. A hosted ML endpoint has a fixed monthly cost. Claude calls are pay-per-token. That means Claude can be cheaper at low volume, while a trained model can be cheaper once volume is high enough.

The cost write-up is here:

[`docs/deployment_and_cost.md`](docs/deployment_and_cost.md)

It maps the same approaches to AWS, GCP, and Azure, and shows where the cost curves cross.

## Five-minute walkthrough

If I were walking through this quickly, I would cover it like this:

1. Here is the support workflow.
2. Here is why I start with the process, not the model.
3. Here are the four implementations.
4. Here is the shared `TriageResult` object.
5. Here is the comparison table.
6. Here is the decision: rules/ML for the stable bulk, GenAI/agents for the messy or branching cases.

## Repository layout

```text
data/generate_dataset.py        builds the synthetic ticket dataset
src/schema.py                   shared result object, routing, and priority rules
src/approach_a_rules.py         rules and regex baseline
src/approach_b_ml.py            TF-IDF, logistic regression, and MLP classifier
src/image_classifier.py         image classification stand-in
src/approach_c_genai.py         Claude structured-output version
src/approach_d_agent.py         LangGraph branching agent
src/providers.py                same Claude pattern on Anthropic, Bedrock, or Vertex
src/evaluate.py                 runs the comparison
src/tco.py                      cost model
docs/deployment_and_cost.md     cloud mapping and cost notes
notebooks/                      walkthrough notebook
```

## Stack

Python, Jupyter, Google Colab, pandas, scikit-learn, LangGraph, and Claude.

Built by Peter Mathews.
