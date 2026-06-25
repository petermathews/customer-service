# Customer service ticket triage, built four ways

Same problem solved four different ways, then compared on cost, accuracy, and what each one can actually do.

The problem: a support ticket comes in, sometimes with an attachment. Work out what it's about, how urgent it is, pull any data off the attachment, and route it to the right team.

I built that as a rules script, a trained ML model, a single Claude call, and a LangGraph agent. The interesting part isn't any one of them. It's deciding which one fits a given step, and backing that decision with numbers instead of opinion.

The dataset is synthetic. Real ticket queues are proprietary, which is the reason a public demo like this has to make its own data.

## The four versions

| Version | Stack | What it covers |
| --- | --- | --- |
| A. Rules | Python, regex | Keyword matching and a regex receipt parser. The cheap starting point. |
| B. Classic ML | pandas, scikit-learn | TF-IDF with logistic regression, plus an MLP neural net for comparison. |
| Image model | scikit-learn | A small classifier on image pixels, for photo attachments. |
| C. GenAI | Claude API, structured output | One call classifies the ticket and extracts receipt fields, no training data. |
| D. Agent | LangGraph + Claude | A graph that routes differently depending on intent and what's attached. |

Every version returns the same result object, so they line up directly. `src/evaluate.py` runs them all and prints a table: accuracy, cost per 1,000 tickets at real Claude pricing, latency, and what each one can and can't handle.

## Map the process first

Before reaching for a model, write down the current process and who does what.

**People:** frontline agents read each ticket, guess the queue, open attachments by hand, and escalate upset customers on instinct.

**Process:** ticket arrives, someone reads it, classifies it, checks sentiment and priority, opens any receipt or photo to pull details, routes it, and acts.

**Technology:** a shared inbox and a ticketing tool. Routing is human judgment and nothing gets measured.

Then the real question: which of those steps should be a rule, which a trained model, which an LLM call, which an agent, and what does each one cost.

## Start in the notebook

[`notebooks/customer_service_triage.ipynb`](notebooks/customer_service_triage.ipynb) walks the whole thing end to end. It runs in Google Colab (open it from the GitHub tab inside Colab and paste the repo URL) or locally.

## Run it locally

```bash
pip install -r requirements.txt
python data/generate_dataset.py      # builds data/tickets.csv
python src/evaluate.py               # prints the comparison table
```

To run the Claude and agent versions against the real API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install langgraph langchain-anthropic
python src/evaluate.py --live        # adds measured GenAI accuracy and tokens
python src/approach_c_genai.py       # one live Claude triage call
python src/approach_d_agent.py       # one agent run
```

## Where it runs and what it costs

Accuracy is only half the call. The other half is where it runs and how the bill works. [`docs/deployment_and_cost.md`](docs/deployment_and_cost.md) maps each version to its AWS, GCP, and Azure service, shows that the same Claude code runs on Bedrock, Vertex, or the Anthropic API with only the client and model id changing ([`src/providers.py`](src/providers.py)), and [`src/tco.py`](src/tco.py) models the monthly cost at scale.

```bash
python src/tco.py
```

A provisioned ML endpoint is a fixed cost whether one ticket flows through it or ten million. Claude calls are billed by the token, so they cost nothing when idle and scale with volume. They cross over around 238,000 tickets a month: below that, calling Haiku is cheaper than renting an endpoint; above it, host the trained model. Worth working out before you pick an architecture, not after.

## Layout

```
data/generate_dataset.py        reproducible synthetic ticket dataset
src/schema.py                   shared result object, routing and priority rules
src/approach_a_rules.py         A. keyword and regex baseline
src/approach_b_ml.py            B. TF-IDF with LogReg and an MLP neural net
src/image_classifier.py         image classification
src/approach_c_genai.py         C. Claude structured output, Haiku/Sonnet/Opus
src/approach_d_agent.py         D. LangGraph branching agent
src/providers.py                same Claude code on Anthropic, Bedrock, or Vertex
src/evaluate.py                 runs everything and prints the comparison
src/tco.py                      cost model and the fixed vs variable crossover
docs/deployment_and_cost.md     cloud mapping and total cost of ownership
notebooks/                      the walkthrough
```

## Stack

Python, Jupyter and Colab, pandas, scikit-learn, LangGraph, and the Claude API (`anthropic`, `langchain-anthropic`).

Built by Peter Mathews.
