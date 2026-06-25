# Smart Customer-Service Intake — one process, four ways

A single business process — **triage an inbound customer-service ticket and
route it** — implemented four ways, from a keyword script to a Claude-powered
agent, and compared on accuracy, cost, and capability.

It's built the way I coach: take a real process, map the people / process /
technology around it, implement it at increasing levels of sophistication, and
make the team **defend with numbers** which one actually fits. Knowing *when* a
rule beats ML beats GenAI is the judgment a coach has to teach — so this repo is
organized around that decision, not around any one model.

> Everything here is original teaching material. The dataset is synthetic on
> purpose: real ticket queues are proprietary, which is exactly why a public
> demo like this exists.

## The same intake, four ways

| Approach | Stack | What it demonstrates |
| --- | --- | --- |
| **A · Rules** | Python, regex | The honest, free baseline every model must beat |
| **B · Classic ML** | pandas, scikit-learn (TF-IDF + LogReg **and** a neural net) | Learned intent/sentiment, real held-out metrics |
| **B′ · Image classification** | scikit-learn (MLP on pixels) | Reading a photo attachment — the 4th ML concept |
| **C · GenAI** | Claude API, `messages.parse()` structured output | Zero-shot classification **and** receipt extraction in one call |
| **D · Agent** | LangGraph + Claude | Conditional routing on intent *and* attachment type |

Each approach returns the **same** `TriageResult`, so they're directly
comparable. The payoff is a **decision matrix** scoring all of them on accuracy,
cost per 1k tickets (from real Claude pricing), latency, and capability — plus a
**total-cost-of-ownership model** that maps every approach to its AWS / GCP /
Azure managed service and finds the cost crossover.

## The business process (People · Process · Technology)

| Layer | Current state of a typical support org |
| --- | --- |
| **People** | Tier-1 agents read each ticket, guess the queue, manually open attachments, escalate angry customers by feel. |
| **Process** | ticket → read → classify intent → check sentiment/priority → if a receipt/photo is attached, open it and extract details → route → act. |
| **Technology** | A shared inbox + ticketing tool. Routing is human judgment; nothing is measured. |

The transformation question: **which steps should become a rule, a trained
model, an LLM call, or an agent — and what does each cost?**

## Start here

Open **[`notebooks/customer_service_triage.ipynb`](notebooks/customer_service_triage.ipynb)** —
it walks the whole thing end to end with a *Coach's Note* at each step. It runs
in **Google Colab** (`File → Open notebook → GitHub`, paste this repo's URL) or
locally.

## Run it locally

```bash
pip install -r requirements.txt
python data/generate_dataset.py      # builds data/tickets.csv (deterministic)
python src/evaluate.py               # prints the decision matrix (offline + cost estimates)
```

To run the GenAI and agent paths against the real API:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install langgraph langchain-anthropic
python src/evaluate.py --live        # fills in measured GenAI accuracy + tokens
python src/approach_c_genai.py       # one live Claude triage call
python src/approach_d_agent.py       # one LangGraph agent run
```

## Layout

```
data/generate_dataset.py        # reproducible synthetic ticket dataset (+ tickets.csv)
src/schema.py                   # shared TriageResult + routing/priority policy
src/approach_a_rules.py         # A — keyword + regex baseline
src/approach_b_ml.py            # B — TF-IDF + LogReg / MLP neural net
src/image_classifier.py         # B′ — image-classification mini-lesson
src/approach_c_genai.py         # C — Claude structured output (tiered: Haiku/Sonnet/Opus)
src/approach_d_agent.py         # D — LangGraph branching agent
src/providers.py                # same Claude code on Anthropic / Bedrock / Vertex
src/evaluate.py                 # runs everything → the decision matrix
src/tco.py                      # cost model + fixed-vs-variable crossover
docs/deployment_and_cost.md     # AWS/GCP/Azure mapping + TCO framework
notebooks/                      # the narrated walkthrough
```

## Where it runs and what it costs

The comparison doesn't stop at accuracy. [`docs/deployment_and_cost.md`](docs/deployment_and_cost.md)
maps each approach to its **AWS / GCP / Azure** managed service, shows that the
same Claude code runs on Bedrock, Vertex, or the Anthropic API (only the client
+ model-id change — [`src/providers.py`](src/providers.py)), and
[`src/tco.py`](src/tco.py) models monthly cost at scale:

```
python src/tco.py
# → a provisioned ML endpoint ($88/mo fixed) beats per-token Claude Haiku
#   above ~238,000 tickets/month. Below that, just call the LLM.
```

That fixed-vs-variable crossover — not "use the biggest model" — is the kind of
call a coach should make a team compute before they pick an architecture.

## What the comparison shows

No single approach wins — and saying so *is* the lesson. Rules are free but
brittle; classic ML is near-free at inference and strong on stable labels but
needs data and can't read new formats; GenAI handles the long tail and
extraction zero-shot but costs real money per call; the agent is the right tool
when the *process itself* branches. The design I'd actually ship blends them —
which is the judgment this repo exists to teach.

## Stack

Python · Jupyter / Google Colab · pandas · scikit-learn · LangGraph ·
Claude API (`anthropic`, `langchain-anthropic`) — the same tools the Fellows
work in.

---

Built by Peter Mathews as a hands-on teaching artifact for the AI Coach role.
