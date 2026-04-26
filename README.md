# Device Intelligence Assistant

> Ask plain questions about your test equipment and get instant, reasoned answers.

---

## What does it do?

Engineers working with measurement devices spend a lot of time hunting for information: *Which oscilloscope do we have that supports embedded work? What's inside the Fluke multimeter? Which of our devices share the same ADC chip?*

This system lets you ask those questions in plain language, like talking to a colleague who knows every device in your lab, and get back a direct answer with a transparent reasoning trail.

```
You:  "Which devices have the component RF Mixer?"

System:  "The RF Mixer component is used in the Rohde & Schwarz FSW Signal Analyzer,
          which falls under the category of spectrum analyzers."
```

No search engines, no manual browsing, no spreadsheets. Just a question and an answer.

---

## Business value

Knowledge about test equipment is usually scattered, locked in spreadsheets, email threads, and the heads of experienced engineers. When someone leaves, that knowledge walks out the door. When someone new joins, they spend weeks getting up to speed. When a decision needs to be made quickly, the right information is hard to find.

This system changes that. It makes your device fleet knowledge **available to everyone, instantly**, regardless of their technical background.

### For engineering teams

- No more digging through PDFs or asking the "person who knows"
- Quickly identify which devices share a critical component, essential for impact analysis when a part is discontinued or recalled
- Onboard new team members faster: they can explore the device inventory on their own

### For procurement and supply chain

- Instantly see which devices depend on a specific component or manufacturer
- Assess supplier risk: *"How many of our devices rely on <Device X>?"*
- Support sourcing decisions with accurate, up-to-date inventory data.

### For quality and compliance

- Trace component usage across the entire device spectrum
- Identify affected devices when a component has a known issue or recall
- Document what's in each device without manual audits

### For customers and sales

- Customers interested in a product can ask plain questions and get accurate, detailed answers without waiting for a sales contact
- Questions like *"What components does this device use?"* or *"Is this device suitable for power analysis?"* are answered instantly, 24/7
- Reduces back-and-forth between customers and technical staff for standard product inquiries
- Builds confidence in a purchase decision by making product knowledge transparent and accessible

### For management

- Reduces dependency on individual experts, as knowledge is institutionalized rather than personalized
- Faster decisions: questions that used to take hours (or days waiting for the right person) are answered in seconds
- Low barrier to entry: anyone can ask a plain-language question, no training required

---

## How it works

```
  ┌────────────────────────────────────────────────────────────────┐
  │                       Streamlit UI                             │
  │  Streams reasoning steps and answer tokens in real time        │
  └──────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
  ┌───────────────────────────────────────────────────────────────┐
  │               LangGraph ReAct Agent  (Qwen 2.5 via vLLM)      │
  │                                                               │
  │  • Up to 6 reasoning steps before answering                   │
  │  • Conversation memory across turns (per session)             │
  │  • Answers strictly grounded in tool results                  │
  └──────────┬───────────────────┬───────────────────┬────────────┘
             │                   │                   │
             ▼                   ▼                   ▼
  ┌──────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
  │  Knowledge Graph │ │ Semantic Search  │ │   Hybrid Search    │
  │  (Neo4j)         │ │ (ChromaDB)       │ │ (KG + Vector)      │
  │                  │ │                  │ │                    │
  │  Precise queries:│ │ Intent queries:  │ │ Complex queries:   │
  │  components,     │ │ "best device for │ │ "what device for   │
  │  manufacturers,  │ │  RF work"        │ │  RF and what's     │
  │  categories      │ │                  │ │  inside it?"       │
  └──────────────────┘ └─────────────────┘ └─────────────────────┘
             │                   │                   │
             └───────────────────┴───────────────────┘
                                 │
                                 ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                         Your Answer                            │
  │           + the reasoning steps that led to it                 │
  └────────────────────────────────────────────────────────────────┘
```

The agent has three ways to look things up:

- **Knowledge Graph**: A structured map of every device and its parts in Neo4j. Great for precise questions like *"which devices share this chip?"*, *"what category does this component belong to?"*, or *"who manufactures the ADC?"*
- **Semantic Search**: ChromaDB with sentence-transformer embeddings understands the *intent* behind a question. Great for open-ended queries like *"what device should I use for RF work?"*
- **Hybrid Search** (Combines both): Finds semantically relevant devices and enriches each result with its full component data in a single call. Best for complex multi-part queries.

The agent picks the right approach automatically, and you can always see its reasoning steps in the UI.

---

## Example questions

**Find devices by component**
- Which devices have the component RF Mixer?

**Find components for a device**
- What components does the Fluke 87V Multimeter use?

**Find the right device for a task**
- Which device is best for measuring power efficiency?

**Learn about a component**
- Who manufactures the ADC (Analog-to-Digital Converter)?

**Follow-up questions (conversation memory)**
- Tell me about the Fluke 87V. *(then)* What category are its components?

---

## Getting started

**Prerequisites:** [Podman](https://podman.io/getting-started/installation) with [podman-compose](https://github.com/containers/podman-compose).

**1. Configure environment**

```bash
cp .env.template .env
```

Edit `.env` and set your HuggingFace token (`HF_TOKEN`). A free account is enough to get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**2. Start all services**

```bash
podman-compose up -d
```

This starts vLLM (LLM inference), Neo4j (knowledge graph), ChromaDB (vector store), and the Streamlit app. On the **first run**, the model (`Qwen/Qwen2.5-3B-Instruct`) is downloaded into a named volume which can take a few minutes. Subsequent starts are fast since the model is cached.

> **Note:** vLLM performs best with a GPU. CPU-only is supported but inference will be slower.

**3. Open the app**

Go to [http://localhost:8501](http://localhost:8501) in your browser and start asking questions.

**4. Stop the services**

```bash
podman-compose down
```

To also remove all stored data (Neo4j graph, ChromaDB vectors, model cache):

```bash
podman-compose down -v
```

---

*Built with [vLLM](https://docs.vllm.ai), [LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://python.langchain.com), [Neo4j](https://neo4j.com), [ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).*

---

> **Note:** This project is under active development. Features, supported question types, and the underlying architecture are subject to change.
