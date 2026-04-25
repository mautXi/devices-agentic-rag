# Device Intelligence Assistant

> Ask plain questions about your test equipment and get instant, reasoned answers.

---

## What does it do?

Engineers working with measurement devices spend a lot of time hunting for information: *Which oscilloscope do we have that supports embedded work? What's inside the Fluke multimeter? Which of our devices share the same ADC chip?*

This system lets you ask those questions in plain language, like talking to a colleague who knows every device in your lab, and get back a direct answer with a transparent reasoning trail.

```
You:    "Which devices use the component High-Voltage Isolation Amplifier?"

System: The Keysight DSOX1204G Oscilloscope and Rigol DG1062Z Function Generator are two devices that use the High-Voltage Isolation Amplifier component.
```

No search engines, no manual browsing, no spreadsheets. Just a question and an answer.

---

## Business value

Knowledge about test equipment is usually scattered, locked in datasheets, spreadsheets, email threads, and the heads of experienced engineers. When someone leaves, that knowledge walks out the door. When someone new joins, they spend weeks getting up to speed. When a decision needs to be made quickly, the right information is hard to find.

This system changes that. It makes your device fleet knowledge **available to everyone, instantly**, regardless of their technical background.

### For engineering teams

- No more digging through PDFs or asking the "person who knows"
- Quickly identify which devices share a critical component, essential for impact analysis when a part is discontinued or recalled
- Onboard new team members faster: they can explore the device inventory on their own

### For procurement and supply chain

- Instantly see which devices depend on a specific component or manufacturer
- Assess supplier risk: *"How many of our devices rely on Xilinx FPGAs?"*
- Support sourcing decisions with accurate, up-to-date inventory data without needing to request a report from engineering

### For quality and compliance

- Trace component usage across the entire device fleet in seconds
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
                        ┌─────────────────────────────────────────┐
                        │           Your Question                 │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │              AI Agent                   │
                        │  Thinks through the question, decides   │
                        │  what to look up, and forms an answer   │
                        └────────────┬────────────────┬───────────┘
                                     │                │
                    ┌────────────────▼──┐       ┌─────▼──────────────────┐
                    │  Component Graph  │       │    Semantic Search     │
                    │                   │       │                        │
                    │  Knows exactly    │       │  Understands meaning   │
                    │  which device     │       │  and finds the right   │
                    │  uses which part  │       │  device for a task     │
                    └───────────────────┘       └────────────────────────┘
                                     │                │
                                     └────────┬───────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────────┐
                        │              Your Answer                │
                        │  + the reasoning steps that led to it   │
                        └─────────────────────────────────────────┘
```

The agent has two ways to look things up:

- **Component Graph**: A structured map of every device and its parts. Great for precise questions like *"which devices share this chip?"* or *"what's inside device X?"*
- **Semantic Search**: Understands the *intent* behind a question. Great for open-ended questions like *"what device should I use for RF work?"*

The agent picks the right approach automatically, and you can always see its reasoning steps in the UI.

---

## Example questions

**Find devices by component**
- Which devices have the component RF Mixer?
- Which devices use an FPGA, and who makes it?

**Find components for a device**
- What components does the Fluke 87V Multimeter use?

**Find the right device for a task**
- What device should I use for RF signal analysis?
- Which device is best for measuring power efficiency?

**Learn about a component**
- Who manufactures the TCXO in our devices?

---

## Getting started

**Prerequisites:** [Podman](https://podman.io/getting-started/installation) with [podman-compose](https://github.com/containers/podman-compose).

**1. Start all services**

```bash
podman-compose up -d
```

This starts the LLM server, the databases, and the web interface. On the **first run**, the model is downloaded into a named volume which takes some time. Subsequent starts are fast since the model is cached.

> **Note:** vLLM performs best with a GPU. CPU-only is supported but inference will be slower.

**2. Open the app**

Go to [http://localhost:8501](http://localhost:8501) in your browser and start asking questions.

**3. Stop the services**

```bash
podman-compose down
```

---

*Built with [vLLM](https://docs.vllm.ai), [Neo4j](https://neo4j.com), [ChromaDB](https://www.trychroma.com), and [Streamlit](https://streamlit.io).*

---

> **Note:** This project is under active development. Features, supported question types, and the underlying architecture are subject to change. Current capabilities reflect an early working version. Broader device coverage, additional query types, and architectural improvements are planned.
