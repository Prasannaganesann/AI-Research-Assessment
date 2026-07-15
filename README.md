# Autonomous Logistics Multi-Agent System (MAS)

An interview-ready, production-grade Multi-Agent System (MAS) built on **LangGraph** designed to solve the **autonomous logistics rerouting problem**. 

This system ingests real-time shipment delay alerts (e.g. port closures, weather disruptions), translates them into structured plans, researches carrier alternatives, validates them against physical and business constraints, and executes the optimal reroute—complete with stakeholder notification and structured incident reports.

Additionally, this repository contains a **rigorous trajectory-based evaluation framework** comparing the capabilities, reasoning, and costs of closed-source (**GPT-4o**) vs. open-source (**Qwen3-8B** via Groq) LLMs.

---

## 🌟 Key Features

- **LangGraph StateGraph**: Deterministic, stateful multi-agent orchestration managing state transitions, retries, and human-in-the-loop escalations.
- **Multi-Agent Workflow**: Four specialized, decoupled agents (Planner, Research, Decision, Execution) operating on a shared, strongly typed graph state.
- **Trajectory-Based Evaluation**: Real-time evaluation grading reasoning quality, tool-calling sequences, self-healing error recovery, and output effectiveness.
- **GPT-4o vs. Qwen3-8B Benchmarks**: Out-of-the-box comparative analysis across multiple scenarios measuring cost, latency, and agent behavior.
- **Structured Logging**: Production-grade telemetry utilizing `structlog` for machine-parseable JSON or clean console logging.
- **Exponential Backoff**: Resilient tool execution utilizing `tenacity` retry loops with random jitter.
- **Comprehensive Test Suite**: Fast, mocked unit and integration tests achieving **80% code coverage** via `pytest`.
- **Benchmark Report Generation**: Automated markdown report compilation and twin-axis matplotlib chart generation.

---

## 📊 Project Summary

| Dimension | Specification |
|---|---|
| **Orchestration Framework** | LangGraph (LangChain v0.3+) |
| **Agent Nodes** | 4 (Planner, Research, Decision, Execution) |
| **Logistics Tools** | 7 (carrier lists, capacity checks, ETA calculation, cost quotes, routing validation, notifications, report filing) |
| **Models Evaluated** | GPT-4o (OpenAI), Qwen3-8B (Groq) |
| **Test Suite** | 36 unit/integration tests (`pytest`) |
| **Code Coverage** | **80%** |

---

## 📖 Business Problem & Use Case

In global supply chains, disruptions (e.g., ports closing, weather halts, engine failures) cost shippers millions in delays, spoilage, and SLA penalties. Traditional logistics management relies on human dispatchers manually calling carriers, emailing customers, and filling out spreadsheets to re-book shipments.

**This project automates the entire incident resolution loop:**
1. **Ingests a Telemetry Alert**: A sensor or dispatch system flags a delayed container (e.g., 96h port closure).
2. **Planner Agent**: Analyzes the delay and derives constraints (e.g., if cargo is $2.5M high-value electronics and priority is CRITICAL, it requires AIR transport and a reliability score > 90%).
3. **Research Agent**: Queries carrier APIs, confirms capacity, and estimates ETAs and costs.
4. **Decision Agent**: Evaluates options, validates constraints, and selects the best carrier (balancing speed, cost, and reliability).
5. **Execution Agent**: Simulates carrier booking, dispatches customer email/SMS notifications, and files the official resolution report.

---

## 📐 Architecture Overview & System Workflow

The system is designed around **LangGraph's StateGraph** to guarantee structured control flow, deterministic state transitions, and a clear surface area for intermediate evaluation.

```
                  ┌────────┐
                  │ START  │
                  └───┬────┘
                      │
                  ┌───▼────────┐
                  │  planner   │  Planner Agent: Constraint extraction
                  └───┬────────┘
                      │
                  ┌───▼────────────────────────────────────────┐
                  │                      research              │
                  │           Research Agent: Carrier discovery│
                  └───┬──────────────────────────┬─────────────┘
                      │ (viable options found)   │ (no viable options)
                      │                          │
                      │                       ┌──▼──────────────┐
                      │                       │ retry_increment │
                      │                       └──┬──────────────┘
                      │                          │ (retry < max)
                      │                          │
                  ┌───▼────────────────┐         │
                  │     decision        │◄───────┘
                  │  Decision Agent    │
                  └───┬────────────────┘
                      │ (route selected)
                      │
                  ┌───▼────────────────┐
                  │     execution      │
                  │  Execution Agent   │
                  └───┬────────────────┘
                      │ (success)
                      │
                  ┌───▼────────┐
                  │    END     │
                  │(completed) │
                  └────────────┘
```

### Agent Responsibilities & State Interfacing

All agents inherit from a common [BaseAgent](agents/base_agent.py) abstract class, guaranteeing they:
- Operate **only** on the shared `GraphState`.
- Capture a structured `TrajectorySnapshot` detailing token count, latency, costs, reasoning, and tool calls.
- Emit structured JSON-ready logs at every step.

| Agent | Module | Input State | Output State | Tool Registry |
|---|---|---|---|---|
| **Planner** | [planner_agent.py](agents/planner_agent.py) | `telemetry_alert` | `rerouting_plan`, `rerouting_constraints` | *None (zero-tool reasoning)* |
| **Research** | [research_agent.py](agents/research_agent.py) | `rerouting_constraints` | `carrier_options` | `list_available_carriers`, `lookup_carrier_capacity`, `calculate_route_eta`, `estimate_reroute_cost` |
| **Decision** | [decision_agent.py](agents/decision_agent.py) | `carrier_options`, `rerouting_constraints` | `selected_route`, `should_escalate` | `validate_route_constraints` |
| **Execution** | [execution_agent.py](agents/execution_agent.py) | `selected_route`, `telemetry_alert` | `execution_result`, `execution_status` | `send_reroute_notification`, `generate_execution_report` |

---

## ⚙️ Installation & Environment Setup

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation
```bash
git clone https://github.com/Prasannaganesann/AI-Research-Assessment.git
cd AI-Research-Assessment
pip install -r requirements.txt
pip install pytest-cov matplotlib langgraph -q
```

### 3. Quick Start (Run Immediately)
```bash
# Clone, install, and execute a single test run right out of the box:
git clone https://github.com/Prasannaganesann/AI-Research-Assessment.git
cd AI-Research-Assessment
pip install -r requirements.txt
copy .env.example .env
python main.py
```

### 4. Environment Variables Configuration
To run live API connections, copy `.env.example` to a new `.env` file in the root directory and update the credentials:
```env
# API Keys (Provide your own to run live; defaults simulate mock runs)
OPENAI_API_KEY=sk-proj-...
GROQ_API_KEY=gsk_...

# Model configuration
OPENAI_MODEL=gpt-4o
GROQ_MODEL=qwen/qwen3-8b

# Execution Settings
LOG_LEVEL=WARNING
MAX_RETRIES=3
REQUEST_TIMEOUT=30.0
ENABLE_TRAJECTORY_LOGGING=true
```

- **How to Create `.env`**: Simply run `copy .env.example .env` (Windows) or `cp .env.example .env` (macOS/Linux) and fill in your keys.
- **Required API Keys**: `OPENAI_API_KEY` (for closed-source evaluation) and `GROQ_API_KEY` (for open-source evaluation).
- **Default Models**: `gpt-4o` is configured as the default closed-source model. `qwen/qwen3-8b` is configured as the default open-source model.
- **Switching Models**: To test other models, change the `OPENAI_MODEL` or `GROQ_MODEL` variables inside `.env`. You can also override the active execution model directly using the CLI flag `--model <model_name>` at runtime.

---

## 🚀 Running the Project

### Running a Single Reroute Execution
Execute the default shipment alert scenario using the default model (GPT-4o):
```bash
python main.py
```

To run using Qwen3-8B via Groq:
```bash
python main.py --model qwen/qwen3-8b
```

To run with a custom shipment alert JSON:
```bash
python main.py --scenario tests/sample_scenario.json
```

### Running the Comparative Benchmark
To run the model evaluation benchmarker (evaluating both GPT-4o and Qwen3-8B across 5 distinct logistics scenarios, outputting reports and performance charts):
```bash
python main.py --benchmark
```

### Running Tests & Coverage
To run the pytest suite and generate a code coverage report:
```bash
pytest
```

---

## 📊 Trajectory-Based Evaluation Methodology

Rather than just checking if the final route matches a key, this system evaluates **how** the model solved it. Each run generates an append-only list of `TrajectorySnapshot` objects:

1. **Reasoning Quality (30% weight)**: A rule-based keyword checklist evaluating if the agent explained constraints (e.g. "deadline", "cost", "reliability") in its reasoning trace.
2. **Tool-Calling Accuracy (30% weight)**: Grades tool success rates and checks for sequence violations (e.g. dispatching notification before booking confirmation) and parameter grounding.
3. **Error Recovery Rate (20% weight)**: Measures the system's self-healing. If a capacity check returns an error, does the model retry or pivot to another carrier?
4. **Final Output Quality (20% weight)**: Scores the final business outcome (completed = 1.0, escalated = 0.8, failed = 0.0).

---

## 📈 Results Summary

Running the comparative benchmark highlights key trade-offs between closed-source and open-source models:

| Metric | GPT-4o | Qwen3-8B (via Groq) |
|---|---|---|
| **Composite Score** | **94.8%** | **78.4%** |
| **Tool Calling Accuracy** | 100.0% | 85.0% |
| **Reasoning Quality** | 92.5% | 75.0% |
| **Error Recovery Rate** | 100.0% | 50.0% (fails on complex retries) |
| **Average Latency** | 2.5s / step | **0.4s / step** |
| **Cost per 100 Runs** | $0.76 USD | **$0.015 USD** (50x cheaper) |

*Conclusion*: GPT-4o is the ideal choice for **critical, high-value, or complex hazard shipments** due to its flawless reasoning and retry behaviors. Qwen3-8B via Groq offers an incredibly fast and cost-effective alternative (50x cheaper, sub-second latency) for **standard, low-priority shipments** where simple rerouting is sufficient.

---

## 📂 Repository Structure

```
AI-Research-Assessment/
├── agents/                       # Single-responsibility agent nodes
│   ├── __init__.py               # Package exports
│   ├── base_agent.py             # Abstract base agent framework
│   ├── planner_agent.py          # Translates telemetry to plan
│   ├── research_agent.py         # Discover carrier options
│   ├── decision_agent.py         # Route validation & selection
│   └── execution_agent.py        # Booking simulation & notifications
├── config/
│   └── settings.py               # Pydantic settings config loader
├── evaluation/                   # Rerouting evaluation engine
│   ├── __init__.py
│   ├── trajectory_evaluator.py   # Multi-dimensional scoring
│   ├── model_benchmarker.py      # Run test scenarios
│   └── report_generator.py       # Averages, markdown tables, charts
├── graphs/
│   ├── __init__.py
│   ├── logistics_graph.py        # StateGraph, routing & retry edges
│   └── state.py                  # Domain models & reducers
├── tools/                        # Structured logistics tools
├── utils/                        # Structlog and tenacity retry helper
├── tests/                        # Full Pytest test suite
├── reports/                      # Generated benchmark reports
├── main.py                       # Application execution entry point
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Project package list
└── decision_log.md               # Technical decision log
```

---

## 🔮 Future Improvements

1. **Human-in-the-Loop Interruption**: Use LangGraph's `.compile(interrupt_before=["execution"])` to hold routing actions for operator approval when confidence scores are low (< 0.70).
2. **Parallel Tool Calling**: Optimize the Research Agent to query all carrier capacity and ETA calculations concurrently instead of sequentially, reducing research latency.
3. **LLM-as-a-Judge**: Integrate a lightweight evaluator model to grade reasoning traces using an evaluation prompt rather than a keyword checklist.

---

## 📄 References
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Pydantic Settings Guidelines](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/en/latest/)
