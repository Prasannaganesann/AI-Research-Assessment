# Technical Decision Log

This document records the architectural decisions, design choices, manual modifications, and technical trade-offs made during the development of the Multi-Agent System (MAS) for supply chain resilience.

---

## 1. Architectural Decisions

### Decision 1.1: Select LangGraph StateGraph over CrewAI or AutoGen
- **Status**: Approved
- **Rationale**: 
  - **Deterministic Control Flow**: Supply chain incident resolution requires a highly controlled state machine where agents execute in a strict, auditable order.
  - **Fine-Grained Telemetry**: Since trajectory-based evaluation requires inspecting the exact input, output, and tool calls of each step, LangGraph’s inspectable nodes and transparent state update boundaries make it simple to append snapshots. High-level frameworks like CrewAI or AutoGen abstract away the runtime execution loop, which introduces non-deterministic hand-offs and restricts intermediate telemetry hook capture.
- **Trade-off**: Requires writing more boilerplate code to define state structures, nodes, and routing edges compared to high-level DSL frameworks.

### Decision 1.2: Capture Trajectory snapshots at Node Boundaries
- **Status**: Approved
- **Rationale**: 
  - **Intermediate Evaluation Surface**: Evaluating only final outcomes misses logical flaws and sequence violations (e.g. sending a notification prior to booking validation). Appending a structured `TrajectorySnapshot` (Pydantic model) at each node execution exposes tool argument accuracy, reasoning traces, latency, and cost step-by-step.
- **Trade-off**: Modestly increases graph state memory overhead since snapshots are accumulated throughout the workflow.

### Decision 1.3: Decouple Retry Policies via Tenacity Decorators
- **Status**: Approved
- **Rationale**: 
  - **Separation of Concerns**: Utilizing the `tenacity` library isolates retry policies (e.g., stopping limits, exponential backoff, random jitter, and log handlers) from the core tools business logic. This ensures API timeouts and transient errors are handled uniformly across all tools without repeating error recovery code.
- **Trade-off**: Adds an external library dependency.

### Decision 1.4: Employ Curated HSL/RGB Colors for Visual Reports
- **Status**: Approved
- **Rationale**: 
  - **Professional Presentation**: Standard red/green/blue styles from basic plotting libraries look unrefined. Using tailored, curated hex color schemes (e.g., `#1a73e8` and `#ea4335`) and clear twin-axis alignments in the generated report charts ensures visual results are polished and ready for executive review.
- **Trade-off**: Requires manual styling setup in matplotlib instead of using default style sheets.

---

## 2. AI Model & Testing Choices

### Decision 2.1: Model Pairs — GPT-4o (Closed-Source) vs. Qwen3-8B (Open-Source)
- **Status**: Approved
- **Rationale**: 
  - **Performance Comparison**: GPT-4o establishes the ceiling for reasoning and planning capabilities. Qwen3-8B (hosted via the Groq API) represents a high-speed, cost-effective open-source alternative. This setup allows comparative benchmark testing across two distinct model providers (OpenAI vs. Groq) without requiring local GPU compute.

### Decision 2.2: Mock-Based Deterministic Unit Testing (pytest)
- **Status**: Approved
- **Rationale**: 
  - **Fast Execution & Cost Savings**: Mocks all live API calls at the `BaseAgent._build_llm` boundary and isolates tool inputs using pre-configured mock values. This makes the test suite fast, deterministic, and runnable in offline environments without consuming live API credits or leaking keys.
- **Trade-off**: Mocks do not catch runtime schema changes or API updates; these must be validated in integration testing.

---

## 3. Manual Engineering Modifications & Bug Fixes

### Decision 3.1: Implement Lazy Agent Loading to Prevent Circular Imports
- **Status**: Implemented
- **Rationale**: 
  - **Dependency Loop**: `graphs/state.py` is imported by `agents/base_agent.py`. The `graphs/__init__.py` file imported the graph builder factory in `graphs/logistics_graph.py`, which imported the agent subclasses at the module scope. This triggered a circular dependency loop where `BaseAgent` was loaded before the module was fully initialized, throwing an `ImportError`.
  - **Resolution**: Refactored `graphs/logistics_graph.py` to perform agent class imports *locally inside* the `build_logistics_graph()` factory method. This defers class loading until the factory is executed, breaking the circular loop.

### Decision 3.2: Escape JSON Braces in System Prompt String Formats
- **Status**: Implemented
- **Rationale**: 
  - **KeyError Crashes**: The system prompts for the Research, Decision, and Execution agents contained literal JSON structures for formatting instructions. When `.format()` was invoked to inject shipment context, Python interpreted the literal JSON braces `{` and `}` as replacement placeholders and failed with a `KeyError: '\n  "reasoning"'`.
  - **Resolution**: Doubled the literal JSON braces (`{{` and `}}`) in the string definitions, keeping only the actual context placeholders as single braces.

### Decision 3.3: Set Default Score for Non-Viable Options to -999999.0
- **Status**: Implemented
- **Rationale**: 
  - **Ranking Inversion**: In `graphs/state.py`, non-viable carrier options initially returned a score of `-1.0` by default. However, when evaluating a highly viable option with a large cost delta (e.g. DHL Express costing $71k, with original cost $15k), its score dropped to `-4.665`. Because `-4.665 < -1.0`, non-viable options (score `-1.0`) were incorrectly sorted *above* the viable option.
  - **Resolution**: Replaced the default non-viable score with `-999999.0`, ensuring rejected carrier options are always sorted at the very bottom of the candidate list.

---

## 4. Technical Trade-offs

### Modularity vs. Simplicity
- **Trade-off**: The codebase is split into `config/`, `utils/`, `tools/`, `agents/`, `graphs/`, and `evaluation/`. While this creates directory overhead, it strictly enforces the **Single Responsibility Principle**. Individual agent nodes are completely isolated, making the workflow easy to extend with new tools or routing logic.

### Heuristic Trace Scoring vs. LLM-as-a-Judge
- **Trade-off**: Reasoning quality is graded using a deterministic keyword-matching heuristic rather than an LLM judge. A heuristic check runs instantly, incurs $0 in API costs, and is 100% reproducible. While an LLM judge is better at style evaluation, it is slower, non-deterministic, and costly. Heuristics were chosen to match the target assessment scope, while leaving LLM-as-a-judge as a future enhancement.
