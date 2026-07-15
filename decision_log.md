# Technical Decision Log

Documenting key architectural decisions, design choices, manual modifications, and technical trade-offs made during the development of the Autonomous Logistics Agent system.

---

## 1. Architectural Decisions

### Decision 1.1: Use LangGraph StateGraph instead of CrewAI or AutoGen
- **Status**: Approved
- **Rationale**: LangGraph provides explicit control over the state machine. Since trajectory-based evaluation requires inspecting the exact input, output, and tool calls of *each* step, StateGraph's inspectable nodes and pure routing functions make it simple to track execution histories. CrewAI and AutoGen abstract the graph loop away, which limits debugging transparency and complicates intermediate trajectory capturing.
- **Trade-off**: Requires more boilerplate code to define nodes and edges compared to high-level framework DSLs.

### Decision 1.2: Capture Trajectory at Node Boundaries
- **Status**: Approved
- **Rationale**: Rather than evaluating only the final shipment route, we capture a structured `TrajectorySnapshot` (Pydantic model) at each node execution. This allows grading reasoning quality, tool arguments, latency, and token consumption step-by-step.
- **Trade-off**: Slight overhead in memory and state size since snapshots are accumulated in the state.

### Decision 1.3: Use Tenacity Retry Library for Sync/Async Backoff
- **Status**: Approved
- **Rationale**: Separates retry policies (stop conditions, exponential delays, jitter, log handlers) from the business tool logic. This ensures API timeouts and transient errors are handled uniformly without cluttering tool code.
- **Trade-off**: Introduces tenacity as an external dependency.

### Decision 1.4: Use HSL/RGB tailored color styling for Visualisation charts
- **Status**: Approved
- **Rationale**: Browser-default or plain red/blue chart styles feel generic. Tailored, curated hex colors (e.g., `#1f77b4` and `#d62728`) with twin-axis alignments ensure visual reports look modern and high-quality.

---

## 2. AI Tool & Collaboration Choices

### Decision 2.1: Model Selections — GPT-4o (Closed-Source) vs. Qwen3-8b (Open-Source)
- **Status**: Approved
- **Rationale**: GPT-4o serves as the benchmark for complex, high-reasoning tasks. Qwen3-8b (run via Groq) represents the cost-effective, low-latency open-source alternative. This setup allows comparative benchmark testing across two distinct model providers (OpenAI vs. Groq) without requiring local GPU setups.

### Decision 2.2: Mock-based Test Suite (pytest)
- **Status**: Approved
- **Rationale**: Mocks all LLM API invocations (`BaseAgent._build_llm`) and tool actions inside agent unit tests. This ensures that the testing suite is fast, deterministic, and can run in CI environments without burning live tokens or requiring API keys.

---

## 3. Manual Engineering Modifications & Bug Fixes

### Decision 3.1: Lazy Import of Agents in `logistics_graph.py`
- **Status**: Implemented to resolve import cycle
- **Rationale**: `agents/base_agent.py` imports `graphs/state.py`. Since `graphs/__init__.py` imported `graphs/logistics_graph.py` to re-export the builder, this triggered an import of `agents/planner_agent.py` (which requires `base_agent.py` to be fully loaded). This resulted in an `ImportError: cannot import name 'BaseAgent' from partially initialized module`.
- **Resolution**: Moved the imports of `PlannerAgent`, `ResearchAgent`, `DecisionAgent`, and `ExecutionAgent` **inside the `build_logistics_graph()` function body**. This deferred loading the agent classes until the graph builder is called, completely resolving the import loop.

### Decision 3.2: Escalation to Double Braces in Prompt Format Strings
- **Status**: Implemented to resolve KeyError
- **Rationale**: Literal JSON formats in the system prompts of the research, decision, and execution agents were written with single braces `{` and `}`. When `.format()` was called on the system prompts, Python treated them as formatting keys and raised a `KeyError: '\n  "reasoning"'`.
- **Resolution**: Escaped the JSON braces by doubling them (`{{` and `}}`), keeping only `{shipment_id}`, `{constraints_block}`, etc. as single braces.

### Decision 3.3: Carrier Ranking sorting logic (cost delta sorting bug)
- **Status**: Implemented to resolve sorting failure
- **Rationale**: In `graphs/state.py`, non-viable options returned a score of `-1.0` by default. However, a viable option with a high cost-delta (e.g. DHL Express costing $71k, with original cost $15k) returned a score of `-4.665`. Since `-4.665 < -1.0`, non-viable options FX and UPS (score `-1.0`) were sorted *above* the viable option (DHL Express), breaking the Decision Agent's priority candidate table.
- **Resolution**: Changed the non-viable option default score to `-999999.0` so they are guaranteed to remain at the bottom of the list.

---

## 4. Technical Trade-offs

### Modularity vs. Simplicity
- **Trade-off**: The project is split into separate modules: `config/`, `utils/`, `tools/`, `agents/`, `graphs/`, and `evaluation/`. While this introduces folder structure overhead, it maintains the **Single Responsibility Principle**. Each agent node is isolated, and adding new carriers, routes, or notification channels does not touch the core LangGraph code.

### Heuristic reasoning scoring vs. LLM-as-a-judge
- **Trade-off**: The Reasoning Quality score uses a keyword checklist instead of an LLM evaluator. A keyword checklist runs instantly, costs $0, and is 100% deterministic (reproducible). An LLM-as-a-judge is more flexible at grading style but is non-deterministic, slow, and expensive. Heuristics were chosen to fit within the 3-4 hour scope of the assessment, while leaving LLM-as-a-judge as a future improvement.
