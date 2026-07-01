# AutoAnalyst — A Self-Correcting Multi-Agent LLM System for Automated, Bias-Aware Data Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2016-black)
![LLM](https://img.shields.io/badge/LLM-LangChain-1c3c3c)
![License](https://img.shields.io/badge/License-MIT-green)

An automated "data scientist" you talk to. Upload a CSV, ask a question in plain
English, and a team of LLM agents **plans** the analysis, **writes and runs its
own Python code**, generates charts, **interprets** them from multiple opposing
viewpoints, and returns a single **neutral, evidence-backed report** — while
streaming its entire reasoning graph to the browser in real time.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
  - [Layer 1 — Execution (Task Graph)](#layer-1--execution-task-graph)
  - [Layer 2 — Self-Correcting Code (Action Graph)](#layer-2--self-correcting-code-action-graph)
  - [Layer 3 — Bias-Contrastive Analysis](#layer-3--bias-contrastive-analysis)
- [End-to-End Request Lifecycle](#end-to-end-request-lifecycle)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Human interpretation of data is inherently biased, and LLMs — used as automated
analysts — tend to either mirror those biases or hide behind a false sense of
"neutrality." **AutoAnalyst** takes a different route: instead of asking one
model to be objective, it deliberately generates **several biased
interpretations** and then **mathematically reconciles them** into a neutral
result.

The engine is built on a **hierarchical graph architecture**. A planner agent
decomposes a request into a Directed Acyclic Graph (DAG) of tasks; each task
generates and self-corrects executable Python; and the resulting charts are
interpreted by persona agents whose bias is measured and cancelled out.

## Key Features

- **Natural-language → full analysis.** Ask a question over a CSV; get code,
  charts, and a written report.
- **Autonomous, self-correcting code generation.** Agents run their own Python,
  read the tracebacks, and fix themselves.
- **Graph-based orchestration.** Tasks execute in dependency order via
  topological sorting, not a fixed script.
- **Bias-contrastive interpretation.** Multiple persona agents read the charts
  through opposing lenses; a synthesizer merges them.
- **Quantified neutrality.** An embedding-based *Neutrality Index* scores how
  biased each interpretation is.
- **Real-time visualization.** The task graph, node statuses, and per-persona
  insights stream live to the UI over Server-Sent Events.
- **Multi-turn sessions.** The task graph and history persist, so follow-up
  questions refine the existing workflow.

---

## System Architecture

AutoAnalyst is organized into three cooperating layers. The first two derive
from the *Data Interpreter* hierarchical-graph approach; the third is the
project's original contribution.

```
                          ┌──────────────────────────────┐
  USER ── query + CSV ───▶│  Next.js UI  (SSE streaming)  │
                          │  React Flow task-graph view   │
                          └───────────────┬───────────────┘
                                          │  HTTP + Server-Sent Events
                          ┌───────────────▼───────────────┐
                          │   FastAPI  (async, streaming)  │
                          │   MasterAgent orchestrator     │
                          └───────────────┬───────────────┘
                                          │
┌─────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — EXECUTION (Task Graph / DAG)                                    │
│   MasterAgent decomposes the query into TaskNodes                         │
│   (data_loading → feature_engineering → model_training → visualization),  │
│   ordered by topological sort over their dependencies.                    │
└───────────────┬───────────────────────────────────────────────────────────┘
                │  per task
┌───────────────▼───────────────────────────────────────────────────────────┐
│ LAYER 2 — SELF-CORRECTING CODE (Action Graph)                             │
│   Each TaskNode generates a sequence of Python snippets (ActionNodes),    │
│   runs them, captures stdout/stderr, and RE-PLANS on failure until the    │
│   code succeeds or the retry budget is exhausted.                         │
│   → produces .png charts, saved models, and a shared agent_state          │
└───────────────┬───────────────────────────────────────────────────────────┘
                │  charts
┌───────────────▼───────────────────────────────────────────────────────────┐
│ LAYER 3 — BIAS-CONTRASTIVE ANALYSIS (the novelty)                         │
│   N persona agents "read" the charts via a vision LLM, each with an       │
│   injected bias (Bull / Bear / Skeptic). A BiasEvaluator scores each       │
│   interpretation's Neutrality Index; a Synthesizer merges the conflicting  │
│   views into one neutral, cited report.                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 1 — Execution (Task Graph)

The **`MasterAgent`** ([`master.py`](backend/app/services/agent/master.py))
receives the user query plus a token-cheap summary of the uploaded data
(`df.info()` rather than raw rows). Using a LangChain structured-output call, it
produces a **`TaskGraph`** ([`graph.py`](backend/app/services/agent/graph.py)) —
a DAG of `TaskNode`s. Each node has:

- a **task type** (`data_loading`, `exploration`, `feature_engineering`,
  `model_training`, `evaluation`, `visualization`),
- an **instruction** describing what it should achieve,
- and a list of **dependencies** on other nodes.

`get_execution_order()` runs a **topological sort** (`graphlib.TopologicalSorter`)
so tasks always run after their prerequisites, and cycles are detected and
rejected. On follow-up turns, the graph is not rebuilt from scratch — the master
issues **atomic edits** (`ADD` / `MODIFY` / `DELETE`) that mutate the persisted
graph, enabling true multi-turn refinement.

### Layer 2 — Self-Correcting Code (Action Graph)

For each task, `plan_actions()` asks the LLM to break the instruction into an
ordered list of executable Python snippets — an **`ActionGraph`** of
`ActionNode`s ([`action_graph.py`](backend/app/services/agent/action_graph.py)).
These run sequentially inside a **`CodeExecutor`**
([`utils.py`](backend/app/services/agent/utils.py)), which executes the code in a
shared namespace and captures both `stdout` and `stderr`.

The **self-correction loop** is the core of this layer:

1. Generate code → execute.
2. If it raises, capture the traceback.
3. Feed the failing plan **and** the error back into the LLM as conversation
   history and **re-plan** (`replan_actions()`).
4. Repeat up to `ACTION_GRAPH_MAX_RETRIES` times.

There is a **second, outer loop** at the task level: if a task still fails after
exhausting its code retries, the `MasterAgent` refines the *whole task graph*
and re-runs (up to `TASK_GRAPH_MAX_RETRIES`). Agents pass information forward
through a lightweight, whitelisted `agent_state` dictionary; heavy artifacts
(models, matrices) are written to disk and referenced by path. Successful tasks
emit `.png` charts into the session's `figures/` directory.

### Layer 3 — Bias-Contrastive Analysis

Once charts exist, the **`AnalysisAgent`**
([`sub_agents.py`](backend/app/services/agent/sub_agents.py)) takes over:

1. **Curation.** An LLM "editor" selects the most relevant charts for the query.
2. **Multi-persona debate.** Each **persona** (e.g. *The Bull* — optimistic,
   *The Bear* — pessimistic, *The Skeptic* — data-integrity focused) is a
   vision-LLM call with the chart images base64-encoded and a **dynamically
   injected bias** in the system prompt. All personas run **concurrently**
   (`ThreadPoolExecutor`).
3. **Bias measurement.** The **`BiasEvaluator`**
   ([`bias_evaluator.py`](backend/app/services/agent/bias_evaluator.py)) embeds
   each persona's text with `all-MiniLM-L6-v2` (SentenceTransformers, CPU-only)
   and compares it against averaged "bullish" and "bearish" anchor vectors:
   - `bias_score = cos_sim(text, positive_anchor) − cos_sim(text, negative_anchor)`
   - `neutrality_index = 1 − |bias_score|`  (0 → 1, higher = more neutral)
   - plus a raw TextBlob sentiment polarity as a cross-check.
4. **Synthesis.** A "Neutral Arbitrator" LLM call merges the conflicting
   interpretations into one objective report, explicitly noting where the
   personas disagreed and embedding the relevant charts inline.

This layer is what turns "the model said so" into a **measurable, reconciled**
conclusion.

---

## End-to-End Request Lifecycle

```
1. User submits prompt + CSV      →  Next.js /api/process → FastAPI POST /api/v1/process
2. Files saved to session/{id}/data/; a background task starts; session_id returned
3. Frontend opens SSE stream       →  GET /api/v1/process/events/{session_id}
4. MasterAgent.run_request():
     a. plan/refine TaskGraph                     ── SSE graph_init  ──▶ UI draws DAG
     b. execute_pipeline() in topological order:
          for each TaskNode:                      ── SSE node_update ──▶ UI colors node
            generate code → run → (re-plan on error)
            success → write charts + update agent_state
     c. select key diagrams
     d. run all persona agents (vision LLM)
     e. BiasEvaluator scores each interpretation
     f. synthesize the neutral report
5. Final report + per-persona views + charts       ── SSE response ──▶ UI renders result
6. Task graph + conversation history persisted for the next turn
```

---

## Tech Stack

**Backend — API & orchestration**
- **FastAPI** + **Uvicorn** — async web server and the `/process` endpoints.
- **Server-Sent Events** (via `StreamingResponse` + `asyncio.Queue`) — live progress streaming.
- **Pydantic / pydantic-settings** — structured LLM output validation and layered configuration.

**Backend — agents & LLMs**
- **LangChain** (`langchain-openai`, `langchain-google-genai`) — model abstraction, structured output, prompt templating, vision message construction.
- **`graphlib`** — topological ordering of the task DAG.
- **`concurrent.futures`** — parallel persona execution.

**Backend — data science (used by generated code)**
- **pandas**, **NumPy**, **scikit-learn**, **matplotlib** (`Agg`), **seaborn**, **scipy**, **statsmodels**.

**Backend — bias evaluation**
- **sentence-transformers** (`all-MiniLM-L6-v2`) — CPU embeddings for the Neutrality Index.
- **TextBlob** — sentiment cross-check.

**Backend — infrastructure**
- **SQLAlchemy (async)** + **MySQL**, **Alembic**, **Redis**, **Celery** — persistence, migrations, caching, task queue.

**Frontend**
- **Next.js 16** + **React 19** + **TypeScript**.
- **@xyflow/react (React Flow)** + **dagre** — interactive, auto-laid-out agent task-graph.
- **Tailwind CSS v4**, **Radix UI**, **lucide-react**, **sonner**, **next-themes** — UI system.

---

## Project Structure

```
backend/
  app/services/agent/
    master.py          MasterAgent — orchestrates the full run
    graph.py           TaskGraph / TaskNode — Layer 1 DAG + execution pipeline
    action_graph.py    ActionGraph / ActionNode — Layer 2 code container
    sub_agents.py      AnalysisAgent — persona vision agents (Layer 3)
    bias_evaluator.py  Neutrality Index scoring
    forecast.py        ForecastAgent — time-series persona forecasting
    schemas.py         Pydantic structured-output + state schemas
    utils.py           CodeExecutor + SessionWorkspace (filesystem)
    file_utils.py      CSV/image formatting for the LLM
  core/config.py       Layered settings (models, prompts, personas)
  settings.json        Runtime prompts, personas, and model config
  app/.../process.py   FastAPI endpoints + SSE streaming
  docker/              MySQL + Redis compose stack
frontend/
  app/                 Next.js pages + API proxy routes
  hooks/use-chat-stream.ts   SSE client
  components/graph/    React Flow task-graph visualization
  components/chat/     Chat, persona grid, artifact viewer
dev/
  langgraph.ipynb      Original research prototype
  *.csv                Sample datasets
```

---

## Getting Started

### 1. Infrastructure (MySQL + Redis)

```bash
cd backend/docker
docker compose up -d
```

### 2. Backend

```bash
cd backend
# Create a .env with your model provider key, e.g.:
#   OPENAI_API_KEY=...
#   GEMINI_API_KEY=...     (free tier: https://aistudio.google.com/apikey)
poetry install
poetry run python3 main.py --env local --debug
```

The API serves on `http://0.0.0.0:8000` (`/docs` for Swagger).

### 3. Frontend

```bash
cd frontend
npm install
# Create .env.local with:  UPSTREAM_URL=http://0.0.0.0:8000
npm run dev
```

Open <http://localhost:3000>.

---

## Configuration

All agent behavior is data-driven via **`backend/settings.json`**:

- **`prompts`** — system prompts for the planner, code generator, and analysis agents.
- **`personas`** — the biased analysts (role, icon, injected bias).
- **`llm_config`** — model name, temperature, timeout, token limits.
- **`graph_config`** — retry budgets for the action- and task-level self-correction loops.

Changing personas or prompts requires **no code changes**.

---

## Roadmap

### Target Architecture (with roadmap improvements)

The diagram below shows the **full enhanced system** once the planned work is in
place. Components marked `[PLANNED]` are on the roadmap; everything else is
already implemented.

```
                          ┌────────────────────────────────────────────┐
  USER ── query + CSV ───▶│  Next.js UI (React Flow + SSE streaming)     │
                          │  [PLANNED] Plan-Approval Gate ◀── human edits│
                          └───────────────────────┬────────────────────-┘
                                                  │  HTTP + Server-Sent Events
                          ┌───────────────────────▼─────────────────────┐
                          │  FastAPI orchestration (MasterAgent)          │
                          │  [PLANNED] Model Router (OpenAI/Gemini/local) │
                          │  [PLANNED] Token & Cost Tracker               │
                          └───────────────────────┬─────────────────────-┘
                                                  │
┌───────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — EXECUTION (Task Graph / DAG)                                      │
│   MasterAgent decomposes query → TaskNodes, ordered by topological sort.    │
│   [PLANNED] Plan-Validity check (acyclic, dependency-complete, on-scope).   │
│   [PLANNED] Human-in-the-loop: pause for plan approval before execution.    │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  per task
┌───────────────▼─────────────────────────────────────────────────────────────┐
│ LAYER 2 — SELF-CORRECTING CODE (Action Graph)                               │
│   Generate Python → run → read stderr → re-plan on failure (nested loops).  │
│   [PLANNED] 🔒 Dockerized sandboxed executor (replaces in-process exec()).  │
│   [PLANNED] Metrics logged: execution success, retries-to-success,          │
│             self-correction recovery rate.                                  │
│   → .png charts, saved models, shared agent_state                           │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │  charts
┌───────────────▼─────────────────────────────────────────────────────────────┐
│ LAYER 3 — BIAS-CONTRASTIVE ANALYSIS                                         │
│   Persona vision agents → BiasEvaluator (Neutrality Index).                 │
│   [PLANNED] Confidence-aware Synthesizer: weight personas by execution-     │
│             grounding + cross-persona agreement → neutral report +          │
│             per-run confidence score.                                       │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────────────────┐
│ [PLANNED] EVALUATION HARNESS  (offline, over a benchmark of analytical      │
│           queries with known-correct outcomes)                              │
│   Plan validity • Execution success • Self-correction recovery •            │
│   Bias-reduction delta • End-to-end success • Confidence                    │
│   → metrics report (CSV/JSON)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Planned enhancements (designed, in progress)

- **Sandboxed execution** — run generated Python inside a Docker container /
  restricted environment instead of in-process, for isolation and safety.
- **Evaluation harness** — a benchmark of analytical queries with a pytest-based
  runner measuring, per layer:
  - *Plan validity rate* (acyclic, dependency-complete DAGs),
  - *Execution success rate* and *self-correction recovery rate*,
  - *Bias-reduction delta* (neutrality of the synthesized report vs. individual personas),
  - *End-to-end task success rate*.
- **Confidence-aware synthesis** — weight persona inputs by execution-grounding
  and cross-persona agreement, surfaced as a per-run confidence score.
- **Multi-provider model routing** — per-task selection across OpenAI / Gemini /
  local models for cost and quality trade-offs.
- **Token & cost tracking** and **context-window optimization**.
- **Human-in-the-loop checkpoints** — approve or edit the plan before execution.

---

## License

MIT — see [`LICENSE`](LICENSE).
