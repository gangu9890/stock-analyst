# Parallel Multi-Agent Stock Analyst — LangGraph

A CrewAI-free implementation of the architecture shown in the reference screenshots.

## Architecture

```text
                         ┌─ Technical Analyst ──────┐
                         ├─ Fundamental Analyst ─────┤
Ticker → validate → fan ─┼─ News Analyst ────────────┼→ Research Editor → report.md
                         └─ Consensus Analyst ───────┘
```

The fan-out uses LangGraph `Send`, so the four analyst nodes can execute in parallel. Their outputs are collected into a reducer-backed `briefings` list, then the editor runs after all four branches finish.

## Setup

Python 3.11+ is recommended.

```bash
uv venv
uv sync
cp .env.example .env
```

Put your Gemini key in `.env`:

```text
GEMINI_API_KEY=...
```

Run:

```bash
uv run python -m stock_analyst RELIANCE.NS
```

The report is written to `output/RELIANCE.NS-report.md`.

## Why LangGraph?

There is no CrewAI dependency. The orchestration is a `StateGraph` with:

- a validation node
- a conditional fan-out returning `Send(...)` for each analyst
- four independent analyst nodes
- a reducer that aggregates their results
- a final editor node

This makes the parallelism explicit in the graph instead of hiding it inside an agent framework.
