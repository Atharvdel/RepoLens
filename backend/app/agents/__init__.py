"""RepoLens agent layer (SDD §9).

The LangGraph orchestration (SDD §15) of four agents — Planner → Search/Context
→ Synthesizer — with the Planner built first, standalone, to gauge how well the
local model handles *structured tool selection* before the full graph is wired.
Agents call the thin deterministic tools in :mod:`app.tools` (SDD §10) by name
and produce/handle structured JSON (SDD §9 "structured, never free text"), so
the Synthesizer works from facts, not paraphrased summaries.

Decoupled from :mod:`app.db` on purpose: the agent layer reads its own
``OLLAMA_HOST`` / ``OLLAMA_MODEL`` config so importing an agent does not pull
the SQLAlchemy engine / Postgres connection (agents run per-query against an
injected session opened by the future graph, SDD §15).
"""
