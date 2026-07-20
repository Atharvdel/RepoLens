"""Throwaway runner for the §15 LangGraph wiring tests.

Not part of the app -- a one-shot harness to run ``tests/test_graph.py`` when the
Bash tool's safety classifier is down (the suite can't be launched from the
agent's Bash tool in that window). Same pattern as
``scripts/run_search_agent_tests.py``: ``sys.path`` bootstrap, then invoke
pytest on the graph test module with ``-v``.

The pure routing + structure-valid + ``_format_context_block`` tests run with no
Ollama / no DB / no ripgrep (they pin the routing decision matrix, the
``build_graph`` compile-time validation, and the Synthesizer's context-block
helper). The single live test (``test_run_query_live_flask``) skips cleanly when
Ollama is down or flask isn't indexed, so the pure layer always runs.

Run from ``backend/`` with the project venv::

    .venv/Scripts/python scripts/run_graph_tests.py

Or, from the repo root::

    backend/.venv/Scripts/python -m scripts.run_graph_tests

Paste the output back so the agent can confirm the run.
"""
import sys
from pathlib import Path

# Match tests/conftest.py: put backend/ on sys.path so `app.*` imports resolve
# regardless of the cwd the script is launched from.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import pytest

if __name__ == "__main__":
    test_file = str(BACKEND_DIR / "tests" / "test_graph.py")
    rc = pytest.main(["-v", test_file])
    sys.exit(int(rc))
