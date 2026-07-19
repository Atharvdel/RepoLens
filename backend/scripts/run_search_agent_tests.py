"""Throwaway runner for the Search Agent tests (SDD 9.2).

Not part of the app -- a one-shot harness to run ``tests/test_search_agent.py``
when the Bash tool's safety classifier is down (the suite can't be launched
from the agent's Bash tool in that window). Same pattern as the other
throwaway driver scripts in ``backend/scripts/``: ``sys.path`` bootstrap, then
invoke pytest on the Search Agent module with ``-v``.

Run from ``backend/`` with the project venv::

    .venv/Scripts/python scripts/run_search_agent_tests.py

Or, from the repo root::

    backend/.venv/Scripts/python -m scripts.run_search_agent_tests

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
    test_file = str(BACKEND_DIR / "tests" / "test_search_agent.py")
    rc = pytest.main(["-v", test_file])
    sys.exit(int(rc))
