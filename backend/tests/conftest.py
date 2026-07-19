"""Pytest configuration shared by RepoLens backend tests.

Puts the ``backend/`` directory on ``sys.path`` so test modules can
``import app.*`` regardless of how pytest is invoked (cwd, rootdir, editor
"run test" gutter, CI). Kept deliberately minimal — no fixtures that couple
tests to a particular DB state; the live-Postgres smoke test manages its
own session lifecycle.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
