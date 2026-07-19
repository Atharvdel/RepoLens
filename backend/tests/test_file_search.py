"""Tests for the File Search tool (SDD §10).

Same split as :mod:`tests.test_symbol_search`:

* ``test_ilike_contains_*`` — deterministic pure, no DB: pin the escape helper.
* ``test_search_files_*`` — live Postgres against the indexed flask repo (skipped
  cleanly when flask isn't indexed / Postgres is down): a known file is found with
  the right shape (path, language, loc, ISO-8601 ``last_modified``);
  case-insensitive partial matching is pinned; a nonsense query returns an empty
  list, not an error; and the ``repository_id`` filter is shown to be
  load-bearing.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_file_search.py -v
"""
import datetime as _dt

import pytest
import sqlalchemy as sa

from app.db import SessionLocal
from app.models import File, Repository, Symbol
from app.tools.file_search import (
    FileResult,
    LIKE_ESCAPE,
    _ilike_contains,
    search_files,
)

# ─── flask readiness probe ─────────────────────────────────────────────────────
# Identical to test_symbol_search's — kept per-file to match the per-file
# `_cleanup` precedent (test_import_graph / test_reference_index) rather than
# coupling conftest to DB state. If this ever diverges from symbol_search's
# probe, mirror it back.


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)``: ready iff a ``flask`` repo has rows in both ``files``
    and ``symbols`` (file-search reads ``files``; the probe checks both so the two
    tool test-suites share one readiness definition)."""
    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return False, None
            n_files = session.scalar(
                sa.select(sa.func.count())
                .select_from(File)
                .where(File.repository_id == repo.id)
            ) or 0
            n_syms = session.scalar(
                sa.select(sa.func.count())
                .select_from(Symbol)
                .join(File, Symbol.file_id == File.id)
                .where(File.repository_id == repo.id)
            ) or 0
            return (
                n_files > 0 and n_syms > 0,
                repo.id if (n_files and n_syms) else None,
            )
    except Exception:
        return False, None


_FLASK_READY, FLASK_REPO_ID = _flask_indexed()
flask_required = pytest.mark.skipif(
    not _FLASK_READY,
    reason="flask repo not indexed in DB (run scripts/run_walker_once.py first)",
)


# ─── deterministic pure: the escape helper (no DB) ────────────────────────────


def test_ilike_contains_wraps_and_escapes():
    """A path fragment is wrapped in ``%…%`` and its LIKE metacharacters escaped
    so they match literally (the analog of test_symbol_search's helper test, for
    this tool's own copy of the helper)."""
    assert _ilike_contains("app.py") == "%app.py%"
    assert _ilike_contains("a_b") == r"%a\_b%"  # _ escaped
    assert _ilike_contains("a%b") == r"%a\%b%"  # % escaped
    assert _ilike_contains("a\\b") == r"%a\\b%"  # backslash doubled
    assert LIKE_ESCAPE == "\\"


# ─── live Postgres against the indexed flask repo ─────────────────────────────


@flask_required
def test_search_files_finds_known_file():
    """``src/flask/app.py`` is found by the fragment ``app.py`` with the full SDD
    §10 File Search shape — path, language ('python'), a positive loc, and an
    ISO-8601 ``last_modified`` (the tool serializes the DB datetime so the result
    is JSON-ready)."""
    with SessionLocal() as session:
        results = search_files(FLASK_REPO_ID, "app.py", session)

    assert results, "expected at least one file matching 'app.py'"
    app = next((r for r in results if r.path == "src/flask/app.py"), None)
    assert app is not None, [r.path for r in results]
    assert app.language == "python", app
    assert app.loc >= 1, app
    # last_modified is serialized to an ISO string (the walker set it from fs
    # mtime, so it is present for indexed files — not None).
    assert isinstance(app.last_modified, str), app
    assert app.last_modified
    # Round-trips through fromisoformat (naive, no TZ — matches the walker's
    # datetime.fromtimestamp), proving the datetime was ISO-encoded, not dropped
    # to a repr or left as a datetime the JSON layer couldn't serialize.
    _dt.datetime.fromisoformat(app.last_modified)
    assert all(isinstance(r, FileResult) for r in results)


@flask_required
def test_search_files_case_insensitive_and_partial():
    """``ILIKE '%query%'`` on the path: uppercase ``APP.PY`` still finds
    ``src/flask/app.py`` (case-insensitive), and the fragment ``blueprints``
    finds ``src/flask/blueprints.py`` (partial substring, not just a
    prefix/exact)."""
    with SessionLocal() as session:
        upper = search_files(FLASK_REPO_ID, "APP.PY", session)
        partial = search_files(FLASK_REPO_ID, "blueprints", session)

    upper_paths = {r.path for r in upper}
    partial_paths = {r.path for r in partial}
    assert "src/flask/app.py" in upper_paths, upper_paths
    assert "src/flask/blueprints.py" in partial_paths, partial_paths


@flask_required
def test_search_files_nonsense_query_returns_empty_not_error():
    """A nonsense query returns an empty list — not an error (SDD §10: an empty
    result is a valid answer)."""
    with SessionLocal() as session:
        results = search_files(FLASK_REPO_ID, "zzqqxxnofile9991.py", session)
    assert results == [], results


@flask_required
def test_search_files_is_scoped_to_repository():
    """The ``repository_id`` filter is load-bearing: a nonexistent repo returns no
    files for 'app.py' (proves repo-scoping, not a global scan)."""
    with SessionLocal() as session:
        results = search_files(999_999_999, "app.py", session)
    assert results == [], results
