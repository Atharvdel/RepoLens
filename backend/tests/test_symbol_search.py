"""Tests for the Symbol Search tool (SDD §10).

Two layers, matching the project's pure-core / live-DB split:

* ``test_ilike_contains_*`` — *deterministic pure, no DB*: pin the
  :func:`_ilike_contains` escape helper. The LIKE-metacharacter escaping is the
  subtle correctness core (a literal ``__init__`` must match its underscores, not
  the single-char wildcard ``_``); this pins it without Postgres, without flask,
  the way ``test_parse_ripgrep_output_*`` pins the reference-index parse.
* ``test_search_symbols_*`` — *live Postgres against the indexed flask repo*
  (skipped cleanly when flask isn't indexed / Postgres is down). The flask repo is
  already in the DB (``repositories.name == 'flask'``; the walker + symbol-parser
  scripts populated ``files`` and ``symbols`` — SDD §7 steps 1–3). These read that
  ground truth: a known symbol is found with the right shape; case-insensitive
  partial matching and the optional ``kind`` filter are pinned; a nonsense query
  returns an empty list, not an error; and the ``repository_id`` filter is shown to
  be load-bearing.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_symbol_search.py -v
"""
import pytest
import sqlalchemy as sa

from app.db import SessionLocal
from app.models import File, Repository, Symbol
from app.tools.symbol_search import (
    LIKE_ESCAPE,
    SymbolResult,
    _ilike_contains,
    search_symbols,
)

# ─── flask readiness probe (skip, not error, when flask isn't indexed) ─────────
# Defensively swallows any DB error / absence to `(False, None)` so collection in
# a fresh environment skips cleanly — mirrors the `rg_required` skip in
# test_reference_index. Kept per-file (rather than in conftest) to match the
# per-file `_cleanup` precedent in test_import_graph / test_reference_index and
# keep conftest minimal (it deliberately couples no DB state).


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)``: ready iff a ``flask`` repo has rows in both ``files``
    and ``symbols`` (the symbol-search tests touch both tables)."""
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
    reason="flask repo not indexed in DB (run scripts/run_walker_once.py "
    "+ index_all_flask_symbols.py first)",
)


# ─── deterministic pure: the escape helper (no DB) ────────────────────────────


def test_ilike_contains_wraps_plain_query_preserving_case():
    """A plain query is wrapped in ``%…%`` for substring matching, with its case
    untouched — case-insensitivity comes from ``ILIKE`` at query time, not from
    the helper mutating the query."""
    assert _ilike_contains("Widget") == "%Widget%"
    assert _ilike_contains("widget") == "%widget%"


def test_ilike_contains_escapes_like_metacharacters():
    """``_`` (single-char wildcard), ``%`` (multi-char wildcard), and the escape
    char itself are all escaped so they match literally."""
    assert _ilike_contains("__init__") == r"%\_\_init\_\_%"
    assert _ilike_contains("a%b") == r"%a\%b%"
    assert _ilike_contains("a\\b") == r"%a\\b%"  # backslash doubled
    assert LIKE_ESCAPE == "\\"


# ─── live Postgres against the indexed flask repo ─────────────────────────────


@flask_required
def test_search_symbols_finds_known_symbol():
    """The bedrock flask ``Flask`` class (in ``src/flask/app.py``) is found with
    the full SDD §10 shape — name, kind, file path (joined from the ``file_id``
    FK), a sane line range, and a plumbed (nullable) docstring field."""
    with SessionLocal() as session:
        results = search_symbols(FLASK_REPO_ID, "Flask", session)

    assert results, "expected at least one symbol matching 'Flask'"
    flask_class = next(
        (
            r
            for r in results
            if r.name == "Flask" and r.kind == "class" and r.file == "src/flask/app.py"
        ),
        None,
    )
    assert flask_class is not None, [(r.name, r.kind, r.file) for r in results]
    # Shape: a real, ordered line range and a plumbed (nullable) docstring field.
    assert flask_class.line_start >= 1, flask_class
    assert flask_class.line_end >= flask_class.line_start, flask_class
    assert flask_class.docstring is None or isinstance(flask_class.docstring, str)
    # Every row is the right type — catches a stray ORM row / dict sneaking in.
    assert all(isinstance(r, SymbolResult) for r in results)


@flask_required
def test_search_symbols_case_insensitive_and_partial():
    """``ILIKE '%query%'``: lowercase finds the capitalized ``Flask`` class
    (case-insensitive), and a mid-name substring ``lueprint`` finds ``Blueprint``
    (partial — the match is a substring, not a prefix/exact)."""
    with SessionLocal() as session:
        lower = search_symbols(FLASK_REPO_ID, "flask", session)
        partial = search_symbols(FLASK_REPO_ID, "lueprint", session)

    assert "Flask" in {r.name for r in lower}, {r.name for r in lower}
    assert "Blueprint" in {r.name for r in partial}, {r.name for r in partial}


@flask_required
def test_search_symbols_kind_filter_restricts_and_excludes():
    """The optional ``kind`` filter restricts results to one kind and excludes
    others: with ``kind="class"`` only classes return (and the ``Flask`` class is
    among them); with ``kind="function"`` the ``Flask`` *class* is absent (a class
    is not a function) — proving the filter is applied, not silently ignored."""
    with SessionLocal() as session:
        classes = search_symbols(FLASK_REPO_ID, "Flask", session, kind="class")
        functions = search_symbols(FLASK_REPO_ID, "Flask", session, kind="function")

    assert classes, "expected class results for 'Flask' kind='class'"
    assert all(r.kind == "class" for r in classes), [r.kind for r in classes]
    assert any(r.name == "Flask" and r.file == "src/flask/app.py" for r in classes)

    assert all(r.kind == "function" for r in functions), [r.kind for r in functions]
    # The Flask *class* must not appear under a function filter — if the filter
    # were ignored, the class would leak back in.
    assert not any(r.name == "Flask" and r.kind == "class" for r in functions), [
        r for r in functions if r.name == "Flask"
    ]


@flask_required
def test_search_symbols_nonsense_query_returns_empty_not_error():
    """A nonsense query returns an empty list — not an error, not None (SDD §10:
    an empty result is a valid answer)."""
    with SessionLocal() as session:
        results = search_symbols(FLASK_REPO_ID, "zzqqxxnotasymbol9991", session)
    assert results == [], results


@flask_required
def test_search_symbols_is_scoped_to_repository():
    """The ``repository_id`` filter is load-bearing: searching 'Flask' against a
    nonexistent repository returns nothing — proving the tool is repo-scoped, not
    a global scan that would have leaked flask's ``Flask`` through."""
    with SessionLocal() as session:
        results = search_symbols(999_999_999, "Flask", session)
    assert results == [], results
