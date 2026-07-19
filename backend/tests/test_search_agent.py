"""Tests for the Search Agent dispatcher (SDD 9.2).

The Search Agent is a pure, no-LLM dispatcher -- it takes one plan step (the
``{"agent": "search", "tool": <name>, "args": {...}}`` shape the Planner emits,
SDD 9.1) and routes it to the real tool function in :mod:`app.tools`, injecting
the orchestrator-supplied ``repository_id`` + ``Session``. Two test layers,
matching the project's pure-core / live split used by the sibling tool tests:

* ``test_dispatch_* malformed / pure helpers`` -- **deterministic pure, no DB,
  no ripgrep**: an unknown/malformed tool name, a non-object step, a non-object
  ``args``, and a missing ``query`` all raise :class:`SearchDispatchError` (a
  :class:`ValueError` subclass) with a message that names the problem and lists
  the valid tools -- i.e. fail *clearly*, never silently. The private
  kwarg-forwarding / regex-coercion helpers are pinned directly the way the
  sibling tests pin ``_ilike_contains`` / ``_parse_*.output``.

* ``test_dispatch_*_returns_real_hits`` -- **live**: one dispatch test per tool
  that confirms a plan step dispatches to the *right* tool and returns its real
  structured hits from the indexed flask repo (``SymbolResult`` /
  ``FileResult`` / ``TextHit``), in the consistent :class:`SearchStepResult`
  wrapper the Synthesizer will consume. Symbol/file dispatch need flask indexed
  in Postgres (DB-backed tools); text dispatch additionally needs ripgrep on
  PATH and the on-disk clone (it shells out). All skip cleanly when their
  precondition is missing -- per the per-file ``_flask_indexed`` /
  ``rg_required`` precedent in the sibling tool test files.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_search_agent.py -v
"""
import pytest
import sqlalchemy as sa

from app.agents.search_agent import (
    SEARCH_TOOLS,
    SearchDispatchError,
    SearchStepResult,
    _coerce_regex,
    _forward_kwargs,
    build_search_step_result,
    dispatch_search_step,
)
from app.db import SessionLocal
from app.indexing import ripgrep_available
from app.models import File, Repository, Symbol
from app.tools.file_search import FileResult
from app.tools.symbol_search import SymbolResult
from app.tools.text_search import TextHit

# --- flask readiness probes ------------------------------------------------
# Kept per-file (mirrors test_symbol_search / test_file_search / test_text_search):
# conftest deliberately couples no DB state, so each test file owns a probe that
# returns ``(ready, repo_id)`` and defensively swallows any DB / filesystem error
# to ``(False, None)`` so collection in a fresh environment skips cleanly.
#
# Two probes because the live dispatch tests have different preconditions:
#   * symbol + file dispatch are pure Postgres reads -> DB-only probe.
#   * text dispatch shells out to ripgrep against the *on-disk* clone -> needs
#     the clone present too. (And a separate ``rg_required`` skip for ripgrep.)


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)`` for the **DB-backed** dispatch tests: ready iff a
    ``flask`` repo has rows in both ``files`` and ``symbols`` (symbol-search
    reads both via the join; file-search reads ``files``; checking both keeps
    one readiness definition for the two DB-only dispatch tests)."""
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


def _flask_on_disk_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)`` for the **text** dispatch test: like
    :func:`_flask_indexed` but also requires the clone to exist on disk at the
    path the walker stored (``url_or_path``) and at least one indexed file to
    resolve under it -- Text Search shells ripgrep at that root, so a missing or
    moved clone must skip rather than error. Mirrors
    :mod:`tests.test_text_search`'s probe."""
    from pathlib import Path

    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return False, None
            root = repo.url_or_path
            if not root or not Path(root).is_dir():
                return False, None
            sample_path = session.scalar(
                sa.select(File.path)
                .where(File.repository_id == repo.id)
                .order_by(File.path)
                .limit(1)
            )
            if not sample_path or not (Path(root) / sample_path).exists():
                return False, None
            return True, repo.id
    except Exception:
        return False, None


_FLASK_READY, FLASK_REPO_ID = _flask_indexed()
flask_required = pytest.mark.skipif(
    not _FLASK_READY,
    reason="flask repo not indexed in DB "
    "(run scripts/run_walker_once.py + index_all_flask_symbols.py first)",
)

_FLASK_ON_DISK_READY, FLASK_ON_DISK_REPO_ID = _flask_on_disk_indexed()
flask_on_disk_required = pytest.mark.skipif(
    not _FLASK_ON_DISK_READY,
    reason="flask repo not indexed in DB / not on disk "
    "(run scripts/run_walker_once.py first)",
)

rg_required = pytest.mark.skipif(
    not ripgrep_available(), reason="ripgrep binary not on PATH"
)


# --- deterministic pure dispatch (no DB, no ripgrep) ------------------------
# The malformed-step contract: fail CLEARLY (raise SearchDispatchError naming
# the problem), never silently return an empty result that a downstream agent
# would misread as "nothing matched". These run with no Postgres / no ripgrep,
# so they pin the contract in any environment.


def test_dispatch_unknown_tool_raises_search_dispatch_error():
    """An unknown tool name raises :class:`SearchDispatchError` (not a silent
    empty result, not a bare ``KeyError``): the message names the bad tool AND
    lists the valid tools, so a developer / the future graph knows exactly what
    went wrong. ``SearchDispatchError`` is a :class:`ValueError` subclass so a
    caller that treats it as generic bad input still catches it."""
    step = {"agent": "search", "tool": "frobnicate", "args": {"query": "Flask"}}
    with pytest.raises(SearchDispatchError) as exc_info:
        build_search_step_result(step, repository_id=1, session=None)
    msg = str(exc_info.value)
    # Names the bad tool and the valid set -- the "fails clearly" contract.
    assert "frobnicate" in msg, msg
    assert "search_symbols" in msg and "search_text" in msg, msg
    # ValueError subclass: a generic bad-input handler still catches it.
    assert isinstance(exc_info.value, ValueError)


def test_dispatch_missing_tool_key_raises():
    """A step with no ``tool`` key at all raises (the value is ``None``, which is
    not a string) -- not silently dispatched, not a quiet ``[]``."""
    step = {"agent": "search", "args": {"query": "Flask"}}
    with pytest.raises(SearchDispatchError):
        build_search_step_result(step, repository_id=1, session=None)


def test_dispatch_non_string_tool_raises():
    """A step whose ``tool`` is a non-string (e.g. a number from a confused
    model) raises rather than being used as a dict key / splatted."""
    step = {"agent": "search", "tool": 123, "args": {"query": "Flask"}}
    with pytest.raises(SearchDispatchError):
        build_search_step_result(step, repository_id=1, session=None)


def test_dispatch_non_dict_step_raises():
    """A step that is not an object (a list, a bare string) raises -- the
    Planner is documented to emit an object per step; a garbage step surfaces
    here rather than crashing on ``step.get``."""
    for bad in ([], "search_symbols", 42, None):
        with pytest.raises(SearchDispatchError):
            build_search_step_result(bad, repository_id=1, session=None)


def test_dispatch_non_dict_args_raises():
    """A step whose ``args`` is present but not an object (a list, a string)
    raises -- the kwargs-forwarding logic needs a dict; a malformed ``args``
    surfaces here, not as a confusing ``AttributeError`` deeper in."""
    step = {"agent": "search", "tool": "search_symbols", "args": ["Flask"]}
    with pytest.raises(SearchDispatchError):
        build_search_step_result(step, repository_id=1, session=None)


def test_dispatch_missing_query_raises():
    """A step whose ``args`` has no ``query`` raises :class:`SearchDispatchError`
    rather than passing ``None`` down to the tool (which would crash with an
    unhelpful ``AttributeError`` on ``query.strip()``). The Planner's validator
    should have caught this first, but the dispatcher does not trust the plan.
    Note the difference from an *empty-string* query, which the tools treat as
    a valid empty result -- the missing-KEY case is the dispatch error."""
    for args in ({}, {"kind": "class"}):  # no 'query' key
        step = {"agent": "search", "tool": "search_symbols", "args": args}
        with pytest.raises(SearchDispatchError) as exc_info:
            build_search_step_result(step, repository_id=1, session=None)
        assert "query" in str(exc_info.value), exc_info.value


def test_search_tools_registry_matches_planner_known_tools():
    """The dispatcher's :data:`SEARCH_TOOLS` keys are exactly the three real
    tool function names the Planner is allowed to emit (:data:`KNOWN_TOOLS`) --
    so a plan the Planner produces is dispatchable as-is, and the two modules
    can't drift (a tool added to one but not the other would be a silent
    dead-plan-step bug)."""
    from app.agents.planner import KNOWN_TOOLS

    assert set(SEARCH_TOOLS) == set(KNOWN_TOOLS), (
        set(SEARCH_TOOLS),
        set(KNOWN_TOOLS),
    )
    assert set(SEARCH_TOOLS) == {"search_symbols", "search_files", "search_text"}


def test_search_tools_registry_keyword_sets_match_planner():
    """The forwardable-kwarg set carried per tool in :data:`SEARCH_TOOLS` mirrors
    the Planner's :data:`OPTIONAL_TOOL_ARGS` -- so the dispatcher forwards
    exactly the kwargs the Planner is permitted to emit, no more, no less."""
    from app.agents.planner import OPTIONAL_TOOL_ARGS

    for name, (_fn, allowed) in SEARCH_TOOLS.items():
        assert allowed == OPTIONAL_TOOL_ARGS[name], (name, allowed, OPTIONAL_TOOL_ARGS[name])


# --- pure helpers (no DB, no ripgrep) ---------------------------------------


def test_coerce_regex_handles_bool_int_and_string_forms():
    """``regex`` arrives as a bool (correct), a number, or a weak-model string
    (``"true"``/``"false"``); the coercion yields a real bool in every case so
    ``search_text``'s ``regex: bool`` never sees a string."""
    assert _coerce_regex(True) is True
    assert _coerce_regex(False) is False
    assert _coerce_regex("true") is True
    assert _coerce_regex("TRUE") is True
    assert _coerce_regex("yes") is True
    assert _coerce_regex("false") is False
    assert _coerce_regex("0") is False
    assert _coerce_regex(1) is True
    assert _coerce_regex(0) is False
    # A garbage string is just falsy -- never raises.
    assert _coerce_regex("maybe") is False
    assert _coerce_regex(None) is False


def test_forward_kwargs_whitelists_and_drops_unknown():
    """``_forward_kwargs`` forwards ONLY the tool's allowed kwargs present in
    ``args``; a stray ``cap`` / ``repository_id`` / ``session`` a misbehaving
    plan happens to include is dropped on the floor -- so a planner mistake
    never reaches the tool as an unexpected ``TypeError``-triggering kwarg.
    ``regex`` is bool-coerced on the way through; ``kind`` passes verbatim."""
    sym_allowed = frozenset({"kind"})
    # kind is forwarded verbatim; a stray 'query'/'cap'/'repository_id' is dropped
    # (query is positional, not a kwarg -- it crosses via build_search_step_result,
    # not here).
    fwd = _forward_kwargs(
        {"kind": "class", "query": "Flask", "cap": 5, "repository_id": 9},
        sym_allowed,
    )
    assert fwd == {"kind": "class"}, fwd

    txt_allowed = frozenset({"regex"})
    # regex is bool-coerced ("true" str -> True); stray keys dropped.
    fwd = _forward_kwargs(
        {"regex": "true", "cap": 50, "session": object()},
        txt_allowed,
    )
    assert fwd == {"regex": True}, fwd
    # regex=False string -> False
    fwd = _forward_kwargs({"regex": "false"}, txt_allowed)
    assert fwd == {"regex": False}, fwd

    # file search has no allowed kwargs -> nothing forwarded regardless of args.
    fwd = _forward_kwargs({"query": "app.py", "regex": True}, frozenset())
    assert fwd == {}, fwd
    # absence of an allowed kwarg -> empty (not a KeyError).
    assert _forward_kwargs({}, sym_allowed) == {}
    assert _forward_kwargs({"query": "x"}, sym_allowed) == {}


# --- live dispatch against the indexed flask repo ---------------------------
# One test per tool: a plan step in the Planner's shape dispatches to the RIGHT
# tool and returns that tool's real structured hits, wrapped in the consistent
# SearchStepResult (tool name echoes back; args carries the query + forwarded
# kwargs; hits are the tool's concrete dataclass).


@flask_required
def test_dispatch_search_symbols_returns_real_hits():
    """A ``search_symbols`` plan step dispatches to symbol-search and returns
    real :class:`SymbolResult` hits from flask: the ``Blueprint`` class (in
    ``src/flask/blueprints.py``) is found with ``kind="class"``, and the
    returned ``SearchStepResult`` carries the tool name, the injected
    ``repository_id``, the args actually used (query + the forwarded ``kind``),
    and the hits. Pins both the dispatch routing AND the kwarg-forwarding (the
    ``kind`` filter reached the tool -- only ``class`` rows return)."""
    step = {
        "agent": "search",
        "tool": "search_symbols",
        "args": {"query": "Blueprint", "kind": "class"},
    }
    with SessionLocal() as session:
        result = dispatch_search_step(step, FLASK_REPO_ID, session)

    assert isinstance(result, SearchStepResult), result
    assert result.tool == "search_symbols", result.tool
    assert result.repository_id == FLASK_REPO_ID, result.repository_id
    # The args echo the query + the forwarded kind kwarg only (no stray keys).
    assert result.args == {"query": "Blueprint", "kind": "class"}, result.args
    assert result.hits, "expected Blueprint class hits from flask"
    assert all(isinstance(h, SymbolResult) for h in result.hits), result.hits
    # The kind filter reached the tool: every hit is a class, and Blueprint is
    # among them (proving dispatch + kind-forwarding both work end-to-end).
    assert all(h.kind == "class" for h in result.hits), [h.kind for h in result.hits]
    assert any(h.name == "Blueprint" for h in result.hits), [h.name for h in result.hits]


@flask_required
def test_dispatch_search_files_returns_real_hits():
    """A ``search_files`` plan step dispatches to file-search and returns real
    :class:`FileResult` hits: ``app.py`` finds ``src/flask/app.py``. ``search_files``
    takes no optional kwargs, so a step that carries a stray ``kind``/``regex``
    (a confused plan) drops it -- the dispatch still succeeds and the returned
    args carry only ``query`` (uncontaminated by the dropped kwarg). Pins the
    whitelist-drop behavior end-to-end through the live tool."""
    step = {
        "agent": "search",
        "tool": "search_files",
        "args": {"query": "app.py", "kind": "bogus_should_be_dropped"},
    }
    with SessionLocal() as session:
        result = dispatch_search_step(step, FLASK_REPO_ID, session)

    assert isinstance(result, SearchStepResult), result
    assert result.tool == "search_files", result.tool
    assert result.repository_id == FLASK_REPO_ID, result.repository_id
    # search_files has no allowed kwargs -> 'kind' dropped, args carry only query.
    assert result.args == {"query": "app.py"}, result.args
    assert result.hits, "expected app.py file hits from flask"
    assert all(isinstance(h, FileResult) for h in result.hits), result.hits
    assert any(h.path == "src/flask/app.py" for h in result.hits), [h.path for h in result.hits]


@rg_required
@flask_on_disk_required
def test_dispatch_search_text_returns_real_hits():
    """A ``search_text`` plan step dispatches to text-search (ripgrep against the
    on-disk flask clone, ``repo_root`` resolved from the repositories row) and
    returns real :class:`TextHit` hits: the known docstring phrase ``URL rules,
    template configuration`` appears in ``src/flask/app.py`` (the ``Flask`` class
    docstring) and ``src/flask/sansio/app.py`` (the shared base class), so
    ``>= 2`` hits come back. The ``regex`` kwarg (sent here as the string
    ``"false"``, a weak-model habit) is bool-coerced and forwarded; the returned
    args carry the coerced bool. Pins dispatch routing + regex coercion +
    the search_text tool's on-disk-repo-root resolution, end-to-end."""
    step = {
        "agent": "search",
        "tool": "search_text",
        "args": {"query": "URL rules, template configuration", "regex": "false"},
    }
    with SessionLocal() as session:
        result = dispatch_search_step(step, FLASK_ON_DISK_REPO_ID, session)

    assert isinstance(result, SearchStepResult), result
    assert result.tool == "search_text", result.tool
    assert result.repository_id == FLASK_ON_DISK_REPO_ID, result.repository_id
    # 'regex' was string-co-erced to a real bool and forwarded.
    assert result.args == {
        "query": "URL rules, template configuration",
        "regex": False,
    }, result.args
    assert result.hits, "expected text hits for the known flask docstring phrase"
    assert all(isinstance(h, TextHit) for h in result.hits), result.hits
    # The primary anchor: app.py carries the phrase (mirror test_text_search's
    # anchor; the sansio line is allowed to drift, so it's asserted only via the
    # multi-file >=2 count, not by line).
    assert any(
        h.file == "src/flask/app.py" and "URL rules, template configuration" in h.matched_text
        for h in result.hits
    ), [(h.file, h.line) for h in result.hits]
    assert len(result.hits) >= 2, [(h.file, h.line) for h in result.hits]
