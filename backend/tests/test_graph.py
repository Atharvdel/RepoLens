"""Tests for the §15 LangGraph wiring (:mod:`app.agents.graph`).

The graph owns no agent logic of its own -- it wraps the four §9 agents as
nodes and wires edges -- so this suite targets the two things the graph *does*
own: the **routing decisions** and the **compile-time structure**. Three
finite, fast, no-LLM / no-DB / no-ripgrep layers plus one live integration test:

* ``test_needs_replan_*`` / ``test_replan_target_*`` /
  ``test_route_after_planner_*`` / ``test_route_after_search_*`` -- the **pure
  routing decision matrix**. Each module-level routing function reads state
  alone (no I/O, no closures, no infra) and returns the next node name; pinned
  across the ``replans_used`` / ``had_search`` / ``total_hits`` /
  ``had_context`` Cartesian product the way the sibling tests pin
  ``_ilike_contains`` / ``_parse_*.output``. These run in any environment.

* ``test_build_graph_*`` -- **structure-valid**: :func:`build_graph` compiles
  with a no-op session (compile validates the :class:`RepoLensState` TypedDict
  resolves under ``from __future__ import annotations`` + that every conditional
  edge target is a real node -- the real compile check) and the compiled graph
  carries the four agent nodes. No LLM, no DB.

* ``test_format_context_block_*`` -- the Synthesizer's new
  :func:`_format_context_block` helper (the context_results -> prompt block added
  for §15): empty -> ``""`` (the byte-identical-to-search-only invariant), a
  real ContextStepResult renders its tool name + the "CONTEXTUAL only; NOT
  citation grounds" header, the ``max_context`` result-count cap surfaces its
  NOTE banner, and the per-result :data:`CONTEXT_RESULT_CAP` dump cap adds its
  ellipsis. Pure, no LLM/DB.

* ``test_run_query_live_flask`` -- **live**: one real flask question end-to-end
  through :func:`run_query`, asserting the graph completes and the node trace is
  well-formed (starts with ``planner``, ends with ``synthesizer``). Skips
  cleanly when Ollama is down or flask isn't indexed -- per the per-file
  ``_flask_indexed`` precedent in the sibling test files.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_graph.py -v
"""
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest
import sqlalchemy as sa

from app.agents.context_agent import ContextStepResult
from app.agents.graph import (
    GraphResult,
    RepoLensState,
    _needs_replan,
    _replan_target,
    _route_after_planner,
    _route_after_search,
    build_graph,
    run_query,
)
from app.agents.planner import OLLAMA_HOST, OLLAMA_MODEL
from app.agents.synthesizer import CONTEXT_RESULT_CAP, _format_context_block
from app.db import SessionLocal
from app.models import File, Repository, Symbol


# ─── pure routing decision matrix (no LLM, no DB, no ripgrep) ────────────────
# The four module-level routing functions decide the next node from state alone.
# Pinned across the full routing Cartesian product so a future edge edit that
# flips a decision shows up here first, not in a live run.


@pytest.mark.parametrize(
    "had_search, total_hits, replans_used, expected",
    [
        # The one §15 replan trigger: search ran, returned zero hits, no replan yet.
        (True, 0, 0, True),
        # Cap reached: a second zero-hit pass routes to synth, not back to planner.
        (True, 0, 1, False),
        # Got hits this pass: no replan needed.
        (True, 5, 0, False),
        # Over the cap (defensive -- replans_used never exceeds 1, but the
        # predicate must not route back to planner a third time).
        (True, 0, 2, False),
        # No search ran (context-only or empty plan): "zero hits" is not the
        # §15 replan trigger (no search to retry), so never replan.
        (False, 0, 0, False),
        (False, 5, 0, False),
    ],
)
def test_needs_replan_matrix(had_search, total_hits, replans_used, expected):
    """The §15 "replan once if Search returns zero hits" predicate, gated on
    had_search (so an empty/context-only plan never loops) and capped at 1 via
    replans_used."""
    assert _needs_replan(had_search, total_hits, replans_used) is expected


def test_replan_target_replan():
    """had_search + zero hits + zero replans -> back to ``planner`` (one retry)."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "total_hits": 0, "replans_used": 0
    }
    assert _replan_target(state) == "planner"


def test_replan_target_capped_goes_to_synth():
    """had_search + zero hits + replans_used >= 1 -> ``synthesizer`` (cap reached)."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "total_hits": 0, "replans_used": 1
    }
    assert _replan_target(state) == "synthesizer"


def test_replan_target_got_hits_goes_to_synth():
    """had_search + hits found -> ``synthesizer`` (nothing to replan)."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "total_hits": 5, "replans_used": 0
    }
    assert _replan_target(state) == "synthesizer"


def test_replan_target_no_search_goes_to_synth():
    """had_search False (context-only / empty plan) -> ``synthesizer`` regardless
    of total_hits -- the §15 replan trigger is search-specific, so a plan that
    never ran a search never replans."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": False, "total_hits": 0, "replans_used": 0
    }
    assert _replan_target(state) == "synthesizer"


def test_route_after_planner_search_only():
    """A search-only plan routes to ``search``."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "had_context": False
    }
    assert _route_after_planner(state) == "search"


def test_route_after_planner_search_first_when_both():
    """A plan with both search and context steps routes to ``search`` FIRST
    (sequential -- SDD 15 permits it; search runs, then context follows via
    _route_after_search), not to context."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "had_context": True
    }
    assert _route_after_planner(state) == "search"


def test_route_after_planner_context_only():
    """A context-only plan (no search) routes to ``context``."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": False, "had_context": True
    }
    assert _route_after_planner(state) == "context"


def test_route_after_planner_empty_to_synth():
    """An empty plan / planner failure (neither search nor context) routes
    straight to ``synthesizer`` -- the synthesizer says nothing matched."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": False, "had_context": False
    }
    assert _route_after_planner(state) == "synthesizer"


def test_route_after_search_continues_to_context():
    """After search, if the plan had context steps too, continue to ``context``
    (regardless of hit count) -- context runs after search in the sequential
    topology."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "had_context": True, "total_hits": 0, "replans_used": 0
    }
    assert _route_after_search(state) == "context"


def test_route_after_search_no_context_replan():
    """After search with no context steps following, fall through to the replan
    decision; zero hits + replans unused -> ``planner``."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "had_context": False, "total_hits": 0, "replans_used": 0
    }
    assert _route_after_search(state) == "planner"


def test_route_after_search_no_context_capped():
    """After search with no context, zero hits but the replan cap is reached ->
    ``synthesizer``."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "had_context": False, "total_hits": 0, "replans_used": 1
    }
    assert _route_after_search(state) == "synthesizer"


def test_route_after_search_no_context_got_hits():
    """After search with no context, hits were found -> ``synthesizer``."""
    state: RepoLensState = {  # type: ignore[typeddict-item]
        "had_search": True, "had_context": False, "total_hits": 3, "replans_used": 0
    }
    assert _route_after_search(state) == "synthesizer"


# ─── structure-valid: build_graph compiles + carries the four nodes ──────────
# No LLM, no DB: compile validates the RepoLensState TypedDict resolves under
# ``from __future__ import annotations`` AND that every conditional-edge
# path_map target is a real node; a TypedDict resolution failure or a dangling
# edge raises HERE (the real compile check), not at invoke time.


def test_build_graph_compiles():
    """``build_graph`` with a no-op session compiles without raising -- this is
    the compile-time validation of the TypedDict + every edge target (the
    cheapest real check the graph is structurally sound). The session is only
    touched by the node closures at invoke, so ``None`` is fine for build."""
    compiled = build_graph(repository_id=1, session=None)
    assert compiled is not None


def test_build_graph_has_four_agent_nodes():
    """The compiled graph carries exactly the four agent nodes (planner / search
    / context / synthesizer); ``compiled.nodes`` (a ``dict[str, PregelNode]``)
    is the stable introspection surface."""
    compiled = build_graph(repository_id=1, session=None)
    assert {"planner", "search", "context", "synthesizer"} <= set(compiled.nodes)


# ─── Synthesizer _format_context_block (the §15 context_results -> prompt) ──


@dataclass
class _FakeArchResult:
    """A plain-fields dataclass standing in for a real SDD 10 context result
    (e.g. ArchitectureResult); :func:`dataclasses.asdict` + ``json.dumps`` is
    total on it, the same way it is on the real results."""

    subject: str
    key_files: list


def test_format_context_block_empty_returns_empty_string():
    """An empty (None or []) context_results renders as ``""`` -- the invariant
    the user-message builder relies on so the prompt is byte-identical to the
    pre-context search-only prompt whenever no context steps ran (preserves the
    11 verified demo/diagnostic runs)."""
    assert _format_context_block(None, max_context=6) == ""
    assert _format_context_block([], max_context=6) == ""


def test_format_context_block_renders_tool_name_and_header():
    """A real ContextStepResult renders its tool name + Context results header."""
    csr = ContextStepResult(
        tool="architecture",
        repository_id=1,
        args={"target": None},
        result=_FakeArchResult(subject="flask", key_files=["src/flask/app.py"]),
    )
    block = _format_context_block([csr], max_context=6)
    assert block  # non-empty
    assert "architecture" in block, block
    assert "Context results" in block, block
    # The result body made it in (the key file the context tool surfaced).
    assert "src/flask/app.py" in block, block


def test_format_context_block_max_context_cap_surfaces_note():
    """``max_context`` truncates the NUMBER of results rendered and names the
    dropped tail in a NOTE banner (visible, not swallowed -- the project's
    no-silent-caps posture): with 3 results and max_context=2, the NOTE names
    "only the first 2" and exactly 2 tool-name bullets render."""
    csrs = [
        ContextStepResult(
            tool="architecture",
            repository_id=1,
            args={},
            result=_FakeArchResult(subject=f"r{i}", key_files=[]),
        )
        for i in range(3)
    ]
    block = _format_context_block(csrs, max_context=2)
    assert "only the first 2" in block, block
    # Exactly the 2 kept results render a tool-name bullet (the 3rd is dropped).
    # "- architecture" is the bullet marker for every kept row.
    assert block.count("- architecture") == 2, block


def test_format_context_block_per_result_cap_ellipsis():
    """``CONTEXT_RESULT_CAP`` truncates each per-result dump with an ellipsis so
    one ballooning architecture/dependency result can't eat the prompt budget."""
    long_subject = "x" * (CONTEXT_RESULT_CAP + 200)
    csr = ContextStepResult(
        tool="file_history",
        repository_id=1,
        args={"target": "a.py"},
        result=_FakeArchResult(subject=long_subject, key_files=[]),
    )
    block = _format_context_block([csr], max_context=6)
    assert " …" in block, block  # the per-result cap tail marker


# ─── live: one real flask question end-to-end through run_query ─────────────
# Smoke-tests the FULL graph wiring (the four nodes + the conditional edges +
# the replan loop) on the indexed flask repo. Skips cleanly when Ollama is down
# or flask isn't indexed -- per the per-file ``_flask_indexed`` precedent.


def _ollama_reachable() -> bool:
    """``GET /api/tags`` responds -> Ollama is running (a model may or may not be
    installed; the graph's nodes catch a missing-model OllamaError and record it
    to errors, so the smoke test still asserts the wiring completes)."""
    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=5) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)``: ready iff a ``flask`` repo has rows in both
    ``files`` and ``symbols`` (the search dispatch reads both via the join).
    Mirrors :func:`tests.test_search_agent._flask_indexed`."""
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


_OLLAMA_READY = _ollama_reachable()
_FLASK_READY, FLASK_REPO_ID = _flask_indexed()
live_required = pytest.mark.skipif(
    not (_OLLAMA_READY and _FLASK_READY),
    reason="needs Ollama running (reachable) + flask repo indexed in DB",
)


@live_required
def test_run_query_live_flask():
    """One real flask question end-to-end through :func:`run_query`: the graph
    completes, returns a :class:`GraphResult`, and the node trace is well-formed
    (starts with ``planner``, ends with ``synthesizer``). The Planner only emits
    ``search`` steps today (context menu not yet wired), so the trace is
    ``planner -> search -> synthesizer`` (or with a ``planner[replan]`` if this
    question happens to return zero hits on the first pass). Asserts the WIRING,
    not the model's answer quality (model-dependent + temp=0 already
    characterized in the Synthesizer known-limitation note)."""
    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    with SessionLocal() as session:
        result = run_query(
            "which file contains the Flask class",
            FLASK_REPO_ID,
            session,
            host=host,
            model=model,
        )
    assert isinstance(result, GraphResult)
    # The wiring invariant: planner first, synthesizer last. Everything in
    # between is the execution branch(es) [+ one possible replan loop].
    assert result.node_trace, "node_trace should not be empty"
    assert result.node_trace[0] == "planner", result.node_trace
    assert result.node_trace[-1] == "synthesizer", result.node_trace
    # A replan -- if it fired -- is bounded at one (the §15 cap); never more.
    assert result.replans_used <= 1, result.replans_used
