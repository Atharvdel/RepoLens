"""Tests for the Context Agent dispatcher (SDD §9.3).

The Context Agent is a pure, no-LLM dispatcher (the structural twin of
:mod:`app.agents.search_agent`) -- it takes one plan step (the
``{"agent": "context", "tool": <name>, "args": {...}}`` shape the Planner
emits, SDD §9.1) and routes it to the real Context tool in :mod:`app.tools`,
injecting the orchestrator-supplied ``repository_id`` + ``Session`` and
forwarding only the whitelisted optional kwargs. Two test layers, matching the
Search Agent tests:

* ``test_dispatch_*`` malformed-step / pure helpers -- *deterministic pure, no
  DB*: an unknown/malformed tool name, a non-object step/args, and a missing
  REQUIRED arg (``target`` for the identifier-keyed tools) all raise
  :class:`ContextDispatchError` (a :class:`ValueError` subclass) with a message
  that names the problem and lists the valid context tools -- fail *clearly*,
  never silently. The private kwarg-forwarding helper and the per-tool required/
  optional-arg registry are pinned directly (the same way the Search Agent tests
  pin ``_forward_kwargs`` / the SEARCH_TOOLS registry). These run anywhere.

* ``test_dispatch_*_live`` -- *live against the indexed flask repo*: one
  dispatch test per tool that confirms a plan step dispatches to the RIGHT
  Context tool and returns its real structured result wrapped in the consistent
  :class:`ContextStepResult` (tool name echoes back; the injected
  ``repository_id``; the args actually used; the tool's concrete dataclass).
  The two graph tools assert real flask structure; the two metadata tools assert
  their clear-empty-until-§7 interim posture. All skip cleanly when their
  precondition is missing -- per the per-file ``flask_imports_required`` /
  ``flask_required`` precedent in the sibling test files.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_context_agent.py -v
"""
from __future__ import annotations

import pytest
import sqlalchemy as sa

from app.agents.context_agent import (
    CONTEXT_OPTIONAL_ARGS,
    CONTEXT_REQUIRED_ARGS,
    CONTEXT_TOOLS,
    ContextDispatchError,
    ContextStepResult,
    _forward_kwargs,
    build_context_step_result,
    dispatch_context_step,
)
from app.db import SessionLocal
from app.models import Edge, File, Repository
from app.tools.architecture import ArchitectureResult
from app.tools.dependency_graph import DependencyGraphResult
from app.tools.file_history import FileHistoryResult
from app.tools.github_metadata import GitHubMetadataResult

# --- flask readiness probes (per-file, mirrors test_search_agent) ------------
# Two probes because the live dispatch tests have different preconditions:
#   * the two graph tools read the internal import-graph slice -> imports indexed.
#   * the two metadata tools just need a flask file to resolve (file_history) or
#     a flask repo to exist (github_metadata) -> files indexed.


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)`` for the metadata-tools' live dispatch tests: ready
    iff a ``flask`` repo has ``files`` rows (file_history needs a resolvable
    target with ``last_modified``; github_metadata just needs the repo)."""
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
            return (n_files > 0, repo.id if n_files else None)
    except Exception:
        return False, None


def _flask_imports_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)`` for the graph tools' live dispatch tests: ready iff
    flask has ``files`` rows AND at least one internal import edge
    (``edge_type="imports"``, ``source_type="file"``, ``target_type="file"``) --
    the slice architecture + dependency_graph read."""
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
            n_edges = session.scalar(
                sa.select(sa.func.count())
                .select_from(Edge)
                .where(
                    Edge.repository_id == repo.id,
                    Edge.edge_type == "imports",
                    Edge.source_type == "file",
                    Edge.target_type == "file",
                    Edge.target_id.is_not(None),
                )
            ) or 0
            ready = n_files > 0 and n_edges > 0
            return (ready, repo.id if ready else None)
    except Exception:
        return False, None


_FLASK_READY, FLASK_REPO_ID = _flask_indexed()
flask_required = pytest.mark.skipif(
    not _FLASK_READY,
    reason="flask repo not indexed in DB (run scripts/run_walker_once.py first)",
)

_FLASK_IMPORTS_READY, FLASK_IMPORTS_REPO_ID = _flask_imports_indexed()
flask_imports_required = pytest.mark.skipif(
    not _FLASK_IMPORTS_READY,
    reason="flask import graph not indexed (run scripts/index_all_flask_imports.py)",
)


# ═══════════════════════════════════════════════════════════════════════════
# deterministic pure dispatch (no DB) -- the malformed-step contract
# ═══════════════════════════════════════════════════════════════════════════
# Each raises ContextDispatchError BEFORE the tool is called (validation runs
# first; a session=None never reaches the tool), so these need no Postgres.
# They pin the "fail CLEARLY, never a silent empty result" contract.


def test_dispatch_unknown_tool_raises_context_dispatch_error():
    """An unknown tool name raises :class:`ContextDispatchError` (not a silent
    empty, not a bare ``KeyError``): the message names the bad tool AND lists
    the valid context tools. ``ContextDispatchError`` is a :class:`ValueError`
    subclass so a generic bad-input handler still catches it."""
    step = {"agent": "context", "tool": "frobnicate", "args": {"target": "x.py"}}
    with pytest.raises(ContextDispatchError) as exc_info:
        build_context_step_result(step, repository_id=1, session=None)
    msg = str(exc_info.value)
    assert "frobnicate" in msg, msg
    assert all(t in msg for t in CONTEXT_TOOLS), msg  # lists the valid set
    assert isinstance(exc_info.value, ValueError)


def test_dispatch_missing_tool_key_raises():
    """A step with no ``tool`` key (value ``None``) raises -- not silently
    dispatched."""
    step = {"agent": "context", "args": {"target": "x.py"}}
    with pytest.raises(ContextDispatchError):
        build_context_step_result(step, repository_id=1, session=None)


def test_dispatch_non_string_tool_raises():
    """A non-string ``tool`` (a confused model emitting a number) raises rather
    than being used as a dict key / dispatched."""
    step = {"agent": "context", "tool": 123, "args": {"target": "x.py"}}
    with pytest.raises(ContextDispatchError):
        build_context_step_result(step, repository_id=1, session=None)


def test_dispatch_non_dict_step_raises():
    """A non-object step (list, bare string, number, None) raises -- the
    Planner is documented to emit an object per step; garbage surfaces here
    rather than crashing on ``step.get``."""
    for bad in ([], "architecture", 42, None):
        with pytest.raises(ContextDispatchError):
            build_context_step_result(bad, repository_id=1, session=None)


def test_dispatch_non_dict_args_raises():
    """``args`` present but not an object (list, string) raises -- the kwargs
    forwarding needs a dict; a malformed ``args`` surfaces here rather than as
    a confusing ``AttributeError`` deeper in."""
    step = {"agent": "context", "tool": "file_history", "args": ["x.py"]}
    with pytest.raises(ContextDispatchError):
        build_context_step_result(step, repository_id=1, session=None)


def test_dispatch_missing_required_target_raises():
    """The identifier-keyed tools (``dependency_graph`` / ``file_history``)
    REQUIRE ``target`` -- a step missing it raises :class:`ContextDispatchError`
    (not passed down as ``None``, which the resolver would turn into an empty
    result a downstream agent would MISREAD as "Planner meant a whole-repo
    query"). The message names ``target``. Pins the per-tool heterogeneous
    required-args model -- distinct from the Search Agent's shared ``query``."""
    for tool in ("dependency_graph", "file_history"):
        step = {"agent": "context", "tool": tool, "args": {}}
        with pytest.raises(ContextDispatchError) as exc_info:
            build_context_step_result(step, repository_id=1, session=None)
        assert "target" in str(exc_info.value), (tool, exc_info.value)
        # And a step with extra-but-not-target kwargs still requires target.
        step2 = {"agent": "context", "tool": tool, "args": {"depth": 2}}
        with pytest.raises(ContextDispatchError):
            build_context_step_result(step2, repository_id=1, session=None)


def test_dispatch_missing_required_target_message_names_tool():
    """The error message names WHICH tool is missing the arg (so a developer
    / the future graph knows which plan step was malformed)."""
    step = {"agent": "context", "tool": "file_history", "args": {}}
    with pytest.raises(ContextDispatchError) as exc_info:
        build_context_step_result(step, repository_id=1, session=None)
    msg = str(exc_info.value)
    assert "file_history" in msg, msg


# ═══════════════════════════════════════════════════════════════════════════
# pure: registry consistency (the CONTEXT_TOOLS single-source-of-truth check)
# ═══════════════════════════════════════════════════════════════════════════


def test_context_tools_registry_all_four_tools():
    """``CONTEXT_TOOLS`` keys are exactly the four real Context tool function
    names (architecture, dependency_graph, file_history, github_metadata) --
    the names a plan step carries (once the Planner's context menu is wired)
    dispatch as-is, and there's no stray half-wired entry."""
    assert set(CONTEXT_TOOLS) == {
        "architecture",
        "dependency_graph",
        "file_history",
        "github_metadata",
    }, set(CONTEXT_TOOLS)


def test_context_required_args_derived_from_registry():
    """``CONTEXT_REQUIRED_ARGS`` is derived from :data:`CONTEXT_TOOLS`
    (single source of truth) -- a tool added to the registry appears in both
    maps automatically, so the two can't drift. Pins the derivation + the
    heterogeneous per-tool required set: the identifier-keyed tools require
    ``target``; ``architecture`` / ``github_metadata`` require nothing."""
    for name, (_fn, required, _opt) in CONTEXT_TOOLS.items():
        assert CONTEXT_REQUIRED_ARGS[name] == required, (name, required)
    # Heterogeneous required set — the load-bearing assertion.
    assert CONTEXT_REQUIRED_ARGS["architecture"] == frozenset()
    assert CONTEXT_REQUIRED_ARGS["github_metadata"] == frozenset()
    assert CONTEXT_REQUIRED_ARGS["dependency_graph"] == frozenset({"target"})
    assert CONTEXT_REQUIRED_ARGS["file_history"] == frozenset({"target"})


def test_context_optional_args_derived_from_registry():
    """``CONTEXT_OPTIONAL_ARGS`` is likewise derived from :data:`CONTEXT_TOOLS`
    -- each tool's optional-knob set (the whitelist the dispatcher forwards),
    pinned so the knobs the Planner is permitted to emit are exactly these."""
    for name, (_fn, _req, optional) in CONTEXT_TOOLS.items():
        assert CONTEXT_OPTIONAL_ARGS[name] == optional, (name, optional)
    assert CONTEXT_OPTIONAL_ARGS["architecture"] == frozenset({"target", "top_k", "radius"})
    assert CONTEXT_OPTIONAL_ARGS["dependency_graph"] == frozenset({"depth"})
    assert CONTEXT_OPTIONAL_ARGS["file_history"] == frozenset({"recent_cap"})
    assert CONTEXT_OPTIONAL_ARGS["github_metadata"] == frozenset({"target"})


def test_context_required_and_optional_are_disjoint_per_tool():
    """For each tool, required and optional arg sets are disjoint (an arg
    can't be both required and optional). ``architecture`` and
    ``github_metadata`` list ``target`` as OPTIONAL (whole-repo when absent),
    never required; ``dependency_graph`` / ``file_history`` list it REQUIRED"""
    for name in CONTEXT_TOOLS:
        req, opt = CONTEXT_REQUIRED_ARGS[name], CONTEXT_OPTIONAL_ARGS[name]
        assert not (req & opt), (name, req, opt)


# ═══════════════════════════════════════════════════════════════════════════
# pure: _forward_kwargs (whitelist, not splat)
# ═══════════════════════════════════════════════════════════════════════════


def test_forward_kwargs_whitelists_and_drops_unknown():
    """``_forward_kwargs`` forwards ONLY the requested ``allowed`` keys present
    in ``args``; a stray ``cap`` / ``repository_id`` / ``session`` a
    misbehaving plan happens to include is dropped -- so a planner mistake
    never reaches a tool as an unexpected ``TypeError``-triggering kwarg. It
    iterates over ``allowed`` (not ``args``), so a plan can't smuggle in an
    unforwarded key by merely naming it."""
    # dependency_graph knob set:
    fwd = _forward_kwargs(
        {"depth": 2, "target": "x.py", "recent_cap": 5, "repository_id": 9},
        frozenset({"depth"}),
    )
    assert fwd == {"depth": 2}, fwd
    # file_history knob set:
    fwd = _forward_kwargs(
        {"recent_cap": 3, "target": "x.py", "depth": 2, "session": object()},
        frozenset({"recent_cap"}),
    )
    assert fwd == {"recent_cap": 3}, fwd
    # architecture knobs:
    fwd = _forward_kwargs(
        {"top_k": 5, "radius": 1, "target": "x.py", "depth": 9},
        frozenset({"target", "top_k", "radius"}),
    )
    assert fwd == {"top_k": 5, "radius": 1, "target": "x.py"}, fwd


def test_forward_kwargs_absence_is_empty_not_error():
    """An allowed key absent from ``args`` → not forwarded (no KeyError); an
    empty ``args`` / empty ``allowed`` yields ``{}``."""
    assert _forward_kwargs({}, frozenset({"depth"})) == {}
    assert _forward_kwargs({"target": "x"}, frozenset({"depth"})) == {}
    assert _forward_kwargs({"depth": 2}, frozenset()) == {}


# ═══════════════════════════════════════════════════════════════════════════
# live dispatch against the indexed flask repo
# ═══════════════════════════════════════════════════════════════════════════
# One test per tool: a plan step in the Planner's shape dispatches to the RIGHT
# Context tool and returns that tool's real structured result, wrapped in the
# consistent ContextStepResult (tool name + injected repository_id + the args
# actually used + the tool's concrete dataclass). The graph tools assert real
# flask structure; the metadata tools assert their clear-empty-until-§7 shape.


@flask_imports_required
def test_dispatch_architecture_whole_repo_live():
    """A ``architecture`` step with no ``target`` dispatches to the architecture
    tool's whole-repo view (an empty required set, so no ``target`` is a valid
    call, not a dispatch error): scope "whole", a non-empty flask module map +
    edges. ``result.args`` carries only the args actually used -- here ``{}``
    (no target, no knobs). Pins dispatch routing + the empty-required-set path
    + the args-echo contract end-to-end."""
    step = {"agent": "context", "tool": "architecture", "args": {}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert isinstance(result, ContextStepResult)
    assert result.tool == "architecture"
    assert result.repository_id == FLASK_IMPORTS_REPO_ID
    # No target, no knobs -> args used are exactly {}.
    assert result.args == {}, result.args
    assert isinstance(result.result, ArchitectureResult)
    assert result.result.scope == "whole"
    assert result.result.focus_path is None
    assert result.result.modules, "flask architecture should have a module map"
    assert result.result.edges, "flask architecture should have import edges"


@flask_imports_required
def test_dispatch_architecture_focused_live():
    """A ``architecture`` step with ``target="src/flask/app.py"`` dispatches to
    the focused (file-scope) view: scope "file", focus_path resolved, an
    induced-subgraph edge set. ``result.args`` echoes the target (positionally
    forwarded, not as a stray kwarg)."""
    step = {"agent": "context", "tool": "architecture", "args": {"target": "src/flask/app.py"}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert isinstance(result, ContextStepResult)
    assert result.tool == "architecture"
    # target forwarded positionally (not as a colliding kwarg); echoed in args.
    assert result.args == {"target": "src/flask/app.py"}, result.args
    assert isinstance(result.result, ArchitectureResult)
    assert result.result.scope == "file"
    assert result.result.focus_path == "src/flask/app.py"
    assert result.result.edges, "app.py should have a focused subgraph"


@flask_imports_required
def test_dispatch_dependency_graph_live():
    """A ``dependency_graph`` step (``target`` required) dispatches to the
    dependency-graph tool and returns app.py's neighborhood: it imports
    (neighbors_out) and is depended on (neighbors_in). ``result.args`` echoes
    the target. Pins the required-target validation path (no false dispatch
    error) + the dispatch routing end-to-end."""
    step = {"agent": "context", "tool": "dependency_graph", "args": {"target": "src/flask/app.py"}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert isinstance(result, ContextStepResult)
    assert result.tool == "dependency_graph"
    assert result.repository_id == FLASK_IMPORTS_REPO_ID
    assert result.args == {"target": "src/flask/app.py"}, result.args
    assert isinstance(result.result, DependencyGraphResult)
    assert result.result.node_path == "src/flask/app.py"
    assert result.result.depth == 1  # default depth
    assert result.result.neighbors_out, "app.py should import other flask files"
    assert result.result.neighbors_in, "app.py should be imported by other flask files"


@flask_imports_required
def test_dispatch_dependency_graph_forwards_depth_and_drops_stray_kwargs_live():
    """A ``dependency_graph`` step carrying ``depth=2`` AND stray ``recent_cap``
    + ``top_k`` (knobs the tool does NOT accept) dispatches with only ``depth``
    forwarded (whitelist drop); ``result.args`` echoes target + depth only;
    ``result.result.depth == 2``. Pins the whitelist + depth forwarding through
    the live tool end-to-end."""
    step = {
        "agent": "context",
        "tool": "dependency_graph",
        "args": {
            "target": "src/flask/app.py",
            "depth": 2,
            "recent_cap": 5,  # file_history knob -- must be DROPPED here
            "top_k": 3,       # architecture knob -- must be DROPPED here
        },
    }
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert result.tool == "dependency_graph"
    # Only target + depth survived the whitelist (stray recent_cap/top_k dropped).
    assert result.args == {"target": "src/flask/app.py", "depth": 2}, result.args
    assert isinstance(result.result, DependencyGraphResult)
    assert result.result.depth == 2  # the forwarded knob reached the tool


@flask_imports_required
def test_dispatch_architecture_forwards_top_k_and_radius_live():
    """An ``architecture`` step carrying ``top_k`` + ``radius`` knobs forwards
    them; ``top_k=1`` bounds the key-files list to one entry. Pins knob
    forwarding + the _coerce_int floor (top_k floored at 1) end-to-end."""
    step = {
        "agent": "context",
        "tool": "architecture",
        "args": {"target": "src/flask/app.py", "top_k": 1, "radius": 1, "depth": 9},
    }
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert result.tool == "architecture"
    # target + top_k + radius forwarded; depth (architecture doesn't take it) dropped.
    assert result.args == {"target": "src/flask/app.py", "top_k": 1, "radius": 1}, result.args
    assert isinstance(result.result, ArchitectureResult)
    assert result.result.scope == "file"
    assert len(result.result.key_files) == 1, result.result.key_files


@flask_required
def test_dispatch_file_history_clear_empty_until_section7_live():
    """A ``file_history`` step dispatches to the history tool; against flask
    today it returns the clear-empty-until-§7 shape (``last_modified`` set by
    the walker, empty contributor/commit lists until §7 step 7 lands).
    ``result.args`` echoes the target. Pins the required-target path + the
    honest interim backing posture through the live tool."""
    # Pick a real flask file path to target.
    with SessionLocal() as session:
        sample = session.scalar(
            sa.select(File.path)
            .where(File.repository_id == FLASK_REPO_ID)
            .order_by(File.path)
            .limit(1)
        )
    assert sample, "flask should have indexed files"

    step = {"agent": "context", "tool": "file_history", "args": {"target": sample}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_REPO_ID, session)

    assert isinstance(result, ContextStepResult)
    assert result.tool == "file_history"
    assert result.args == {"target": sample}, result.args
    assert isinstance(result.result, FileHistoryResult)
    assert result.result.file_path == sample
    assert isinstance(result.result.top_contributors, list)
    assert isinstance(result.result.recent_commits, list)
    assert len(result.result.top_contributors) >= 1 or len(result.result.recent_commits) >= 1


@flask_required
def test_dispatch_github_metadata_whole_repo_clear_empty_until_section8_live():
    """A ``github_metadata`` step with no ``target`` (whole-repo, valid because
    its required set is empty) dispatches and returns the clear-empty-until-§7
    shape (both lists empty until §7 step 8 PyGithub lands). ``result.args`` is
    ``{}`` (no target, no knobs). Pins the empty-required-set path + interim
    backing posture."""
    step = {"agent": "context", "tool": "github_metadata", "args": {}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_REPO_ID, session)

    assert isinstance(result, ContextStepResult)
    assert result.tool == "github_metadata"
    assert result.args == {}, result.args
    assert isinstance(result.result, GitHubMetadataResult)
    assert result.result.file_path is None
    assert result.result.issues == [] and result.result.prs == []


@flask_imports_required
def test_dispatch_github_metadata_focused_unresolvable_clear_empty_live():
    """A ``github_metadata`` step with a ``target`` that resolves to a real flask
    file (but the issues table is empty until §7 step 8) → ``file_path`` set to
    the resolved path, empty issues/PRs. Distinguishes "target resolved, no
    linked metadata yet" (file_path set) from "target unresolvable" (file_path
    None) -- both clear-empty, different shape."""
    step = {"agent": "context", "tool": "github_metadata", "args": {"target": "src/flask/app.py"}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert result.tool == "github_metadata"
    assert result.args == {"target": "src/flask/app.py"}, result.args
    assert isinstance(result.result, GitHubMetadataResult)
    # target resolved to a real path (the file exists) even though no metadata.
    assert result.result.file_path == "src/flask/app.py"
    assert result.result.issues == [] and result.result.prs == []


@flask_imports_required
def test_dispatch_github_metadata_unresolvable_target_clear_empty_live():
    """A ``github_metadata`` step with an unresolvable ``target`` → clear empty
    with ``file_path=None`` (the "don't guess" posture -- the resolver returned
    None, so no file filter is applied and NO rows should match an unresolvable
    path anyway; the honest empty). Distinguishes from the resolved-but-unlinked
    case above (file_path set there, None here)."""
    step = {"agent": "context", "tool": "github_metadata", "args": {"target": "no/such/flask/file.py"}}
    with SessionLocal() as session:
        result = dispatch_context_step(step, FLASK_IMPORTS_REPO_ID, session)

    assert result.tool == "github_metadata"
    assert isinstance(result.result, GitHubMetadataResult)
    assert result.result.file_path is None  # unresolvable -> None, distinct from resolved-empty
    assert result.result.issues == [] and result.result.prs == []
