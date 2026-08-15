"""Context Agent for the RepoLens agent layer (SDD 9.3).

The third of the four agents, built **standalone before LangGraph** is wired
(SDD 15), the same way :mod:`app.agents.search_agent` was: a single,
independently testable unit that will later become one node in the graph. The
merged Architecture + GitHub Context agent (SDD 9.3: "answer structural/relational
questions and pull historical/GitHub metadata for a given file or symbol").

Like the Search Agent, this contains **no LLM call at all** -- it is a pure
dispatcher. SDD 9.3's inputs ("a file or symbol identifier from the Search
Agent's output") arrive as a plan step the **Planner** chose (SDD 9.1's example
writes ``{"agent": "context", "tool": "file_history", "args": {"file": "..."}}``
-- the Planner picks the *context tool*; this agent *executes* it). So the
Context Agent validates the step, injects the orchestrator-supplied
``repository_id`` + ``Session``, forwards only the whitelisted kwargs, and
returns the tool's structured result wrapped in a consistent
:class:`ContextStepResult` the Synthesizer consumes -- the structural twin of
:func:`app.agents.search_agent.build_search_step_result`.

Why a dispatcher and not bespoke per-step glue (the same rationale the Search
Agent docstring gives, re-stated for the context tools): one place owns the
name -> callable table, so a malformed plan step (unknown context tool,
non-object step/args) fails **loudly** here -- never as a silent empty result a
downstream agent would misread as "nothing matched." The "structured, never free
text" + "keep tools/agents small and independently testable" ethos (SDD 10 /
SDD 18 risk row) is what shaped the Search Agent, and the same shape fits here.

Design split (the project's pure-core / live-wrapper convention, mirrored from
:mod:`app.agents.search_agent`'s ``SearchStepResult`` / ``SearchDispatchError`` /
``SEARCH_TOOLS`` / ``build_search_step_result``):

* :class:`ContextStepResult` -- the consistent return shape (tool name, the args
  actually used, the result), every field a plain JSON type so
  ``json.dumps(dataclasses.asdict(result))`` works for a ``tool_trace``
  (SDD 11 ``chat_messages.tool_trace``) / the Synthesizer.
* :class:`ContextDispatchError` -- raised on an unknown tool name or a malformed
  step (``step`` not a dict, ``tool`` not a string, ``args`` not a dict, a
  required arg missing). NOT raised on an empty result -- an unresolvable
  ``target`` is a valid empty answer (SDD 10), returned by the tools, not a
  dispatch failure. Subclass of :class:`ValueError` (matches the Search Agent).
* :data:`CONTEXT_TOOLS` -- ``name -> (callable, required-arg-keys, optional-
  arg-keys)``. Unlike the search tools (all share ``query``), the context tools
  have **different required args**: the three identifier-keyed tools
  (``dependency_graph`` / ``file_history`` / ``github_metadata``) require
  ``target``; ``architecture`` requires *none* (whole-repo is the default when
  ``target`` is absent, though it accepts one for a focused view). Modeling the
  per-tool required set in the registry keeps the dispatcher generic and
  honest -- a step for ``architecture`` with no ``target`` is *not* a dispatch
  error (whole-repo), while a ``file_history`` step with no ``target`` is.
* :func:`build_context_step_result` -- **pure wrt DB**: validate the step, inject
  ``repository_id`` + ``session``, forward whitelisted kwargs, call the tool,
  build the result. No I/O of its own; all I/O lives in the tools.
* :func:`dispatch_context_step` -- thin convenience; the obvious entrypoint for
  the future LangGraph node (SDD 15).

Kwarg forwarding is **whitelisted, not splatted** (the Search Agent's posture):
only the optional kwargs each tool documents are forwarded; a stray ``cap`` /
``repository_id`` / ``session`` from a misbehaving plan is dropped, never let
through to crash a tool with ``TypeError``. ``depth`` / ``top_k`` / ``radius`` /
``recent_cap`` are int-coerced at the tool boundaries already, so the dispatcher
forwards them as-is (the tools own their coercion; the dispatcher owns only the
``required-keys-present`` and ``unknown-kwarg-drop`` contracts -- a clean split
of concerns the Search Agent established).

**Planner-menu status (when built): not yet wired.** This delivery builds the
Context Agent + its four tools standalone; it owns its *own* required/optional
arg sets (:data:`CONTEXT_REQUIRED_ARGS` / :data:`CONTEXT_OPTIONAL_ARGS`), *not*
the Planner's :data:`app.agents.planner.KNOWN_TOOLS` / ``OPTIONAL_TOOL_ARGS``, so
the verified Planner is untouched. Teaching the Planner to *emit*
``agent: "context"`` steps is a discrete follow-up (it needs its own Ollama
re-verification: a new tool menu in :data:`app.agents.planner.DEFAULT_TOOLS_DESCRIPTION`,
new entries in the Planner's ``KNOWN_TOOLS`` / ``OPTIONAL_TOOL_ARGS``, and a
re-run of the 5-question demo). Until that lands, the Context Agent is exercised
by hand-constructed plan steps in the demo / tests -- the same way the Search
Agent was proven before the Planner had a context menu. The two registries
converge in that follow-up; today they're documented siblings, not mirrored
(yet), so the dispatcher's registry-drift test asserts internal consistency
only (the SDD 9.1 example's ``"file"`` arg name vs the tool's ``"target"`` is
also reconciled *then* -- see the note on :data:`CONTEXT_REQUIRED_ARGS`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.tools.architecture import ArchitectureResult, query_architecture
from app.tools.dependency_graph import DependencyGraphResult, query_dependency_graph
from app.tools.file_history import FileHistoryResult, query_file_history
from app.tools.github_metadata import (
    GitHubMetadataResult,
    query_github_metadata,
)

# ─── the four context tools (SDD 9.3 / SDD 10), keyed by the real function name ─
# The names are the REAL tool function names (query_*), matching how the Search
# Agent keys SEARCH_TOOLS by the real search function names -- so a plan step
# the Planner emits (once its menu is wired) is dispatchable as-is. Each value
# is the callable plus its required + optional arg-key sets: the *required* set
# is the per-tool contract the dispatcher enforces (see module docstring for
# why the context tools have heterogeneous required args, unlike the search
# tools' shared ``query``).
#
# NOTE on the arg name vs SDD 9.1: the SDD 9.1 example writes
# ``{"args": {"file": "..."}}`` for a ``file_history`` step; the tool functions
# here take ``target`` (a file path OR module name -- the architecture +
# dependency_graph + github_metadata tools all accept a module identifier too,
# so ``target`` is the honest superset). When the Planner's context menu is
# wired, the Planner emits ``target`` (matching the tools) OR the dispatcher
# accepts ``file`` -> ``target`` as an alias -- whichever the Planner-verify
# step settles. Today the dispatcher uses ``target``; a concrete ``"file"``
# alias is deferred to that follow-up (documented here so it isn't lost).
CONTEXT_TOOLS: dict[str, tuple[Callable[..., Any], frozenset[str], frozenset[str]]] = {
    # Architecture Query: `target` optional (whole-repo when absent); `top_k` /
    # `radius` optional knobs the Planner may tune.
    "architecture": (query_architecture, frozenset(), frozenset({"target", "top_k", "radius"})),
    # Dependency Graph Query: `target` required; `depth` optional.
    "dependency_graph": (query_dependency_graph, frozenset({"target"}), frozenset({"depth"})),
    # History Search: `target` required (a file path, module name, or "repository_root"); `recent_cap` optional.
    "file_history": (query_file_history, frozenset({"target"}), frozenset({"recent_cap"})),
    # GitHub Metadata Loader: `target` optional (whole-repo when absent); no other knobs.
    "github_metadata": (query_github_metadata, frozenset(), frozenset({"target"})),
}

# Per-tool required / optional args exposed as plain dicts for parity with
# :data:`app.agents.planner.KNOWN_TOOLS` / ``OPTIONAL_TOOL_ARGS`` (and for the
# future Planner-menu wiring to import, so the two layers converge rather than
# drift). Derived from :data:`CONTEXT_TOOLS` so there's a single source of
# truth -- a tool added to the registry appears in both maps automatically.
CONTEXT_REQUIRED_ARGS: dict[str, frozenset[str]] = {
    name: required for name, (_fn, required, _opt) in CONTEXT_TOOLS.items()
}
CONTEXT_OPTIONAL_ARGS: dict[str, frozenset[str]] = {
    name: optional for name, (_fn, _req, optional) in CONTEXT_TOOLS.items()
}


# ─── result types ────────────────────────────────────────────────────────────


@dataclass
class ContextStepResult:
    """The consistent return shape the Synthesizer (SDD 9.4) consumes for one
    executed context plan step -- the structural twin of
    :class:`app.agents.search_agent.SearchStepResult`.

    Every field is a plain JSON type: ``tool`` is the tool name that ran,
    ``args`` is the args actually used (``repository_id`` + forwarded kwargs),
    and ``result`` is the tool's structured object
    (:class:`ArchitectureResult` / :class:`DependencyGraphResult` /
    :class:`FileHistoryResult` / :class:`GitHubMetadataResult`). The SDD 10 tool
    result dataclasses are themselves JSON-serializable (plain fields /
    dataclasses of plain fields), so ``json.dumps(dataclasses.asdict(result))``
    works -- straight into a ``chat_messages.tool_trace`` row (SDD 11) or the
    Synthesizer's context.

    ``result`` is typed ``Any`` because the four tools return four different
    dataclasses; the per-tool type narrowness is pinned by the live dispatch
    tests (each asserts the expected concrete type). An empty result
    (unresolvable ``target``) is a valid :class:`ContextStepResult` with an empty
    tool result -- NOT a dispatch failure (SDD 10: an empty result is a valid
    answer)."""

    tool: str
    repository_id: int
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None


# ─── errors ──────────────────────────────────────────────────────────────────


class ContextDispatchError(ValueError):
    """A context plan step could not be dispatched. Subclass of
    :class:`ValueError` (so a caller that treats it as generic bad input still
    catches it; :class:`app.agents.search_agent.SearchDispatchError` is the twin).
    A distinct class so the future graph can catch context-dispatch failures and
    route to re-planning (SDD 15) -- a *malformed* step is not "empty result",
    so it surfaces here instead of being swallowed into an empty ContextStepResult.

    Raised when:
    * ``step`` is not a dict (the Planner is documented to emit an object).
    * ``tool`` is missing / not a string / not one of :data:`CONTEXT_TOOLS`.
    * ``args`` is present but not a dict.
    * ``args`` (after defaulting absent to ``{}``) is missing a **required** key
      for the tool (e.g. ``file_history`` with no ``target``).

    NOT raised for an empty result (an unresolvable ``target`` is a valid
    empty tool answer, SDD 10) or for an absent-but-optional arg (``architecture``
    with no ``target`` is the whole-repo query, a valid call). A missing
    ``args`` key alone is lenient (defaults to ``{}``) -- but a tool that then
    requires an arg raises per the last point; the Planner's validator
    (:func:`app.agents.planner._validate_plan`) should already have caught that
    once its context menu is wired."""


# ─── pure dispatch (the unit-testable shape) ─────────────────────────────────


def _forward_kwargs(
    args: dict[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    """Build the kwarg dict to forward to the tool. **Whitelist, not splat**
    (the twin of :func:`app.agents.search_agent._forward_kwargs`): only keys in
    ``allowed`` (the tool's optional-arg set) pass through; everything else (a
    stray ``recent_cap`` for a tool that doesn't take it, ``repository_id``,
    ``session`` from a misbehaving plan) is dropped, so a planner mistake never
    reaches the tool as an unexpected ``TypeError``-triggering kwarg. Iterates
    over ``allowed`` (not ``args``) so a plan can't smuggle in an unforwarded
    key by naming it -- the tool's own contract, not the model's output, decides
    what crosses.

    No bool/string coercion is needed here (unlike the Search Agent's
    ``regex`` coercion): the context tools' optional knobs are *numeric* --
    the tools themselves int-coerce + clamp (``depth`` / ``top_k`` / ``radius`` /
    ``recent_cap``), so the dispatcher forwards values as-is and lets the tool
    own its coercion. Splitting concerns the way the Search Agent split
    "dispatcher validates shape, tool validates value"."""
    fwd: dict[str, Any] = {}
    for key in allowed:
        if key in args:
            fwd[key] = args[key]
    return fwd


def build_context_step_result(
    step: dict[str, Any],
    repository_id: int,
    session: Session,
) -> ContextStepResult:
    """Dispatch one context plan step to its tool and return a
    :class:`ContextStepResult`.

    *Injects* ``repository_id`` and ``session`` (orchestrator-supplied -- the
    Planner is never allowed to put them in ``args``; SDD 9.1's example step
    carries only the user-relevant arg). Calls the tool, wraps the result.

    Structural checks (raise :class:`ContextDispatchError`, never a silent empty
    result):
    * ``step`` must be a dict.
    * ``step["tool"]`` must be a string in :data:`CONTEXT_TOOLS`.
    * ``step["args"]`` if present must be a dict; if absent, defaults to ``{}``.

    A step whose ``args`` is missing a tool's **required** key (the per-tool
    required set in :data:`CONTEXT_TOOLS`) raises :class:`ContextDispatchError`.
    Tools whose required set is empty (``architecture`` / ``github_metadata``)
    accept a no-``target`` step as the whole-repo query. The dispatcher does not
    trust the plan -- forwarding ``None`` for a required ``target`` down would
    make the resolver return ``None`` and produce an empty tool result that a
    downstream agent would misread as "Planner meant a whole-repo query" when it
    really meant "I forgot the arg." Surfacing a missing required key here, as
    a clear dispatch error, is the same belt-and-suspenders the Search Agent
    applies to a missing ``query``.
    """
    if not isinstance(step, dict):
        raise ContextDispatchError(
            f"plan step must be an object, got {type(step).__name__}: {step!r}"
        )

    tool_name = step.get("tool")
    if not isinstance(tool_name, str):
        raise ContextDispatchError(
            f"plan step 'tool' must be a string, got {type(tool_name).__name__}: "
            f"{step!r}"
        )
    if tool_name not in CONTEXT_TOOLS:
        raise ContextDispatchError(
            f"unknown context tool {tool_name!r}; expected one of "
            f"{sorted(CONTEXT_TOOLS)}"
        )

    args = step.get("args", {})
    if not isinstance(args, dict):
        raise ContextDispatchError(
            f"plan step 'args' must be an object, got {type(args).__name__}: "
            f"{args!r} (in step {step!r})"
        )

    tool_fn, required, allowed = CONTEXT_TOOLS[tool_name]

    # Required-arg presence: enforce the per-tool required set. A missing
    # required key is a dispatch error (not an empty result) -- see the docstring
    # rationale. An absent-but-optional key is fine (not forwarded). Values are
    # unchecked for emptiness: an empty-string ``target`` is forwarded as-is and
    # the tool's resolver returns ``None`` -> a valid *empty* tool result (SDD
    # 10), distinct from a *missing* key which is the dispatch error here.
    missing = [k for k in sorted(required) if k not in args]
    if missing:
        raise ContextDispatchError(
            f"plan step for {tool_name!r} is missing required arg(s) "
            f"{missing}: {step!r}"
        )

    # ``target`` is the tools' shared SECOND positional arg (``query``-equivalent
    # for the context tools): all four take ``(repository_id, target, session,
    # *, <knobs>)``. So -- mirroring how the Search Agent extracts ``query`` and
    # passes it positionally rather than letting it collide as a kwarg -- we
    # pull ``target`` out of ``args`` here (present when required OR when the
    # Planner chose to include it for an optional-target tool) and pass it as
    # the second positional, forwarding only the *knobs* as kwargs. ``None``
    # (absent for an optional-target tool) is the whole-repo query the tools
    # accept -- ``query_architecture`` / ``query_github_metadata`` both coerce a
    # falsy ``target`` to whole-repo.
    target = args.get("target")
    knob_allowed = allowed - {"target"}  # target goes positional, not as a kwarg
    fwd = _forward_kwargs(args, knob_allowed)
    result = tool_fn(repository_id, target, session, **fwd)

    # The args actually used: the target (if the tool took one / the plan gave
    # one) plus the forwarded knobs. Matches SearchStepResult.args (the
    # user-relevant args, repository_id in the wrapper not repeated here).
    used: dict[str, Any] = {}
    if "target" in args:
        used["target"] = target
    used.update(fwd)
    return ContextStepResult(
        tool=tool_name,
        repository_id=repository_id,
        args=used,
        result=result,
    )


def dispatch_context_step(
    step: dict[str, Any],
    repository_id: int,
    session: Session,
) -> ContextStepResult:
    """Execute one context plan step against the repo and return its structured
    result. Thin convenience over :func:`build_context_step_result` -- the
    obvious entrypoint the future LangGraph node (SDD 15) will wrap (the twin
    of :func:`app.agents.search_agent.dispatch_search_step`). Pass the plan step
    plus the orchestrator-supplied ``repository_id`` and an open ``Session``
    (SDD 15 opens one session per query); the dispatcher owns no transaction --
    the tools are pure reads (SDD 10 / SDD 7 step 9)."""
    return build_context_step_result(step, repository_id, session)


# Re-export the tool result types so a caller importing from this module doesn't
# need to know which tool module each lives in (the Synthesizer / a tool_trace
# serializer iterates results and may want the types for isinstance checks) --
# the twin of the Search Agent's re-exports.
__all__ = [
    "ArchitectureResult",
    "CONTEXT_OPTIONAL_ARGS",
    "CONTEXT_REQUIRED_ARGS",
    "CONTEXT_TOOLS",
    "ContextDispatchError",
    "ContextStepResult",
    "DependencyGraphResult",
    "FileHistoryResult",
    "GitHubMetadataResult",
    "build_context_step_result",
    "dispatch_context_step",
]
