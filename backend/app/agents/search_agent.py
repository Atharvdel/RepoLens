"""Search Agent for the RepoLens agent layer (SDD 9.2).

The second agent, built **standalone before LangGraph** is wired (SDD 15), the
same way :mod:`app.agents.planner` was: a single, independently testable unit
that will later become one node in the graph. Unlike the Planner (which asks the
local LLM to *choose* tools), the Search Agent **executes** them. SDD 9.2 defines
its responsibility narrowly:

    "execute symbol/file/text search tools and return structured, cited results
    (file path + line numbers), never prose summaries of code it hasn't
    retrieved."

So this agent contains **no LLM call at all** -- it is a pure dispatcher. It takes
one plan step (the shape the Planner produces, SDD 9.1: ``{"agent": "search",
"tool": <name>, "args": {...}}``), validates the tool name is one of the three
MVP search tools, *injects* the orchestrator-supplied ``repository_id`` + ``Session``
(neither of which the Planner is ever allowed to put in ``args`` -- the Planner's
prompt tells it so, and :data:`app.agents.planner.KNOWN_TOOLS` / ``OPTIONAL_TOOL_ARGS``
are defined so they never appear), forwards just the tool's optional kwargs
(``kind`` / ``regex``), and returns the tool's structured hits wrapped in a
consistent :class:`SearchStepResult` the Synthesizer (SDD 9.4) will later consume.

Why a dispatcher and not bespoke per-step glue: "isolates all 'find things'
logic so the Synthesizer never has to guess file paths" (SDD 9.2). One place owns
the name -> callable table, so a new tool is one registry line, and a malformed
plan step (unknown tool, non-object ``args``) fails loudly here -- never as a
silent empty result that would read to a downstream agent as "nothing matched."
The same "structured, never free text" + "keep tools/agents small and
independently testable" ethos as the rest of the stack (SDD 10 / SDD 18 risk
row).

Design split (the project's pure-core / live-wrapper convention, mirrored from
:mod:`app.tools.text_search`'s ``find_text`` / ``search_text`` and the indexing
stages' pure / persist):

* :class:`SearchStepResult` -- the consistent return shape (tool name, the args
  actually used, the hits), every field a plain JSON type so
  ``json.dumps(dataclasses.asdict(result))`` works for a ``tool_trace``
  (SDD 11 ``chat_messages.tool_trace``) / the Synthesizer.
* :class:`SearchDispatchError` -- raised on an unknown tool name or a malformed
  step (``step`` not a dict, ``args`` not a dict). NOT raised on an empty result
  -- "no hits" is a valid answer (SDD 10), not a dispatch failure. Subclass of
  :class:`ValueError` so callers that want to treat it as bad input can.
* :data:`SEARCH_TOOLS` -- ``name -> (callable, forwardable-kwarg-set)``. The
  names are the REAL tool function names (``search_symbols`` / ``search_files``
  / ``search_text``), matching :data:`app.agents.planner.KNOWN_TOOLS` so a plan
  the Planner emits is dispatchable as-is (the Planner's docstring notes it
  writes the real names over SDD 9.1's noun-first example).
* :func:`build_search_step_result` -- **pure wrt DB / ripgrep**: validate the
  step, inject ``repository_id`` + ``session``, call the tool, build the result.
  No I/O of its own; all I/O lives in the tools it calls. The unit-testable
  shape (the malformed-tool test pins this without a DB or ripgrep).
* :func:`dispatch_search_step` -- thin convenience that just calls
  :func:`build_search_step_result`; the obvious entrypoint for the future graph
  node (one line that's easy to swap for a ``Runnable`` / ``@node`` decorator
  later without touching the logic).

Kwarg forwarding is **whitelisted**, not splatted: only ``kind`` (for
``search_symbols``) and ``regex`` (for ``search_text``) pass through, because
those are exactly the optional args the Planner is permitted to emit
(:data:`app.agents.planner.OPTIONAL_TOOL_ARGS`). The tools' *other* kwargs
(``search_text``'s ``cap`` / ``ripgrep_bin``, ``search_symbols``'s positional
``kind``) are orchestrator concerns, so a stray ``cap`` or ``repository_id`` a
misbehaving plan happens to include is **dropped on the floor**, not forwarded:
 forwarding an unknown kwarg to a tool that doesn't accept it would raise
``TypeError`` mid-dispatch and turn a *planner* mistake into a tool-layer crash.
Dropping is the "fail loud at *our* boundary, not inside the tool" posture. A
``regex`` value that arrives as the string ``"true"`` / ``"false"`` (a common
weak-model JSON habit) is coerced to bool before forwarding.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.agents.planner import OPTIONAL_TOOL_ARGS
from app.models.repository import Repository
from app.tools.file_search import FileResult, search_files
from app.tools.symbol_search import SymbolResult, search_symbols
from app.tools.text_search import TextHit, search_text

# The three MVP search tools (SDD 10), keyed by the real function name -- in
# lock-step with app.agents.planner.KNOWN_TOOLS so a plan the Planner emits is
# dispatchable unchanged. Each value is the callable plus the set of kwargs the
# Planner is permitted to put in a step's ``args`` (a mirror of
# OPTIONAL_TOOL_ARGS); everything else is dropped (see module docstring). The
# tool functions are imported directly (not name-mapped through __import__) so a
# reader can follow the symbol, IDE jump-to-def works, and an unused-import
# linter catches a typo -- the same posture as the sibling agent / tool modules.
SEARCH_TOOLS: dict[str, tuple[Callable[..., Any], frozenset[str]]] = {
    "search_symbols": (search_symbols, frozenset(OPTIONAL_TOOL_ARGS["search_symbols"])),
    "search_files": (search_files, frozenset(OPTIONAL_TOOL_ARGS["search_files"])),
    "search_text": (search_text, frozenset(OPTIONAL_TOOL_ARGS["search_text"])),
}

KEYWORD_ALIASES: dict[str, str] = {
    "authentication": "auth",
    "authorization": "auth",
    "database": "db",
    "configuration": "config",
    "repository": "repo",
    "middleware": "middle",
}


# ─── result types ────────────────────────────────────────────────────────────


@dataclass
class SearchStepResult:
    """The consistent return shape the Synthesizer (SDD 9.4) consumes for one
    executed plan step.

    Every field is a plain JSON type: ``tool`` is the tool name that ran,
    ``args`` is the args actually used (``repository_id`` + forwarded kwargs),
    and ``hits`` is the tool's structured result list. The SDD 10 tool result
    dataclasses (:class:`SymbolResult` / :class:`FileResult` / :class:`TextHit`)
    are themselves JSON-serializable (plain fields), so the whole result is
    ``json.dumps(dataclasses.asdict(result))``-able -- straight into a
    ``chat_messages.tool_trace`` row (SDD 11) or the Synthesizer's context.

    ``hits`` is typed ``list[Any]`` because the three tools return three
    different dataclasses; the per-tool type narrowness is pinned by the live
    dispatch tests (each asserts the expected concrete type). An empty ``hits``
    is a valid result (SDD 10: an empty result is a valid answer) and is NOT a
    dispatch failure.
    """

    tool: str
    repository_id: int
    args: dict[str, Any] = field(default_factory=dict)
    hits: list[Any] = field(default_factory=list)
    matched_file_snippets: list[dict[str, str]] = field(default_factory=list)


# ─── errors ──────────────────────────────────────────────────────────────────


class SearchDispatchError(ValueError):
    """A plan step could not be dispatched. Subclass of :class:`ValueError` so
    callers that want to treat it as generic bad input can; a distinct class so
    the future graph can catch it and route to re-planning (SDD 15: "if Search
    returns zero hits, replan" -- but a *malformed* step is not "zero hits", so
    it surfaces here instead of being swallowed into an empty result).

    Raised when:

    * ``step`` is not a dict (the Planner is documented to emit an object).
    * ``tool`` is missing / not a string / not one of :data:`SEARCH_TOOLS`.
    * ``args`` is present but not a dict.
    * ``args`` (after defaulting absent to ``{}``) has no ``query`` key.

    NOT raised for an empty hit list (that's a valid search answer, SDD 10) or
    for an empty-string query (the tools return ``[]``). A missing ``args`` key
    alone is lenient (defaults to ``{}``) -- but a step that then has no
    ``query`` raises per the last point above; the Planner's validator
    (:func:`app.agents.planner._validate_plan`) should already have caught that.
    """


# ─── pure dispatch (the unit-testable shape) ─────────────────────────────────


def _coerce_regex(value: Any) -> bool:
    """Coerce a ``regex`` value from a plan's ``args`` to bool. The Planner's
    prompt says ``regex`` is a boolean, but a weak model sometimes emits
    ``"true"`` / ``"false"`` (strings) or a truthy non-bool; coerce so the tool
    gets a real bool (``search_text``'s ``regex: bool``). Mirrors nothing in the
    sibling tools; kept as a named helper so the coercion is a visible, tested
    piece of logic rather than a buried inline expression."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _forward_kwargs(
    args: dict[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    """Build the kwarg dict to forward to the tool. **Whitelist, not splat**:
    only keys in ``allowed`` (the Planner's ``OPTIONAL_TOOL_ARGS`` for this tool)
    pass through; everything else (a stray ``cap``, ``repository_id``,
    ``session`` from a misbehaving plan) is dropped, so a planner mistake never
    reaches the tool as an unexpected ``TypeError``-triggering kwarg. Iterates
    over ``allowed`` (not ``args``) so a plan can't smuggle in an unforwarded key
    by naming it -- the tool's own contract, not the model's output, decides
    what crosses. ``regex`` is bool-coerced (the Planner's prompt says boolean,
    but a weak model sometimes emits ``"true"``); every other allowed kwarg
    (``kind``) is forwarded as-is, since ``search_symbols`` tolerates a string."""
    fwd: dict[str, Any] = {}
    for key in allowed:
        if key in args:
            fwd[key] = _coerce_regex(args[key]) if key == "regex" else args[key]
    return fwd


def _extract_file_snippets(
    hits: list[Any],
    repository_id: int,
    session: Session,
    max_files: int = 3,
    max_lines: int = 45,
) -> list[dict[str, str]]:
    """Extract initial source code snippets from distinct files matched in hits."""
    if not hits:
        return []

    paths: list[str] = []
    for h in hits:
        p = getattr(h, "file", None) or getattr(h, "path", None)
        if p and p not in paths:
            # Skip package-lock.json or massive bundles
            if not any(ign in p.lower() for ign in ["package-lock", "yarn.lock", ".min.", "chunk"]):
                paths.append(p)

    if not paths:
        return []

    from app.tools._path_resolve import resolve_repo_root
    repo_root = resolve_repo_root(repository_id, session)
    if not repo_root or not os.path.exists(repo_root):
        return []

    snippets: list[dict[str, str]] = []
    for p in paths[:max_files]:
        # Try direct path
        full_p = os.path.join(repo_root, p)
        if not os.path.exists(full_p):
            parts = p.split("/", 1)
            if len(parts) > 1:
                alt_p = os.path.join(repo_root, parts[1])
                if os.path.exists(alt_p):
                    full_p = alt_p

        if os.path.exists(full_p) and os.path.isfile(full_p):
            try:
                with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                snippet = "".join(lines[:max_lines])
                if snippet.strip():
                    snippets.append({"path": p, "snippet": snippet})
            except Exception:
                pass

    return snippets


def build_search_step_result(
    step: dict[str, Any],
    repository_id: int,
    session: Session,
) -> SearchStepResult:
    """Dispatch one plan step to its tool and return a :class:`SearchStepResult`.

    *Injects* ``repository_id`` and ``session`` (orchestrator-supplied -- the
    Planner is never allowed to put them in ``args``; SDD 9.1's example step
    carries only the user-relevant arg). Calls the tool, wraps the hits.

    Structural checks (raise :class:`SearchDispatchError`, never a silent
    ``[]``):

    * ``step`` must be a dict.
    * ``step["tool"]`` must be a string in :data:`SEARCH_TOOLS`.
    * ``step["args"]`` if present must be a dict; if absent, defaults to ``{}``.

    A step whose ``args`` is missing the tool's required ``query`` (every tool
    needs one -- :data:`app.agents.planner.KNOWN_TOOLS`) also raises
    :class:`SearchDispatchError`. The Planner's structural validator
    (:func:`app.agents.planner._validate_plan`) should have caught that first,
    but the dispatcher does not trust the plan -- forwarding ``None`` down would
    crash the tool with an unhelpful ``AttributeError`` (every tool does
    ``query.strip()``), so a missing ``query`` is surfaced here as a clear
    dispatch error rather than as a tool-layer crash. (An *empty-string* query is
    different: the tools return ``[]`` on it, which is a valid empty
    :class:`SearchStepResult`, not a dispatch failure -- "no hits" is a valid
    answer, SDD 10.)
    """
    if not isinstance(step, dict):
        raise SearchDispatchError(
            f"plan step must be an object, got {type(step).__name__}: {step!r}"
        )

    tool_name = step.get("tool")
    if not isinstance(tool_name, str):
        raise SearchDispatchError(
            f"plan step 'tool' must be a string, got {type(tool_name).__name__}: "
            f"{step!r}"
        )
    if tool_name not in SEARCH_TOOLS:
        raise SearchDispatchError(
            f"unknown search tool {tool_name!r}; expected one of "
            f"{sorted(SEARCH_TOOLS)}"
        )

    args = step.get("args", {})
    if not isinstance(args, dict):
        raise SearchDispatchError(
            f"plan step 'args' must be an object, got {type(args).__name__}: "
            f"{args!r} (in step {step!r})"
        )

    tool_fn, allowed = SEARCH_TOOLS[tool_name]
    fwd = _forward_kwargs(args, allowed)

    # The tools take (repository_id, query, session) positionally; the orchestrator
    # supplies repository_id + session, the plan supplies `query` (required) +
    # any forwarded kwargs. A missing `query` -> KeyError here, which we surface
    # as SearchDispatchError so a malformed step never raises a bare KeyError
    # into the graph. (The Planner's validator should have caught a missing
    # query already; this is the dispatcher's own belt-and-suspenders.)
    try:
        query = args["query"]
    except KeyError as exc:
        raise SearchDispatchError(
            f"plan step for {tool_name!r} is missing required arg 'query': "
            f"{step!r}"
        ) from exc

    raw_hits = tool_fn(repository_id, query, session, **fwd)
    hits = list(raw_hits)

    # Keyword alias & tokenized fallback if zero hits
    if not hits and isinstance(query, str):
        q_clean = query.strip().lower()
        alias = KEYWORD_ALIASES.get(q_clean)
        if alias and alias != q_clean:
            alias_hits = tool_fn(repository_id, alias, session, **fwd)
            if alias_hits:
                hits = list(alias_hits)

        # Multi-word compound query fallback (e.g. "authentication session authorization" -> ["auth", "session"])
        if not hits and " " in q_clean:
            words = [w.strip() for w in q_clean.split() if len(w.strip()) >= 3]
            terms = list(dict.fromkeys([KEYWORD_ALIASES.get(w, w) for w in words]))
            for term in terms:
                term_hits = tool_fn(repository_id, term, session, **fwd)
                if term_hits:
                    hits.extend(term_hits)
                    if len(hits) >= 25:
                        break

    # Extract source snippets for matched files
    snippets = _extract_file_snippets(hits, repository_id, session)

    return SearchStepResult(
        tool=tool_name,
        repository_id=repository_id,
        args={"query": query, **fwd},
        hits=list(hits),
        matched_file_snippets=snippets,
    )


def dispatch_search_step(
    step: dict[str, Any],
    repository_id: int,
    session: Session,
) -> SearchStepResult:
    """Execute one plan step against the repo and return its structured hits.

    Thin convenience over :func:`build_search_step_result` -- the obvious
    entrypoint the future LangGraph node (SDD 15) will wrap: one line that's easy
    to decorate (``@node`` / make a ``Runnable``) without touching the dispatch
    logic. The split keeps "build the result object" (pure, testable shape)
    separable from "this is the graph's call signature". Pass the plan step plus
    the orchestrator-supplied ``repository_id`` and an open ``Session`` (SDD 15
    opens one session per query); the dispatcher owns no transaction -- the
    tools are pure reads (SDD 10 / SDD 7 step 9), so the session is read through
    and not committed.
    """
    return build_search_step_result(step, repository_id, session)


# Re-export the tool result types so a caller importing from this module doesn't
# need to know which tool module each lives in (the Synthesizer / a tool_trace
# serializer iterates ``hits`` and may want the types for isinstance checks).
__all__ = [
    "FileResult",
    "SEARCH_TOOLS",
    "SearchDispatchError",
    "SearchStepResult",
    "SymbolResult",
    "TextHit",
    "build_search_step_result",
    "dispatch_search_step",
]
