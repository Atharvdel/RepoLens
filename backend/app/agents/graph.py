"""LangGraph wiring for the RepoLens agent layer (SDD 15).

The connective tissue the four standalone agents (SDD 9.1-9.4) needed to become a
pipeline. Each agent was built + verified standalone -- the Planner (9.1), the
Search Agent (9.2), the Context Agent (9.3), and the Synthesizer (9.4) -- so this
module owns **no agent logic of its own**: it wraps each as a LangGraph node and
wires the edges SDD 15 prescribes. The *narrowness* the user asked for is the point
-- the graph matches how narrow the individual agents already are; it routes and
loops, it does not think.

SDD 15 flow: Planner -> (Search and/or Context, based on what the plan actually
contains -- skip a branch entirely if the plan has no steps for that agent) ->
Synthesizer, with one replan loop ("retry once if Search returns zero hits"),
capped at 1 retry via ``replans_used`` to bound worst-case latency. The cap is the
deliberate latency bound SDD 15 calls out: without it, a weak local model could
plausibly loop indefinitely on an unanswerable query.

State shape (SDD 15): a single typed object carried node-to-node -- the six control
fields ``{query, plan, search_results, context_results, replans_used,
final_answer}`` are reproduced **verbatim** from SDD 15. The extra fields
(``had_search`` / ``had_context`` / ``total_hits`` / ``cited_file_paths`` /
``hit_file_paths`` / ``synth_wall_time_s`` / ``node_trace`` / ``errors``) are
**observability** the routing logic and the demo need, clearly labeled as such --
they are NOT part of the SDD 15 control contract and a future caller can ignore
them. ``node_trace`` + ``errors`` are the only **accumulating** channels (the
ordered node-firing trace + the per-stage friction log); everything else is
last-write-wins (a planner pass replaces the plan, a search pass replaces the
search results + total_hits -- so a replan's second pass cleanly *overwrites* the
first pass's evidence rather than appending to stale zero-hit results).

Design split (the project's pure-core / live-wrapper convention, applied to the
graph layer):

* :data:`RepoLensState` -- the TypedDict LangGraph carries between nodes. Control
  fields verbatim from SDD 15; observability fields clearly separated.
* The four **pure module-level routing functions** -- :func:`_needs_replan`,
  :func:`_replan_target`, :func:`_route_after_planner`, :func:`_route_after_search`
  -- decide the next node from state alone (no I/O, no closures, no infra). They
  are the unit-testable routing matrix: ``tests/test_graph.py`` pins them across
  the replans_used / had_search / total_hits / had_context Cartesian product
  without any LLM or DB.
* :func:`build_graph` -- the live wrapper: closes over the orchestrator-supplied
  ``repository_id`` + ``Session`` + Ollama ``host``/``model`` (the same injected
  infra the two dispatchers take), defines the four node closures (each calls the
  matching agent), wires the conditional edges around the pure routing functions,
  and returns the compiled graph. Closures -- not ``context_schema`` runtime
  config -- keep the wiring simple, matching how narrow each agent already is.
* :func:`run_query` -- the single top-level entrypoint a caller (the demo now;
  the future FastAPI ``/chat`` endpoint) calls without knowing anything about
  LangGraph internals. Opens nothing itself (the caller owns the ``Session`` --
  SDD 15 "one session per query"; the demo opens ``SessionLocal()`` and passes
  it, the same pattern the chain demos + dispatch tests use). Returns a
  :class:`GraphResult` (final answer + the full state) so the caller never touches
  a raw LangGraph state dict.

Routing topology (linear DAG with one back-edge for the replan loop):

    START -> planner -> (cond _route_after_planner) ->
        search | context | synthesizer
    search -> (cond _route_after_search) ->
        context | (cond _replan_target) -> planner | synthesizer
    context -> (cond _replan_target) -> planner | synthesizer
    synthesizer -> END

Search and Context run **sequentially** (search first, then context if the plan
has context steps) -- *not* in parallel. SDD 15's parallelism parenthetical
("run in parallel where the plan allows; sequential where one depends on the
other's output") explicitly permits sequential; for the MVP, sequential is the
simpler, faithful choice (no fan-out barrier to wire, no parallel double-fire
on the convergence node to reason about). The graph skips a branch entirely when
the plan has no steps for that agent (``_route_after_planner`` falls through to
``synthesizer`` when neither search nor context steps are present; the empty-plan
/ planner-failure path is the synthesizer saying plainly that nothing matched).

The replan loop is one back-edge: ``search`` (when no context follows) OR
``context`` (always terminal when present) routes through :func:`_replan_target`,
which returns ``planner`` iff a search actually ran, returned zero hits, and no
replan has been used yet (``replans_used == 0``); otherwise ``synthesizer``. The
``replans_used`` bump happens **in the planner node on re-entry** (detected by
``state["plan"] is not None`` -- the first call sees ``plan=None`` from the
seeded input, a replan re-entry sees the prior pass's plan); so the second
``_replan_target`` evaluation sees ``replans_used == 1`` and routes to
``synthesizer``, bounding the loop at one retry. The bump-on-re-entry is sound
because every replan re-entry is gated on ``had_search`` (which requires the
prior plan to have been a real dict with search steps), so ``plan`` is guaranteed
non-None on every re-entry.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agents.context_agent import (
    ContextDispatchError,
    ContextStepResult,
    build_context_step_result,
)
from app.agents.planner import OLLAMA_HOST, OLLAMA_MODEL, OllamaError, plan
from app.agents.search_agent import (
    SearchDispatchError,
    SearchStepResult,
    build_search_step_result,
)
from app.agents.synthesizer import SynthesizerResult, synthesize

# Re-exports so a caller importing from the graph module doesn't need to know
# which agent module each step-result / error lives in (the future ``/chat``
# endpoint iterates ``GraphResult.search_results`` / ``context_results`` and may
# want the types for isinstance checks) -- the twin of the agents' own re-exports.
__all__ = [
    "ContextStepResult",
    "GraphResult",
    "RepoLensState",
    "SearchStepResult",
    "build_graph",
    "run_query",
]


# ─── state (SDD 15) ──────────────────────────────────────────────────────────


class RepoLensState(TypedDict):
    """The typed object LangGraph carries node-to-node (SDD 15).

    The first six fields are the SDD 15 control contract, reproduced verbatim
    (``{query, plan, search_results, context_results, replans_used,
    final_answer}``). The remaining fields are **observability** the routing
    logic and the demo need; they are clearly not part of the SDD 15 control
    contract and a caller that only wants the answer can ignore them.

    Reducer semantics: every field is last-write-wins **except** ``node_trace``
    and ``errors``, which are ``Annotated[list[str], operator.add]`` so each
    node's append accumulates across nodes (and across replan passes) into the
    ordered firing trace / friction log. Last-write-wins for ``plan`` /
    ``search_results`` / ``context_results`` / ``total_hits`` means a replan's
    second pass cleanly **overwrites** the first pass's evidence -- a planner
    pass replaces the plan with the new one, a search pass replaces
    ``search_results`` + ``total_hits`` with the new pass's hits -- so stale
    zero-hit results from a failed first pass never pollute the second pass's
    evidence or the synthesizer's grounding.
    """

    # ─── SDD 15 control fields (verbatim) ───────────────────────────────────
    query: str
    plan: dict | None
    search_results: list[SearchStepResult]
    context_results: list[ContextStepResult]
    replans_used: int
    final_answer: str | None
    # ─── observability (NOT SDD 15 control; routing + demo use only) ────────
    had_search: bool
    had_context: bool
    total_hits: int
    cited_file_paths: list[str]
    hit_file_paths: list[str]
    synth_wall_time_s: float
    # accumulators (operator.add -- append across nodes + replan passes)
    node_trace: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


# ─── pure routing (unit-testable; no I/O, no closures, no infra) ─────────────


def _needs_replan(had_search: bool, total_hits: int, replans_used: int) -> bool:
    """Predicate: is the single §15 replan ("retry once if Search returns zero
    hits") warranted? True iff a search actually ran (``had_search`` -- gates
    out the empty-plan / context-only / planner-failure paths, where *no search*
    ran so "zero hits" is not the replan-trigger §15 describes), that search
    returned zero hits (``total_hits == 0``), and no replan has been used yet
    (``replans_used == 0`` -- the §15 one-loop cap; a second pass that still
    finds nothing routes to the synthesizer rather than looping again).

    Pure: takes the three state values directly so ``tests/test_graph.py`` can
    pin the matrix across the full Cartesian product without any LLM or DB.
    """
    return bool(had_search and total_hits == 0 and replans_used == 0)


def _replan_target(state: RepoLensState) -> str:
    """The conditional edge out of the terminal execution node (``search`` when
    no context follows, ``context`` always): route to ``planner`` (one §15
    replan) or ``synthesizer`` (no replan / cap reached). Reads the three
    ``_needs_replan`` inputs from state; pure, no I/O."""
    return (
        "planner"
        if _needs_replan(
            state.get("had_search", False),
            state.get("total_hits", 0),
            state.get("replans_used", 0),
        )
        else "synthesizer"
    )


def _route_after_planner(state: RepoLensState) -> str:
    """The conditional edge out of ``planner``: which execution branch (if any)
    the plan actually warrants. Search first (sequential -- SDD 15 permits it,
    and it needs no fan-out barrier), then context; an empty plan / planner
    failure (``had_search`` and ``had_context`` both False) routes straight to
    the synthesizer (which will say plainly that nothing matched -- SDD 10: an
    empty result is a valid answer). The router trusts the planner node's own
    ``had_search``/``had_context`` computation (it walks ``plan["steps"]`` once
    per planner pass), so a replan that changes its mind is honored.

    Pure: reads only ``had_search`` / ``had_context`` from state."""
    if state.get("had_search", False):
        return "search"
    if state.get("had_context", False):
        return "context"
    return "synthesizer"


def _route_after_search(state: RepoLensState) -> str:
    """The conditional edge out of ``search``: either continue to ``context``
    (the plan had context steps too -- search ran first, context runs next) or
    fall through to the replan decision (:func:`_replan_target`). Pure."""
    if state.get("had_context", False):
        return "context"
    return _replan_target(state)


# ─── result wrapper ──────────────────────────────────────────────────────────


@dataclass
class GraphResult:
    """The clean object :func:`run_query` returns so a caller never touches a
    raw LangGraph state dict. Carries the SDD 15 six control fields verbatim
    (``plan`` / ``search_results`` / ``context_results`` / ``replans_used`` /
    ``final_answer`` plus the echo of ``question``) and the observability fields
    the routing logic populated (``had_search`` / ``had_context`` /
    ``total_hits`` / the ordered ``node_trace`` / the per-stage ``errors``) plus
    the synthesizer's citation aid (``cited_file_paths`` / ``hit_file_paths`` /
    ``synth_wall_time_s``) so the demo re-derives the same ground-vs-cited check
    the standalone synthesizer demo prints.

    Two convenience predicates: :attr:`replanned` (a replan loop actually fired)
    and :attr:`answered` (the synthesizer produced a non-empty final answer --
    distinct from "the graph completed," which happens even on planner failure
    with ``final_answer=None``).
    """

    question: str
    plan: dict | None
    search_results: list[SearchStepResult]
    context_results: list[ContextStepResult]
    replans_used: int
    final_answer: str | None
    had_search: bool
    had_context: bool
    total_hits: int
    node_trace: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cited_file_paths: list[str] = field(default_factory=list)
    hit_file_paths: list[str] = field(default_factory=list)
    synth_wall_time_s: float = 0.0

    @property
    def replanned(self) -> bool:
        """True iff the §15 one-replan loop actually fired (``replans_used > 0``)."""
        return self.replans_used > 0

    @property
    def answered(self) -> bool:
        """True iff the synthesizer produced a non-empty final answer (distinct
        from "the graph completed" -- a planner/synthesizer failure completes the
        graph with ``final_answer=None``)."""
        return bool(self.final_answer and self.final_answer.strip())


# ─── live wrapper: build_graph + the four node closures ──────────────────────


def build_graph(
    repository_id: int,
    session: Session,
    *,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
):
    """Build + compile the §15 graph with the four agent nodes wired, closing
    over the orchestrator-supplied ``repository_id`` + ``Session`` + Ollama
    ``host``/``model`` (the same injected infra the two dispatchers take). The
    closures -- rather than ``context_schema`` runtime config -- keep the wiring
    simple and match how narrow each agent already is. Returns the compiled
    LangGraph (``.invoke(state_dict) -> final_state_dict``).

    The ``session`` is shared by the search + context node dispatches for the
    whole invoke (SDD 15 "one session per query"; the tools are pure reads so
    sharing is safe -- the caller owns the transaction and the session's
    lifetime, exactly like the chain demos + dispatch tests).
    """

    # ── planner node (SDD 9.1) ──────────────────────────────────────────────
    def planner_node(state: RepoLensState) -> dict[str, Any]:
        # On a replan re-entry ``plan`` holds the prior pass's plan (not None),
        # which is the only way here that planner gets called twice; that's the
        # cue to bump ``replans_used`` so the second ``_replan_target`` evaluation
        # sees the cap reached (== 1) and routes to synthesizer. The first call
        # sees the seeded ``plan=None`` -> no bump. Sound because every replan
        # re-entry is gated on ``had_search`` (which requires the prior plan to
        # have been a real dict with search steps), so ``plan`` is guaranteed
        # non-None on every re-entry.
        is_replan = state.get("plan") is not None
        replans_used = state.get("replans_used", 0) + (1 if is_replan else 0)
        trace = ["planner[replan]" if is_replan else "planner"]
        question = state.get("query", "")

        try:
            res = plan(
                question,
                host=host,
                model=model,
                fmt="json",
                retries=0,
                temperature=0.0,
                think=False,
            )
        except OllamaError as exc:
            return {
                "plan": None,
                "had_search": False,
                "had_context": False,
                "replans_used": replans_used,
                "node_trace": trace,
                "errors": [f"planner:ollama:{exc}"],
            }

        if not res.json_valid or not isinstance(res.plan, dict):
            # No parseable plan -> had_search/had_context both False, so
            # _route_after_planner falls through to synthesizer (no replan:
            # had_search False means _needs_replan is False). The synthesizer
            # will report that nothing matched.
            parse_err = getattr(res, "parse_error", None) or "non-dict plan"
            return {
                "plan": None,
                "had_search": False,
                "had_context": False,
                "replans_used": replans_used,
                "node_trace": trace,
                "errors": [f"planner:parse:{parse_err}"],
            }

        steps = res.plan.get("steps")
        if not isinstance(steps, list):
            steps = []
        had_search = any(
            isinstance(s, dict) and s.get("agent") == "search" for s in steps
        )
        had_context = any(
            isinstance(s, dict) and s.get("agent") == "context" for s in steps
        )
        return {
            "plan": res.plan,
            "had_search": had_search,
            "had_context": had_context,
            "replans_used": replans_used,
            "node_trace": trace,
        }

    # ── search node (SDD 9.2; no-LLM dispatcher) ────────────────────────────
    def search_node(state: RepoLensState) -> dict[str, Any]:
        trace = ["search"]
        plan_obj = state.get("plan")
        results: list[SearchStepResult] = []
        errors: list[str] = []
        if isinstance(plan_obj, dict):
            steps = plan_obj.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        errors.append(f"search:non-dict-step:{step!r}")
                        continue
                    if step.get("agent") != "search":
                        # search_node dispatches only the steps it owns
                        # (``agent == "search"``); context steps are handled by
                        # context_node. A step with a missing/unknown ``agent``
                        # is a malformed-plan edge case (the Planner only emits
                        # ``search`` today) -- it is skipped here and likewise by
                        # context_node; the dispatchers' own tool-name validation
                        # would catch it loudly if a node tried to run it.
                        continue
                    try:
                        results.append(
                            build_search_step_result(step, repository_id, session)
                        )
                    except SearchDispatchError as exc:
                        errors.append(
                            f"search:dispatch-{step.get('tool')!r}:{exc}"
                        )
                    except Exception as exc:  # noqa: BLE001 -- surface tool-layer crash
                        errors.append(
                            f"search:tool-{step.get('tool')!r}:"
                            f"{type(exc).__name__}:{exc}"
                        )
        # Last-write-wins: this pass's results + total REPLACE the prior pass's
        # (a replan's second search overwrites the zero-hit first pass cleanly).
        total_hits = sum(len(r.hits) for r in results)
        return {
            "search_results": results,
            "total_hits": total_hits,
            "node_trace": trace,
            "errors": errors,
        }

    # ── context node (SDD 9.3; no-LLM dispatcher) ───────────────────────────
    def context_node(state: RepoLensState) -> dict[str, Any]:
        trace = ["context"]
        plan_obj = state.get("plan")
        results: list[ContextStepResult] = []
        errors: list[str] = []
        if isinstance(plan_obj, dict):
            steps = plan_obj.get("steps")
            if isinstance(steps, list):
                for step in steps:
                    if not isinstance(step, dict):
                        errors.append(f"context:non-dict-step:{step!r}")
                        continue
                    if step.get("agent") != "context":
                        continue
                    try:
                        results.append(
                            build_context_step_result(step, repository_id, session)
                        )
                    except ContextDispatchError as exc:
                        errors.append(
                            f"context:dispatch-{step.get('tool')!r}:{exc}"
                        )
                    except Exception as exc:  # noqa: BLE001 -- surface tool-layer crash
                        errors.append(
                            f"context:tool-{step.get('tool')!r}:"
                            f"{type(exc).__name__}:{exc}"
                        )
        # Last-write-wins (same as search_results); context is terminal so this
        # is only re-written if a replan re-runs context, which overwrites fine.
        return {
            "context_results": results,
            "node_trace": trace,
            "errors": errors,
        }

    # ── synthesizer node (SDD 9.4; the only agent that writes prose) ────────
    def synthesizer_node(state: RepoLensState) -> dict[str, Any]:
        trace = ["synthesizer"]
        question = state.get("query", "")
        sr = state.get("search_results") or []
        cr = state.get("context_results") or []
        try:
            synth: SynthesizerResult = synthesize(
                question,
                sr,
                host=host,
                model=model,
                temperature=0.0,
                think=False,
                context_results=cr,
            )
        except OllamaError as exc:
            return {
                "final_answer": None,
                "node_trace": trace,
                "errors": [f"synthesizer:ollama:{exc}"],
            }
        return {
            "final_answer": synth.answer,
            "cited_file_paths": synth.cited_file_paths,
            "hit_file_paths": synth.hit_file_paths,
            "synth_wall_time_s": synth.wall_time_s,
            "node_trace": trace,
        }

    # ── wire the §15 topology ───────────────────────────────────────────────
    graph = StateGraph(RepoLensState)
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("context", context_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    # After planner: search | context | synthesizer (skip a branch entirely when
    # the plan has no steps for that agent; empty plan -> synthesizer directly).
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        path_map={
            "search": "search",
            "context": "context",
            "synthesizer": "synthesizer",
        },
    )
    # After search: continue to context if the plan has context steps, else the
    # replan decision (planner-replan | synthesizer).
    graph.add_conditional_edges(
        "search",
        _route_after_search,
        path_map={
            "context": "context",
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )
    # After context (terminal execution node): the replan decision (context-only
    # plans never replan -- had_search False -> _needs_replan False -> synth).
    graph.add_conditional_edges(
        "context",
        _replan_target,
        path_map={"planner": "planner", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ─── top-level entrypoint ────────────────────────────────────────────────────


def _initial_state(question: str) -> dict[str, Any]:
    """Seed every LastValue channel so the first node read does not hit
    ``EmptyChannelError`` (``LastValue`` raises on read before any write; the
    accumulating channels default to ``[]`` so they are safe to omit, but all
    fields are seeded here for explicitness and robustness).

    ``plan`` is seeded ``None`` -- that is how :func:`planner_node` detects the
    *first* call (vs a replan re-entry, where ``plan`` holds the prior pass's
    dict) and decides whether to bump ``replans_used``."""
    return {
        "query": question,
        "plan": None,
        "search_results": [],
        "context_results": [],
        "replans_used": 0,
        "final_answer": None,
        "had_search": False,
        "had_context": False,
        "total_hits": 0,
        "cited_file_paths": [],
        "hit_file_paths": [],
        "synth_wall_time_s": 0.0,
        "node_trace": [],
        "errors": [],
    }


def run_query(
    question: str,
    repository_id: int,
    session: Session,
    *,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
) -> GraphResult:
    """Run one user question through the full §15 graph and return the answer +
    the full state, callable without knowing anything about LangGraph internals.

    The single top-level entrypoint the demo calls now and the future FastAPI
    ``/chat`` endpoint will call later. Opens nothing itself -- the caller owns
    the ``Session`` (SDD 15 "one session per query"; pass a freshly-opened
    ``SessionLocal()`` so the search + context dispatches share it for the whole
    invoke, exactly like the chain demos). Returns a :class:`GraphResult`.

    The graph is built + compiled per call (closures over
    ``repository_id``/``session``/``host``/``model``); compiling is cheap (a few
    ms) and is how per-call infra is injected cleanly without ``context_schema``
    runtime config. The §15 one-replan loop is bounded by ``replans_used``, so
    the graph completes in <= ~10 supersteps -- well under LangGraph's default
    recursion limit; no checkpointer is needed (the graph is stateless across
    calls, the tools are pure reads).
    """
    compiled = build_graph(
        repository_id, session, host=host, model=model
    )
    final = compiled.invoke(_initial_state(question))

    def _lst(key: str) -> list:
        v = final.get(key)
        return list(v) if isinstance(v, list) else []

    # ``plan`` may be None (planner failure) or a dict -- coerce to dict | None.
    plan_val = final.get("plan")
    return GraphResult(
        question=question,
        plan=plan_val if (plan_val is None or isinstance(plan_val, dict)) else None,
        search_results=_lst("search_results"),
        context_results=_lst("context_results"),
        replans_used=int(final.get("replans_used", 0) or 0),
        final_answer=(
            final.get("final_answer")
            if isinstance(final.get("final_answer"), str)
            else None
        ),
        had_search=bool(final.get("had_search", False)),
        had_context=bool(final.get("had_context", False)),
        total_hits=int(final.get("total_hits", 0) or 0),
        node_trace=_lst("node_trace"),
        errors=_lst("errors"),
        cited_file_paths=_lst("cited_file_paths"),
        hit_file_paths=_lst("hit_file_paths"),
        synth_wall_time_s=float(final.get("synth_wall_time_s", 0.0) or 0.0),
    )
