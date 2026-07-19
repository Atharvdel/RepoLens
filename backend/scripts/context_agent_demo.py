"""THROWAWAY dispatch demo: hand-constructed Context plan steps -> Context Agent (SDD 9.3).

Not part of the application. The whole point: prove the Context Agent's four
tools dispatch end-to-end against the real flask repo and return their real
structured results (or the documented clear-empty-until-§7 shape) *before* the
Planner's context menu is wired (SDD 15). A sibling to
``planner_search_chain_demo.py`` / ``synthesizer_demo.py`` -- but DISTINCT:
those ran the real ``plan()`` LLM call; this one skips it, because teaching the
Planner to EMIT ``agent: "context"`` steps is a discrete follow-up (needs its
own Ollama re-verification -- a new tool menu in
``app.agents.planner.DEFAULT_TOOLS_DESCRIPTION``, new ``KNOWN_TOOLS`` /
``OPTIONAL_TOOL_ARGS`` entries, a re-run of the 5-question demo). Until that
lands, the Context Agent is exercised by HAND-CONSTRUCTED plan steps in exactly
the shape the Planner will emit (the SDD §9.1 example writes
``{"agent": "context", "tool": "file_history", "args": {"file": "..."}}``,
modulo the documented ``file``->``target`` alias reconciliation that lands with
that follow-up). So the hop proven here is the §9.3 *dispatch* hop, not the
§9.1->§9.3 planner hop.

Flow per hand-constructed step:

1. Build the step dict (``{"agent": "context", "tool": <name>, "args": {...}}``,
   the Planner's shape) for one Context tool, covering all four across the run
   (architecture whole-repo + focused, dependency_graph, file_history,
   github_metadata).
2. Pass it UNMODIFIED to :func:`app.agents.context_agent.dispatch_context_step`
   with the flask repo_id + an open Session (the same posture the chain demo
   uses for search steps -- no re-shaping, no arg renaming).
3. Print the resulting :class:`ContextStepResult` (tool name + injected
   repository_id + the args actually used + the tool's structured result),
   rendered via ``dataclasses.asdict`` (the SDD §10 "structured JSON, never free
   text" contract -- every field a plain JSON type, straight into the tool_trace
   a real orchestrator would persist, SDD §11).

The two graph tools (architecture, dependency_graph) print REAL flask structure
(the import-graph slice the import-index stage built: 187 internal edges). The
two metadata tools (file_history, github_metadata) print their documented
clear-empty-until-§7 interim shape -- ``last_modified`` populated by the walker
(§7 step 1, done) but contributor/commit lists empty until the §7 step 7
git-history indexer lands; issues/PRs empty until the §7 step 8 PyGithub
indexer lands -- the SAME "build the tool, flag the pending backing" posture the
architecture/dependency-graph tools took toward the import graph before that
stage landed, now applied to the two metadata tools.

Everything that raises is CAUGHT and printed as a labeled FRICTION point
(ContextDispatchError -> a malformed step; a tool-layer Exception -> a
post-dispatch tool crash), never swallowed -- same posture as the chain demo.
Each step ends with a one-line VERDICT (dispatched / friction), and a summary
table closes the run.

Preconditions: Postgres up with the flask repo indexed -- files (walker) for
the metadata tools, AND internal import edges (``scripts/index_all_flask_imports.py``)
for the two graph tools. A missing import graph skips the graph steps (not the
whole run) and reports why; a missing flask repo aborts (nothing to dispatch to).

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/context_agent_demo.py
"""
import dataclasses
import json
import sys
from pathlib import Path

# Allow running as `python scripts/context_agent_demo.py` from backend/ without a
# pre-set PYTHONPATH (mirrors tests/conftest.py + the other scripts).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import sqlalchemy as sa  # noqa: E402

from app.agents.context_agent import (  # noqa: E402
    CONTEXT_REQUIRED_ARGS,
    ContextDispatchError,
    ContextStepResult,
    dispatch_context_step,
)
from app.db import SessionLocal  # noqa: E402
from app.models import Edge, File, Repository  # noqa: E402


# The hand-constructed plan steps -- the shape the Planner WILL emit once its
# context menu is wired (today it emits only ``agent: "search"`` steps; the
# context menu is the documented discrete follow-up). Each step is the SDD §9.1
# step shape ``{"agent": "context", "tool": <name>, "args": {...}}``. ``target``
# is the tools' shared identifier arg (a file path OR a dotted module name --
# ``flask.app`` would work too; ``src/flask/app.py`` is the path the Search
# Agent returns, the documented common case for the resolver's exact-match
# stage). The four steps span all four tools AND both architecture scopes
# (whole-repo + file-focused).
TARGET = "src/flask/app.py"

STEPS: list[dict] = [
    {
        "label": "architecture (whole-repo -- the 'how is this codebase laid out' view)",
        "step": {"agent": "context", "tool": "architecture", "args": {}},
    },
    {
        "label": "dependency_graph (app.py at depth=2 -- the transitive in/out neighborhood)",
        "step": {"agent": "context", "tool": "dependency_graph",
                  "args": {"target": TARGET, "depth": 2}},
    },
    {
        "label": "architecture (focused on app.py -- the region around one file)",
        "step": {"agent": "context", "tool": "architecture", "args": {"target": TARGET, "radius": 1, "top_k": 5}},
    },
    {
        "label": "file_history (app.py -- git history & contributors; §7 step 7 pending)",
        "step": {"agent": "context", "tool": "file_history", "args": {"target": TARGET}},
    },
    {
        "label": "github_metadata (app.py -- linked issues/PRs; §7 step 8 pending)",
        "step": {"agent": "context", "tool": "github_metadata", "args": {"target": TARGET}},
    },
]

# Tools that need the internal import-graph slice (architecture + dependency_graph
# both read ``edges`` WHERE edge_type="imports" AND target_id IS NOT NULL). The
# metadata tools only need a flask ``files`` row to resolve.
GRAPH_TOOL_NAMES = {"architecture", "dependency_graph"}


def _flask_readiness() -> tuple[int | None, bool]:
    """Return ``(repo_id, imports_indexed)`` for the flask repo, or
    ``(None, False)`` if flask isn't indexed at all. ``imports_indexed`` is True
    iff the internal import-graph slice has rows -- the precondition for the two
    graph tools; the metadata tools just need flask's ``files``. Mirrors the
    readiness probes in the context tool tests."""
    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return None, False
            n_files = session.scalar(
                sa.select(sa.func.count())
                .select_from(File)
                .where(File.repository_id == repo.id)
            ) or 0
            if n_files == 0:
                return None, False
            n_imports = session.scalar(
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
            return repo.id, n_imports > 0
    except Exception:
        return None, False


def _truncate(label: str, width: int = 70) -> str:
    """Ellipsize a label for the summary table so long tool result summaries
    don't blow the column."""
    if len(label) <= width:
        return label
    return label[: width - 1] + "…"


def main() -> None:
    # Windows consoles are often cp1252/cp437; force UTF-8 so non-ASCII (§,
    # arrows, ellipsis) in source / model output survives. Best-effort.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    repo_id, imports_indexed = _flask_readiness()

    print("=" * 78)
    print("RepoLens Context Agent DISPATCH demo (SDD 9.3 -- pre-LangGraph)")
    print("  NOTE: Planner context-menu NOT yet wired -- these are HAND-CONSTRUCTED")
    print("        plan steps in the shape the Planner will emit (see module docstring).")
    print("  Context Agent is a NO-LLM dispatcher (executes tools, never chooses them).")
    print(f"  flask repo_id      : {repo_id}")
    print(f"  import graph indexed: {imports_indexed}  (needed for the 2 graph tools)")
    print("  tools              : architecture / dependency_graph / file_history / github_metadata")
    print("  metadata backing   : §7 step 7 (git history) + §7 step 8 (PyGithub) PENDING")
    print("                       -> file_history & github_metadata return clear-empty shape")
    print("=" * 78)

    if repo_id is None:
        print()
        print("flask repo is not indexed in Postgres (no rows in files for")
        print("repositories.name == 'flask'). Run scripts/run_walker_once.py first.")
        print("For the 2 graph tools ALSO run scripts/index_all_flask_imports.py.")
        print("Aborting -- nothing to dispatch to.")
        raise SystemExit(1)

    if not imports_indexed:
        print()
        print("NOTE: flask's internal import graph is NOT indexed (no `imports` edges).")
        print("The 2 graph tools (architecture, dependency_graph) will be skipped;")
        print("the 2 metadata tools still run. Run scripts/index_all_flask_imports.py")
        print("to light them up. (Continuing.)")
        print("=" * 78)

    rows: list[dict] = []

    for si, spec in enumerate(STEPS, 1):
        label = spec["label"]
        step = spec["step"]
        tool_name = step.get("tool")

        print()
        print("#" * 78)
        print(f"STEP {si}/{len(STEPS)}: {label}")
        print("#" * 78)
        print("  --- hand-constructed plan step (the Planner's shape) ---")
        print("  " + json.dumps(step, ensure_ascii=False).replace("\n", "\n  "))

        # Skip graph tools when the import graph isn't indexed (the test would
        # dispatch to an empty slice and return a vacuous-but-honest empty --
        # clearer to skip + report than to print a misleading empty structure).
        if tool_name in GRAPH_TOOL_NAMES and not imports_indexed:
            print("  SKIPPED: import graph not indexed (run index_all_flask_imports.py)")
            rows.append({"si": si, "tool": tool_name, "label": label,
                         "verdict": "skipped", "detail": "import graph not indexed"})
            continue

        try:
            with SessionLocal() as session:
                result = dispatch_context_step(step, repo_id, session)
        except ContextDispatchError as exc:
            print(f"  CONTEXT DISPATCH ERROR: {exc}")
            rows.append({"si": si, "tool": tool_name, "label": label,
                         "verdict": "dispatch-error", "detail": str(exc)[:64]})
            print("  VERDICT: FRICTION (dispatcher rejected the step)")
            continue
        except Exception as exc:  # noqa: BLE001 -- surface any tool-layer crash
            print(f"  TOOL ERROR ({type(exc).__name__}): {exc}")
            rows.append({"si": si, "tool": tool_name, "label": label,
                         "verdict": "tool-error",
                         "detail": f"{type(exc).__name__}: {str(exc)[:60]}"})
            print("  VERDICT: FRICTION (dispatched OK but the tool raised)")
            continue

        assert isinstance(result, ContextStepResult)

        print()
        print("  --- Context Agent: dispatch result ---")
        print(f"  tool           : {result.tool}")
        print(f"  repository_id  : {result.repository_id}")
        print(f"  args (used)    : {result.args}")
        # The structured result: dataclasses.asdict -> plain JSON types (SDD §10).
        result_dict = dataclasses.asdict(result.result)

        # Human legible per-tool summary line(s) so a reader judges the dispatch
        # at a glance before the full structured dump.
        if result.tool == "architecture":
            print(f"  scope          : {result_dict['scope']}")
            print(f"  focus_path     : {result_dict['focus_path']}")
            print(f"  modules        : {len(result_dict['modules'])}  -> "
                  f"{[m['name'] or '<bare scripts>' for m in result_dict['modules']][:6]}")
            print(f"  key_files      : {len(result_dict['key_files'])}  top -> "
                  f"{[(k['path'], round(k['centrality'], 3)) for k in result_dict['key_files']][:5]}")
            print(f"  edges          : {len(result_dict['edges'])}")
        elif result.tool == "dependency_graph":
            print(f"  node_path      : {result_dict['node_path']}")
            print(f"  depth          : {result_dict['depth']}")
            print(f"  neighbors_in   : {len(result_dict['neighbors_in'])}  -> "
                  f"{[n['path'] for n in result_dict['neighbors_in']][:6]}")
            print(f"  neighbors_out  : {len(result_dict['neighbors_out'])}  -> "
                  f"{[n['path'] for n in result_dict['neighbors_out']][:6]}")
        elif result.tool == "file_history":
            print(f"  file_path      : {result_dict['file_path']}")
            print(f"  last_modified  : {result_dict['last_modified']}  (walker §7 step 1)")
            print(f"  top_contributors: {len(result_dict['top_contributors'])}  "
                  f"(EMPTY until §7 step 7 lands)")
            print(f"  recent_commits : {len(result_dict['recent_commits'])}  "
                  f"(EMPTY until §7 step 7 lands)")
        elif result.tool == "github_metadata":
            print(f"  file_path      : {result_dict['file_path']}")
            print(f"  issues         : {len(result_dict['issues'])}  "
                  f"(EMPTY until §7 step 8 PyGithub lands)")
            print(f"  prs            : {len(result_dict['prs'])}  "
                  f"(EMPTY until §7 step 8 PyGithub lands)")

        # The full structured dump, indented -- the SDD §10 JSON shape a real
        # orchestrator would serialize into a chat_messages.tool_trace row.
        print("  --- full structured result (SDD §10 shape) ---")
        rendered = json.dumps(result_dict, ensure_ascii=False, indent=2)
        for line in (rendered.splitlines() or ["(empty)"]):
            print("    " + line)

        rows.append({
            "si": si, "tool": tool_name, "label": label, "verdict": "dispatched",
            "detail": _summary_detail(result, result_dict),
        })
        print()
        print(f"  VERDICT: DISPATCHED ({tool_name}, structured result returned)")

    # ─── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'#':<2} {'verdict':<10} {'tool':<18} {'label'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        label_short = r["label"].split("--")[0].strip()
        print(f"{r['si']:<2} {r['verdict']:<10} {r['tool']:<18} {_truncate(label_short, 44)}")
    print()
    n_dispatched = sum(1 for r in rows if r["verdict"] == "dispatched")
    n_skipped = sum(1 for r in rows if r["verdict"] == "skipped")
    n_friction = sum(1 for r in rows if r["verdict"] in ("dispatch-error", "tool-error"))
    print(f"dispatched: {n_dispatched}/{len(rows)}   skipped: {n_skipped}   friction: {n_friction}")
    print()
    print("Per-tool required-arg registry (the per-tool heterogeneous required set")
    print("the dispatcher enforces -- identifier tools REQUIRE target, the"
    f" overview tools do not): {dict(CONTEXT_REQUIRED_ARGS)}")
    print()
    print("done.")


def _summary_detail(result: ContextStepResult, result_dict: dict) -> str:
    """A one-line per-tool summary for the summary table."""
    if result.tool == "architecture":
        return (f"scope={result_dict['scope']} modules={len(result_dict['modules'])} "
                f"key_files={len(result_dict['key_files'])} edges={len(result_dict['edges'])}")
    if result.tool == "dependency_graph":
        return (f"node={result_dict['node_path']} depth={result_dict['depth']} "
                f"in={len(result_dict['neighbors_in'])} out={len(result_dict['neighbors_out'])}")
    if result.tool == "file_history":
        return (f"path={result_dict['file_path']} last_modified={'yes' if result_dict['last_modified'] else 'no'} "
                f"contributors={len(result_dict['top_contributors'])} commits={len(result_dict['recent_commits'])}")
    if result.tool == "github_metadata":
        return (f"path={result_dict['file_path']} issues={len(result_dict['issues'])} "
                f"prs={len(result_dict['prs'])}")
    return f"{result.tool}: dispatched"


if __name__ == "__main__":
    main()
