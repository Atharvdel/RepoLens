"""THROWAWAY graph demo: runs the same 3 flask questions as the earlier chain
demos through :func:`app.agents.graph.run_query` -- the FIRST time the actual
SDD §15 LangGraph executes end-to-end -- and prints the final answer plus a trace
of which nodes fired and in what order.

Not part of the application. The four §9 agents (Planner 9.1 / Search 9.2 /
Context 9.3 / Synthesizer 9.4) were each built + verified standalone; this demo
wires them through the §15 graph (:mod:`app.agents.graph`) so the connective
tissue -- the routing, the skip-empty-branch logic, the one-replan loop -- is
exercised for the first time on real flask questions. Siblings to
``planner_search_chain_demo.py`` (proved the 9.1->9.2 hop, 3/3 clean) and
``synthesizer_demo.py`` (appended the 9.4 hop); this one lets the graph itself
do the routing rather than hand-driving the four agents from a script.

Per question:

1. Open a fresh ``SessionLocal`` (SDD 15 "one session per query" -- the search +
   context dispatches share it for the whole invoke) and call
   :func:`app.agents.graph.run_query(question, repo_id, session)` -- the single
   top-level entrypoint that hides every LangGraph internal.
2. Print the headline -- the ordered ``node_trace`` ("planner -> search ->
   synthesizer", or with a "planner[replan]" when the §15 one-replan loop fired) --
   then the plan shape, per-step hit counts, ``replans_used``, the final cited
   answer, and a citation aid re-derived from the returned state.
3. Record a one-line verdict; a summary table closes the run.

Everything the graph records as friction lands in ``GraphResult.errors`` (the
nodes catch their own dispatch / tool / Ollama errors and append rather than
raise -- SDD 15 wants the graph to complete and report, not crash mid-route); a
non-graph exception out of ``run_query`` itself (a recursion-limit / structural
issue, unexpected) is caught and printed as a labeled FRICTION point, never
swallowed -- same posture as the chain demo.

Preconditions (same as the earlier chain demos): Ollama running with a model
(override via ``OLLAMA_MODEL``), Postgres up with the flask repo indexed
(``repositories.name == 'flask'``; run ``scripts/run_walker_once.py`` +
``index_all_flask_symbols.py`` first; for the import-graph / reference-index
context tools the relevant indexing scripts too, though the Planner only emits
``search`` steps today so this run is search-only). Missing preconditions are
reported and the run aborts, not crashes.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/langgraph_demo.py
"""
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Allow running as ``python scripts/langgraph_demo.py`` from backend/ without a
# pre-set PYTHONPATH (mirrors tests/conftest.py + the other scripts).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import sqlalchemy as sa  # noqa: E402

from app.agents.graph import GraphResult, run_query  # noqa: E402
from app.agents.planner import OLLAMA_HOST, OLLAMA_MODEL  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import File, Repository, Symbol  # noqa: E402


# The SAME 3 questions as synthesizer_demo.py / planner_search_chain_demo.py,
# reused verbatim so this graph run is directly comparable to the recorded
# "3/3 clean" Planner->Search baseline and the synthesizer demo's cited answers.
#   1 -> search_symbols (a class by name)
#   2 -> search_text    (a phrase the repo's source mentions)
#   3 -> search_files   (a file by name) -- or symbols; the Planner decides.
QUESTIONS: list[str] = [
    "where is the Blueprint class defined",
    "what files mention URL rules",
    "which file contains the Flask class",
]

# How many hits to print per question at the search stage (the tools can return
# dozens; the demo shows enough to judge grounding without flooding the console).
HIT_PRINT_CAP = 8


def _resolve_model(host: str) -> str:
    """Pick a model: ``OLLAMA_MODEL`` env if set, else the first model Ollama
    reports installed via ``GET /api/tags``, else the Planner default. A copy of
    planner_demo._resolve_model / synthesizer_demo._resolve_model -- kept local
    so this script is self-contained (the same precedent the per-tool copies of
    ``_ilike_contains`` follow)."""
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model.strip()
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        models = [m.get("name") or m.get("model") for m in data.get("models", [])]
        models = [m for m in models if m]
        if models:
            return models[0]
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        pass
    return OLLAMA_MODEL


def _flask_repo_id() -> int | None:
    """Resolve the flask repo's id from Postgres, and confirm it has indexed
    symbols + files (the search dispatch needs both). Returns ``None`` if flask
    isn't indexed -- the caller aborts, not crashes. Mirrors the readiness probe
    in synthesizer_demo.py."""
    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return None
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
            if n_files > 0 and n_syms > 0:
                return repo.id
    except Exception:
        pass
    return None


def _fmt_search_hits(result: GraphResult) -> list[dict]:
    """Render the search step results as plain dicts for the hit sample. Each
    :class:`SearchStepResult` carries a ``hits`` list of tool-specific
    dataclasses; ``dataclasses.asdict`` works for all (every field is a plain
    JSON type -- the SDD 10 contract), so the reader need not know which tool
    produced which row."""
    out = []
    for sr in result.search_results:
        tool = sr.tool
        n = len(sr.hits)
        sample = [dataclasses.asdict(h) for h in sr.hits[:HIT_PRINT_CAP]]
        out.append({"tool": tool, "n_hits": n, "sample": sample})
    return out


def _print_result(qi: int, result: GraphResult) -> tuple[str, str]:
    """Print one question's full graph result and return ``(verdict, detail)``
    for the summary row. The ``node_trace`` is the headline -- it shows, for the
    first time, the actual §15 routing this query triggered."""
    # ── the headline: the ordered node-firing trace ────────────────────────
    arrows = " -> ".join(result.node_trace)
    print(f"  --- node trace (which nodes fired, in order) ---")
    print(f"  {arrows}")
    if result.replanned:
        print(f"  (the §15 one-replan loop fired; replans_used={result.replans_used})")
    print()

    # ── the plan the Planner produced ─────────────────────────────────────
    print(f"  --- plan ---")
    if result.plan is None:
        print("  (no parseable plan -- the Planner failed; routed straight to synth)")
    else:
        steps = result.plan.get("steps") if isinstance(result.plan, dict) else []
        print(f"  {len(steps) if isinstance(steps, list) else 0} step(s):")
        if isinstance(steps, list):
            for s in steps:
                if isinstance(s, dict):
                    agent = s.get("agent", "(no-agent)")
                    tool = s.get("tool", "(no-tool)")
                    args = s.get("args", {})
                    print(f"    - agent={agent!r} tool={tool!r} args={args}")
                else:
                    print(f"    - (non-dict step: {s!r})")
    print()

    # ── search stage: per-tool hit counts + a sample ───────────────────────
    print(f"  --- search stage ({len(result.search_results)} step-result(s), "
          f"total_hits={result.total_hits}) ---")
    rendered = _fmt_search_hits(result)
    if rendered:
        for r in rendered:
            print(f"    tool={r['tool']!r}  hits={r['n_hits']}")
            for h in r["sample"]:
                print("      " + json.dumps(h, ensure_ascii=False))
    else:
        print("    (no search steps ran -- plan had no agent='search' steps)")
    print()

    # ── context stage: did the (not-yet-Planner-wired) context branch run? ──
    print(f"  --- context stage ({len(result.context_results)} step-result(s)) ---")
    if result.context_results:
        for cr in result.context_results:
            print(f"    tool={cr.tool!r}  args={cr.args}")
    else:
        print("    (no context steps ran -- Planner's context menu is not yet wired)")
    print()

    # ── the final cited answer (the only agent that writes prose) ───────────
    if result.final_answer:
        print(f"  --- FINAL ANSWER (synth wall={result.synth_wall_time_s:.1f}s) ---")
        for line in result.final_answer.splitlines() or [""]:
            print("    " + line)
    else:
        print("  --- FINAL ANSWER : (none -- synthesizer produced no answer) ---")
    print()

    # ── citation eyeball aid (re-derived from the returned state) ──────────
    ground = set(result.hit_file_paths)
    cited = set(result.cited_file_paths)
    invented = [p for p in result.cited_file_paths if p not in ground]
    uncited = [p for p in result.hit_file_paths if p not in cited]
    print(f"  --- citations (eyeball aid) ---")
    print(f"  ground (real hit paths, n={len(ground)}) : {sorted(ground)}")
    print(f"  cited (model named,    n={len(cited)}) : {result.cited_file_paths}")
    print(f"  invented? (cited - ground) : {invented if invented else '(none -- every cited path is real)'}")
    print(f"  uncited real? (ground - cited) : {uncited if uncited else '(none -- every real hit was cited)'}")
    print()

    # ── friction the graph recorded (per-node errors; never raised) ────────
    if result.errors:
        print(f"  --- friction ({len(result.errors)}) ---")
        for err in result.errors:
            print(f"    ! {err}")
        print()

    answered = result.answered
    n_inv = len(invented)
    verdict = "answered" if answered else "no-answer"
    detail = (
        f"nodes={len(result.node_trace)} replans={result.replans_used} "
        f"hits={result.total_hits} cited={len(cited)} invented={n_inv} "
        f"errors={len(result.errors)}"
    )
    return verdict, detail


def main() -> None:
    # Windows consoles are often cp1252/cp437; force UTF-8 so non-ASCII in model
    # output / matched source lines / paths survives. Best-effort.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    model = _resolve_model(host)

    print("=" * 78)
    print("RepoLens §15 LangGraph demo: Planner -> (Search|Context) -> Synthesizer")
    print(f"  entrypoint  : run_query(question, repo_id, session)  (hides LangGraph)")
    print(f"  Ollama host : {host}")
    print(f"  model       : {model}")
    print(f"  planner     : fmt='json'  think=False  retries=0  (production-recommended)")
    print(f"  synthesizer : think=False  temperature=0  (prose; no format, no retries)")
    print(f"  replan      : once if Search returns zero hits, capped at 1 by replans_used")
    print(f"  questions   : {len(QUESTIONS)}  (span all 3 MVP search tools)")
    print("=" * 78)

    repo_id = _flask_repo_id()
    if repo_id is None:
        print()
        print("flask repo is not indexed in Postgres (no rows in files+symbols for")
        print("repositories.name == 'flask'). Run scripts/run_walker_once.py +")
        print("index_all_flask_symbols.py first. Aborting -- nothing to dispatch to.")
        raise SystemExit(1)
    print(f"  flask repo_id: {repo_id}")
    print("=" * 78)

    rows: list[dict] = []

    for qi, q in enumerate(QUESTIONS, 1):
        print()
        print("#" * 78)
        print(f"Q{qi}: {q!r}")
        print("#" * 78)

        # One session per query (SDD 15); the search + context dispatches share
        # it for the whole invoke, then it closes. run_query hides the graph.
        try:
            with SessionLocal() as session:
                result = run_query(q, repo_id, session, host=host, model=model)
        except Exception as exc:  # noqa: BLE001 -- surface any graph-layer crash
            # The nodes catch their own errors and append to GraphResult.errors;
            # reaching here means something escaped run_query itself (a LangGraph
            # recursion-limit / structural issue, an unexpected exception) --
            # printed as labeled FRICTION, never swallowed (same posture as the
            # chain demo).
            print(f"  GRAPH CRASH ({type(exc).__name__}): {exc}")
            rows.append(
                {
                    "qi": qi,
                    "q": q,
                    "verdict": "crash",
                    "detail": f"{type(exc).__name__}: {str(exc)[:60]}",
                }
            )
            print("  CHAIN VERDICT: FRICTION (run_query raised -- see error)")
            continue

        verdict, detail = _print_result(qi, result)
        rows.append({"qi": qi, "q": q, "verdict": verdict, "detail": detail})
        print(f"  CHAIN VERDICT: {verdict.upper()} ({detail})")

    # ─── summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Q#':<3} {'verdict':<10} {'detail'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['qi']:<3} {r['verdict']:<10} {r['detail'][:50]}")

    n_answered = sum(1 for r in rows if r["verdict"] == "answered")
    n_friction = sum(1 for r in rows if r["verdict"] in ("crash", "no-answer"))
    print()
    print(f"answered: {n_answered}/{len(rows)}   friction: {n_friction}/{len(rows)}")
    print()
    print("done.")


if __name__ == "__main__":
    main()
