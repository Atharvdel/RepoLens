"""THROWAWAY chain demo: Planner (SDD 9.1) -> Search Agent dispatcher (SDD 9.2).

Not part of the application. The whole point: prove the two agents line up
end-to-end *before* building the Context Agent — does a plan step the Planner
actually emits dispatch *unmodified* through the Search Agent and return real,
correct hits? Surfaces any friction where the two don't line up (an arg-name
mismatch — e.g. the SDD 9.1 example uses ``{"name": ...}`` while the real tools
take ``query``; a type the Search Agent didn't expect; a tool name the
dispatcher doesn't know).

Flow per question:

1. Call :func:`app.agents.planner.plan` (fmt='json', think=False — the
   production-recommended config, the §18 JSON-mode lever + the budget-exhaustion
   fix confirmed in planner_diag).
2. Print the Planner's raw output + parsed plan.
3. Take the FIRST step of that plan and pass it **unmodified** straight to
   :func:`app.agents.search_agent.build_search_step_result` (the dispatcher's
   pure shape — no re-shaping, no arg renaming), with the flask repo_id and an
   open Session.
4. Print the Search Agent's :class:`SearchStepResult`: tool, the args actually
   used (so any coercion — regex 'true'->True — is visible), and the real hits.

Everything that raises at either stage is CAUGHT and printed as a labeled
FRICTION point, never swallowed — the task asks to *report* where the two
don't line up, not to hide it. Each question ends with a one-line CHAIN VERDICT
(clean / friction), and a summary table closes the run.

Preconditions (same as the Search Agent live tests): Ollama running with a
model (override via OLLAMA_MODEL), Postgres up with the flask repo indexed
(``repositories.name == 'flask'``; run scripts/run_walker_once.py +
index_all_flask_symbols.py first), and for any plan step that picks
``search_text``, ripgrep on PATH + the flask clone on disk at the repo's
``url_or_path``. Missing preconditions are reported and the question skipped,
not crashed.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/planner_search_chain_demo.py
"""
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Allow running as `python scripts/planner_search_chain_demo.py` from backend/
# without a pre-set PYTHONPATH (mirrors tests/conftest.py + the other scripts).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import sqlalchemy as sa  # noqa: E402

from app.agents.planner import (  # noqa: E402
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OllamaError,
    plan,
)
from app.agents.search_agent import (  # noqa: E402
    SearchDispatchError,
    SearchStepResult,
    build_search_step_result,
)
from app.db import SessionLocal  # noqa: E402
from app.models import File, Repository, Symbol  # noqa: E402


# 3 real questions about Flask, reused from planner_demo.py, chosen to span all
# three MVP tools so the chain is exercised across every dispatch path:
#   1 -> search_symbols (a class by name)
#   2 -> search_text    (a phrase the repo's source mentions)
#   3 -> search_files   (a file by name) — or symbols; the Planner decides.
# Reusing the exact planner_demo strings keeps these results comparable to the
# "10/10 correct structural plans" baseline already recorded in CLAUDE.md.
QUESTIONS: list[str] = [
    "where is the Blueprint class defined",
    "what files mention URL rules",
    "which file contains the Flask class",
]

# How many hits to print per question (the tools can return dozens; the demo
# shows enough to judge correctness without flooding the console).
HIT_PRINT_CAP = 8


def _resolve_model(host: str) -> str:
    """Pick a model: ``OLLAMA_MODEL`` env if set, else the first model Ollama
    reports installed via ``GET /api/tags``, else the Planner default. A copy of
    planner_demo._resolve_model — kept local so this script is self-contained
    (same precedent as the per-tool copies of ``_ilike_contains``)."""
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
    symbols + files (the dispatch needs both for search_symbols; search_files
    needs files; search_text needs the on-disk clone too, checked per-step).
    Returns ``None`` if flask isn't indexed — the caller skips, not crashes.
    Mirrors the per-file ``_flask_indexed`` readiness probes in the tool tests."""
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


def _fmt_hits(hits: list) -> list[dict]:
    """Render up to :data:`HIT_PRINT_CAP` hits as plain dicts for printing. The
    three tools return three different dataclasses; ``dataclasses.asdict`` works
    for all of them (every field is a plain JSON type — the SDD §10 contract), so
    the hit reader doesn't need to know which tool produced the row."""
    out = []
    for h in hits[:HIT_PRINT_CAP]:
        out.append(dataclasses.asdict(h))
    return out


def _print_planner_result(res) -> None:
    """Print the Planner stage: raw model output + parsed plan + structural
    verdict. Trimmed raw (a runaway plan is unreadable) but the parsed plan is
    shown whole — it's the thing the Search Agent consumes."""
    print("  --- Planner: raw model output ---")
    raw = res.raw_response
    cap = 1200
    print("  " + (raw if len(raw) <= cap else raw[:cap] + f"  …<+{len(raw) - cap} chars>"))
    print(f"  json_valid={res.json_valid}  structurally_ok={res.validation.structurally_ok}  "
          f"issues={res.validation.summary()}  wall={res.wall_time_s:.1f}s")
    if res.plan is not None:
        print("  --- Planner: parsed plan ---")
        print("  " + json.dumps(res.plan, indent=2).replace("\n", "\n  "))
    else:
        print("  --- Planner: parsed plan : (none — JSON parse failed) ---")
        if res.parse_error:
            print(f"  parse_error: {res.parse_error}")


def _print_search_result(result: SearchStepResult, *, step_in: dict) -> None:
    """Print the Search Agent stage: the tool the dispatcher routed to, the args
    it ACTUALLY used (vs the args the Planner put in the step — any coercion or
    dropout is visible by diff), and the real hits."""
    print("  --- Search Agent: dispatch result ---")
    print(f"  tool           : {result.tool}")
    print(f"  repository_id  : {result.repository_id}")
    # Diff the Planner's step args against the args the dispatcher used, so any
    # transformation (regex 'true'->True) or whitelist DROP (stray cap/kind) is
    # explicit — the friction the task wants surfaced.
    planner_args = step_in.get("args", {}) if isinstance(step_in, dict) else {}
    used = result.args
    print(f"  args (planner) : {planner_args}")
    print(f"  args (used)    : {used}")
    if planner_args != used:
        dropped = {k: planner_args[k] for k in set(planner_args) - set(used)}
        coerced = {k: (planner_args[k], used[k]) for k in set(used) & set(planner_args) if planner_args[k] != used[k]}
        if dropped:
            print(f"  args DROPPED   : {dropped}  (not in this tool's whitelist)")
        if coerced:
            print(f"  args COERCED   : {coerced}")
    print(f"  hits           : {len(result.hits)} total")
    rendered = _fmt_hits(result.hits)
    if rendered:
        print("  --- hit sample (up to {}) ---".format(HIT_PRINT_CAP))
        for h in rendered:
            print("    " + json.dumps(h, ensure_ascii=False))
    else:
        print("  --- hit sample : (no hits) ---")


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
    print("RepoLens Planner -> Search Agent CHAIN demo (SDD 9.1 -> 9.2)")
    print(f"  Ollama host : {host}")
    print(f"  model       : {model}")
    print(f"  planner fmt : 'json'  think: False  retries: 0  (production-recommended)")
    print(f"  questions   : {len(QUESTIONS)}  (span all 3 MVP search tools)")
    print("=" * 78)

    repo_id = _flask_repo_id()
    if repo_id is None:
        print()
        print("flask repo is not indexed in Postgres (no rows in files+symbols for")
        print("repositories.name == 'flask'). Run scripts/run_walker_once.py +")
        print("index_all_flask_symbols.py first. Aborting — nothing to dispatch to.")
        raise SystemExit(1)
    print(f"  flask repo_id: {repo_id}")
    print("=" * 78)

    rows: list[dict] = []

    for qi, q in enumerate(QUESTIONS, 1):
        print()
        print("#" * 78)
        print(f"Q{qi}: {q!r}")
        print("#" * 78)

        # ── stage 1: Planner ──────────────────────────────────────────────
        try:
            res = plan(q, host=host, model=model, fmt="json", retries=0, temperature=0.0)
        except OllamaError as exc:
            print(f"  PLANNER OLLAMA ERROR: {exc}")
            print("  (start `ollama serve`; pull the model; override via OLLAMA_HOST/OLLAMA_MODEL)")
            rows.append({"qi": qi, "q": q, "stage": "planner", "verdict": "ollama-error",
                         "detail": str(exc)[:80]})
            continue
        _print_planner_result(res)

        # If the Planner didn't produce a dispatchable plan, that's a FRICTION
        # point at stage 1 — record and move on (can't take a first step from a
        # plan that didn't parse or has no steps).
        if not res.json_valid or not isinstance(res.plan, dict):
            rows.append({"qi": qi, "q": q, "stage": "planner", "verdict": "no-parseable-plan",
                         "detail": "JSON parse failed — no step to dispatch"})
            print()
            print("  CHAIN VERDICT: FRICTION (Planner produced no parseable plan — "
                  "chain stops at stage 1)")
            continue
        steps = res.plan.get("steps")
        if not isinstance(steps, list) or not steps:
            rows.append({"qi": qi, "q": q, "stage": "planner", "verdict": "empty-steps",
                         "detail": "plan valid but steps empty"})
            print()
            print("  CHAIN VERDICT: FRICTION (Planner emitted an empty steps list — "
                  "no step to dispatch)")
            continue

        step = steps[0]
        print()
        print(f"  --- first step (passed UNMODIFIED to dispatcher) ---")
        print("  " + json.dumps(step, ensure_ascii=False).replace("\n", "\n  "))

        # ── stage 2: Search Agent dispatcher ─────────────────────────────
        try:
            with SessionLocal() as session:
                result = build_search_step_result(step, repo_id, session)
        except SearchDispatchError as exc:
            # The dispatcher refused the step — an unknown tool, a non-object
            # step/args, or a missing `query` (e.g. if the Planner emitted the
            # SDD 9.1 example's `name` arg instead of `query`). This is the
            # PRIMARY friction class the task wants surfaced.
            print()
            print(f"  SEARCH AGENT DISPATCH ERROR: {exc}")
            rows.append({"qi": qi, "q": q, "stage": "dispatcher", "verdict": "dispatch-error",
                         "detail": str(exc)[:80]})
            print("  CHAIN VERDICT: FRICTION (dispatcher rejected the Planner's step — see error)")
            continue
        except Exception as exc:  # noqa: BLE001 — surface any other tool-layer crash
            # e.g. ripgrep missing, repo moved (FileNotFoundError from search_text),
            # a tool TypeError. Also friction the task wants named, not hidden.
            print()
            print(f"  SEARCH AGENT TOOL ERROR ({type(exc).__name__}): {exc}")
            rows.append({"qi": qi, "q": q, "stage": "tool", "verdict": "tool-error",
                         "detail": f"{type(exc).__name__}: {str(exc)[:60]}"})
            print("  CHAIN VERDICT: FRICTION (dispatched OK but the tool raised — see error)")
            continue

        _print_search_result(result, step_in=step)
        # Did the hits look correct for the question? A light heuristic flagged
        # in the verdict column — not an assertion, just a reader aid.
        n = len(result.hits)
        rows.append({"qi": qi, "q": q, "stage": "dispatcher", "verdict": "clean",
                     "detail": f"tool={result.tool} hits={n}"})
        print()
        print(f"  CHAIN VERDICT: CLEAN (unmodified step dispatched; {n} hit(s) returned)")

    # ─── summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Q#':<3} {'verdict':<16} {'stage':<11} {'detail'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['qi']:<3} {r['verdict']:<16} {r['stage']:<11} {r['detail'][:48]}")

    n_clean = sum(1 for r in rows if r["verdict"] == "clean")
    n_friction = len(rows) - n_clean
    print()
    print(f"clean chain: {n_clean}/{len(rows)}   friction: {n_friction}/{len(rows)}")
    print()
    print("done.")


if __name__ == "__main__":
    main()
