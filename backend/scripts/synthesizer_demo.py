"""THROWAWAY chain demo: Planner (SDD 9.1) -> Search Agent (9.2) -> Synthesizer (9.4).

Not part of the application. The whole point: prove the three agents line up
end-to-end and produce the FIRST real, user-facing, cited answer *before*
wiring LangGraph (SDD 15). A sibling to ``planner_search_chain_demo.py``, which
proved the Planner -> Search hop (3/3 clean, zero friction); this one appends
the synthesize() hop that turns the Search Agent's structured hits into prose.

Flow per question:

1. Call :func:`app.agents.planner.plan` (fmt='json', think=False, temperature=0
   -- the production-recommended config; the SDD 18 JSON lever + the
   budget-exhaustion fix confirmed in planner_diag).
2. Take the FIRST step of that plan **unmodified** straight to
   :func:`app.agents.search_agent.build_search_step_result` (the dispatcher's
   pure shape -- no re-shaping, no arg renaming), with the flask repo_id and an
   open Session.
3. Pass the resulting :class:`SearchStepResult` (as a one-element list, the
   shape synthesize() takes -- "all structured outputs collected so far", SDD
   9.4) to :func:`app.agents.synthesizer.synthesize` (think=False, temperature=0)
   -- the only agent that writes prose, instructed to cite ONLY paths/lines
   present in the hits (SDD 9.4 grounding rule).
4. Print the final natural-language answer -- the first time the pipeline
   produces one -- plus a citation eyeball aid: the grounding set (real hit
   paths) vs the paths the model actually named, with the invented-vs-uncited
   diffs. The verification build is NOT done here (the task defers it); the
   diff is just a reader aid for eyeballing the demo output together.

Everything that raises at any stage is CAUGHT and printed as a labeled FRICTION
point, never swallowed -- same posture as the chain demo. Each question ends
with a one-line CHAIN VERDICT (answered / friction), and a summary table closes
the run.

Preconditions (same as the chain demo + search-agent live tests): Ollama
running with a model (override via OLLAMA_MODEL), Postgres up with the flask
repo indexed (``repositories.name == 'flask'``; run scripts/run_walker_once.py
+ index_all_flask_symbols.py first), and for any plan step that picks
``search_text``, ripgrep on PATH + the flask clone on disk at the repo's
``url_or_path``. Missing preconditions are reported and the question skipped,
not crashed.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/synthesizer_demo.py
"""
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Allow running as `python scripts/synthesizer_demo.py` from backend/ without a
# pre-set PYTHONPATH (mirrors tests/conftest.py + the other scripts).
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
from app.agents.synthesizer import SynthesizerResult, synthesize  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import File, Repository, Symbol  # noqa: E402


# The SAME 3 questions as planner_search_chain_demo.py, chosen to span all
# three MVP tools and reused verbatim so these answers are comparable to the
# "3/3 clean dispatch" baseline already recorded in CLAUDE.md.
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
    planner_demo._resolve_model / planner_search_chain_demo._resolve_model --
    kept local so this script is self-contained (the same precedent the per-tool
    copies of ``_ilike_contains`` follow)."""
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
    Returns ``None`` if flask isn't indexed -- the caller skips, not crashes.
    Mirrors the readiness probe in planner_search_chain_demo.py."""
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
    for all (every field is a plain JSON type -- the SDD 10 contract)."""
    out = []
    for h in hits[:HIT_PRINT_CAP]:
        out.append(dataclasses.asdict(h))
    return out


def _print_citation_aid(res: SynthesizerResult) -> None:
    """Print the grounding eyeball aid: the real hit paths (what the model was
    permitted to cite) vs the paths the model actually named, with the
    invented-vs-uncited diffs. NOT the verification build (the task defers that)
    -- just enough to eyeball whether the answer stays grounded. Uses set
    difference over the two lists synthesize() captured."""
    ground = set(res.hit_file_paths)
    cited = set(res.cited_file_paths)
    invented = [p for p in res.cited_file_paths if p not in ground]
    uncited = [p for p in res.hit_file_paths if p not in cited]
    print("  --- citations (eyeball aid; verification not built yet) ---")
    print(f"  ground (real hit paths, n={len(ground)}) : {sorted(ground)}")
    print(f"  cited (model named,    n={len(cited)}) : {res.cited_file_paths}")
    print(f"  invented? (cited - ground) : {invented if invented else '(none -- every cited path is real)'}")
    print(f"  uncited real? (ground - cited) : {uncited if uncited else '(none -- every real hit was cited)'}")


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
    print("RepoLens Planner -> Search -> Synthesizer CHAIN demo (SDD 9.1->9.2->9.4)")
    print(f"  Ollama host : {host}")
    print(f"  model       : {model}")
    print(f"  planner     : fmt='json'  think=False  retries=0  (production-recommended)")
    print(f"  synthesizer : think=False  temperature=0  (prose; no format, no retries)")
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

        # ── stage 1: Planner ──────────────────────────────────────────────
        try:
            res = plan(q, host=host, model=model, fmt="json", retries=0, temperature=0.0)
        except OllamaError as exc:
            print(f"  PLANNER OLLAMA ERROR: {exc}")
            print("  (start `ollama serve`; pull the model; override via OLLAMA_HOST/OLLAMA_MODEL)")
            rows.append({"qi": qi, "q": q, "stage": "planner", "verdict": "ollama-error",
                         "detail": str(exc)[:80]})
            continue

        if not res.json_valid or not isinstance(res.plan, dict):
            rows.append({"qi": qi, "q": q, "stage": "planner", "verdict": "no-parseable-plan",
                         "detail": "JSON parse failed -- no step to dispatch"})
            print("  CHAIN VERDICT: FRICTION (Planner produced no parseable plan)")
            continue
        steps = res.plan.get("steps")
        if not isinstance(steps, list) or not steps:
            rows.append({"qi": qi, "q": q, "stage": "planner", "verdict": "empty-steps",
                         "detail": "plan valid but steps empty"})
            print("  CHAIN VERDICT: FRICTION (Planner emitted an empty steps list)")
            continue

        step = steps[0]
        print(f"  --- Planner -> first step (passed UNMODIFIED to dispatcher) ---")
        print("  " + json.dumps(step, ensure_ascii=False).replace("\n", "\n  "))

        # ── stage 2: Search Agent dispatcher ─────────────────────────────
        try:
            with SessionLocal() as session:
                result = build_search_step_result(step, repo_id, session)
        except SearchDispatchError as exc:
            print(f"  SEARCH AGENT DISPATCH ERROR: {exc}")
            rows.append({"qi": qi, "q": q, "stage": "dispatcher", "verdict": "dispatch-error",
                         "detail": str(exc)[:80]})
            print("  CHAIN VERDICT: FRICTION (dispatcher rejected the Planner's step)")
            continue
        except Exception as exc:  # noqa: BLE001 -- surface any tool-layer crash
            print(f"  SEARCH AGENT TOOL ERROR ({type(exc).__name__}): {exc}")
            rows.append({"qi": qi, "q": q, "stage": "tool", "verdict": "tool-error",
                         "detail": f"{type(exc).__name__}: {str(exc)[:60]}"})
            print("  CHAIN VERDICT: FRICTION (dispatched OK but the tool raised)")
            continue

        print()
        print("  --- Search Agent: dispatch result ---")
        print(f"  tool           : {result.tool}")
        print(f"  args (used)    : {result.args}")
        print(f"  hits           : {len(result.hits)} total")
        rendered = _fmt_hits(result.hits)
        if rendered:
            print(f"  --- hit sample (up to {HIT_PRINT_CAP}) ---")
            for h in rendered:
                print("    " + json.dumps(h, ensure_ascii=False))
        else:
            print("  --- hit sample : (no hits) ---")

        # ── stage 3: Synthesizer (the only agent that writes prose) ─────
        try:
            synth = synthesize(
                q,
                [result],  # synthesize() takes the list of all step results
                host=host,
                model=model,
                temperature=0.0,
            )
        except OllamaError as exc:
            print(f"  SYNTHESIZER OLLAMA ERROR: {exc}")
            rows.append({"qi": qi, "q": q, "stage": "synthesizer", "verdict": "ollama-error",
                         "detail": str(exc)[:80]})
            print("  CHAIN VERDICT: FRICTION (search OK but synthesizer couldn't reach Ollama)")
            continue

        print()
        print(f"  --- SYNTHESIZER: final answer (model={synth.model}, wall={synth.wall_time_s:.1f}s) ---")
        # Indent the answer so it reads as a block distinct from the stage logs.
        for line in synth.answer.splitlines() or [""]:
            print("    " + line)
        print()
        _print_citation_aid(synth)

        n_hits = len(result.hits)
        invented = [p for p in synth.cited_file_paths if p not in set(synth.hit_file_paths)]
        rows.append({
            "qi": qi, "q": q, "stage": "synthesizer", "verdict": "answered",
            "detail": f"tool={result.tool} hits={n_hits} cited={len(synth.cited_file_paths)} "
                      f"invented={len(invented)} wall={synth.wall_time_s:.1f}s",
        })
        print()
        n_inv = len(invented)
        print(f"  CHAIN VERDICT: ANSWERED (final answer produced; {n_hits} hit(s), "
              f"{len(synth.cited_file_paths)} cited path(s), {n_inv} invented)")

    # ─── summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Q#':<3} {'verdict':<10} {'stage':<12} {'detail'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['qi']:<3} {r['verdict']:<10} {r['stage']:<12} {r['detail'][:48]}")

    n_answered = sum(1 for r in rows if r["verdict"] == "answered")
    n_friction = len(rows) - n_answered
    print()
    print(f"answered: {n_answered}/{len(rows)}   friction: {n_friction}/{len(rows)}")
    print()
    print("done.")


if __name__ == "__main__":
    main()
