"""THROWAWAY diagnostic: run Q1 through the full Planner->Search->Synthesizer
chain N times and report, per run, whether the final answer cites the
non-sansio ``blueprints.py``, the ``sansio/blueprints.py``, BOTH, or neither.

Purpose: decide whether the Synthesizer's same-name-coverage gap -- citing only
one of two hits that share the EXACT symbol name "Blueprint" -- is a HARD
failure (0/N for "both") or an INCONSISTENT one. If it's 0/N, the gap is likely
a real local-model capability limit at temperature=0 / this model size, NOT a
prompt-wording issue, and further prompt iteration is not the right lever. If
it's e.g. 1/N or 2/N, the rule is landing intermittently and wording IS worth
tuning. NB: not a verdict ON the model -- a verdict on whether to keep
prompt-iterating at all (same posture as planner_diag: measure before tuning).

Mirrors ``scripts/synthesizer_demo.py``'s Q1 path verbatim (same ``plan()`` /
``synthesize()`` config: fmt=\\='json'\\, think=False, temperature=0, retries=0)
so these runs are comparable to the demo's recorded Q1 output. NOT part of the
app -- a one-shot harness, same status as the other throwaway ``scripts/``.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/synth_q1_repeat.py
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Allow running as `python scripts/synth_q1_repeat.py` from backend/ without a
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
    FileResult,
    SearchDispatchError,
    SymbolResult,
    TextHit,
    build_search_step_result,
)
from app.agents.synthesizer import synthesize  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import File, Repository, Symbol  # noqa: E402


# The SAME Q1 as synthesizer_demo.py (reused verbatim so the runs are directly
# comparable to the recorded Q1 output). Routes to search_symbols -> SymbolResult
# hits, which carry a POSIX-rel ``.file`` path and a line range.
Q1 = "where is the Blueprint class defined"
N_RUNS = 5


def _resolve_model(host: str) -> str:
    """Pick a model: OLLAMA_MODEL env if set, else first installed via
    GET /api/tags, else the Planner default. Same logic as the other demos."""
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
    """Resolve the flask repo_id from Postgres; confirm it has indexed files +
    symbols. Returns None (caller skips, not crashes) if flask isn't indexed."""
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


def _hit_path(hit) -> str | None:
    """POSIX-rel path of a hit across the three SDD-10 hit dataclasses. A local
    copy of synthesizer._hit_path so this script needs no private import."""
    if isinstance(hit, SymbolResult):
        return hit.file
    if isinstance(hit, FileResult):
        return hit.path
    if isinstance(hit, TextHit):
        return hit.file
    return None


def _tail2(p: str) -> str:
    """Last two path segments of ``p`` (e.g. ``src/flask/sansio/blueprints.py``
    -> ``sansio/blueprints.py``). Used as a citation signal shorter than the
    full path but LONG enough to disambiguate the two same-basename Blueprint
    files (their tail2 forms differ: ``sansio/blueprints.py`` vs
    ``flask/blueprints.py`` -- the ``sansio/`` vs ``flask/`` parent is the
    distinguisher, since no path contains both). A bare ``blueprints.py`` basename
    is NOT a safe signal because both files share it."""
    segs = p.split("/")
    return "/".join(segs[-2:]) if len(segs) >= 2 else p


def _blueprint_def_paths(ground: list[str]) -> tuple[list[str], list[str]]:
    """From the grounding set, pick the paths whose basename is ``blueprints.py``
    (the files that DEFINE the Blueprint class) and split them into
    (sansio_paths, nonsansio_paths) by whether 'sansio' is in the path. The two
    real Blueprint definitions live in ``flask/blueprints.py`` (the canonical,
    WSGI one) and ``flask/sansio/blueprints.py`` (the sans-IO sibling)."""
    bp = [p for p in ground if Path(p).name == "blueprints.py"]
    sansio = sorted(p for p in bp if "sansio" in p)
    nonsansio = sorted(p for p in bp if "sansio" not in p)
    return sansio, nonsansio


def _path_cited(ans: str, paths: list[str]) -> bool:
    """Is any of ``paths`` cited in the answer? True if the FULL path appears as
    a substring (the model cites ``path/to/file.py:line``) OR the 2-segment tail
    appears (the model sometimes shortens the prefix). The tail still
    disambiguates the two Blueprint files because their tail2 forms differ."""
    for p in paths:
        if p in ans:
            return True
        if _tail2(p) in ans:
            return True
    return False


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    model = _resolve_model(host)

    print("=" * 78)
    print(f"Q1 repeat diagnostic: {Q1!r}  x{N_RUNS}  (full chain each run)")
    print(f"  host={host}  model={model}  planner/synth think=False temperature=0")
    print("=" * 78)

    repo_id = _flask_repo_id()
    if repo_id is None:
        print("flask repo is not indexed in Postgres. Run scripts/run_walker_once.py")
        print("+ index_all_flask_symbols.py first. Aborting -- nothing to dispatch to.")
        raise SystemExit(1)
    print(f"  flask repo_id: {repo_id}")
    print("=" * 78)

    rows: list[dict] = []
    first_step_json: str | None = None
    first_ground: list[str] | None = None
    sansio_paths: list[str] = []
    nonsansio_paths: list[str] = []

    for i in range(1, N_RUNS + 1):
        print()
        print("#" * 78)
        print(f"RUN {i}/{N_RUNS}")
        print("#" * 78)

        # ── stage 1: Planner ──────────────────────────────────────────────
        try:
            res = plan(Q1, host=host, model=model, fmt="json", retries=0, temperature=0.0)
        except OllamaError as exc:
            print(f"  PLANNER OLLAMA ERROR: {exc}")
            rows.append({"run": i, "bucket": "planner-error"})
            continue
        if not res.json_valid or not isinstance(res.plan, dict):
            print("  PLANNER: no parseable plan")
            rows.append({"run": i, "bucket": "planner-no-plan"})
            continue
        steps = res.plan.get("steps")
        if not isinstance(steps, list) or not steps:
            print("  PLANNER: empty steps")
            rows.append({"run": i, "bucket": "planner-empty"})
            continue
        step = steps[0]
        cur_step_json = json.dumps(step, ensure_ascii=False)
        if first_step_json is None:
            first_step_json = cur_step_json
            print(f"  Planner first step: {cur_step_json}")
        elif cur_step_json != first_step_json:
            print(f"  Planner step CHANGED from run 1: {cur_step_json}")
        else:
            print("  Planner first step: (same as run 1 -- temp=0 deterministic)")

        # ── stage 2: Search Agent dispatcher ─────────────────────────────
        try:
            with SessionLocal() as session:
                result = build_search_step_result(step, repo_id, session)
        except SearchDispatchError as exc:
            print(f"  SEARCH DISPATCH ERROR: {exc}")
            rows.append({"run": i, "bucket": "dispatch-error"})
            continue
        except Exception as exc:  # noqa: BLE001 -- surface any tool-layer crash
            print(f"  SEARCH TOOL ERROR ({type(exc).__name__}): {exc}")
            rows.append({"run": i, "bucket": "tool-error"})
            continue

        ground = sorted({p for h in result.hits if (p := _hit_path(h))})
        if first_ground is None:
            first_ground = ground
            sansio_paths, nonsansio_paths = _blueprint_def_paths(ground)
            print(f"  Search tool={result.tool}  hits={len(result.hits)}")
            print(f"  ground paths (n={len(ground)}):")
            for p in ground:
                print(f"    {p}")
            print(f"  Blueprint defs -> sansio={sansio_paths}  nonsansio={nonsansio_paths}")

        # ── stage 3: Synthesizer ─────────────────────────────────────────
        try:
            synth = synthesize(Q1, [result], host=host, model=model, temperature=0.0)
        except OllamaError as exc:
            print(f"  SYNTHESIZER OLLAMA ERROR: {exc}")
            rows.append({"run": i, "bucket": "synth-error"})
            continue

        ans = synth.answer
        print(f"  --- final answer (wall={synth.wall_time_s:.1f}s) ---")
        for line in ans.splitlines() or [""]:
            print("    " + line)
        print(f"  cited_file_paths (extractor): {synth.cited_file_paths}")

        sansio_cited = _path_cited(ans, sansio_paths)
        nonsansio_cited = _path_cited(ans, nonsansio_paths)
        # Edge case: a cited BARE basename 'blueprints.py' (no parent) matches
        # both files -- flag it explicitly so the printed answer can adjudicate
        # rather than silently mis-bucketing as one side.
        bare_basename = ("blueprints.py" in ans) and not sansio_cited and not nonsansio_cited
        if bare_basename:
            print("  NOTE: bare 'blueprints.py' basename cited but no 2-seg path did (ambiguous -- adjudicate the answer above)")

        if sansio_cited and nonsansio_cited:
            bucket = "both"
        elif sansio_cited and not nonsansio_cited:
            bucket = "sansio-only"
        elif nonsansio_cited and not sansio_cited:
            bucket = "nonsansio-only"
        else:
            bucket = "neither"
        print(f"  --> sansio_cited={sansio_cited}  nonsansio_cited={nonsansio_cited}  BUCKET={bucket}")
        rows.append({
            "run": i, "bucket": bucket,
            "sansio": sansio_cited, "nonsansio": nonsansio_cited,
            "bare": bare_basename,
        })

    # ─── summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'run':<5} {'bucket':<16} {'sansio':<8} {'nonsansio':<11} {'bare-base?'}")
    print("-" * 52)
    for r in rows:
        if r["bucket"] in ("both", "sansio-only", "nonsansio-only", "neither"):
            print(f"{r['run']:<5} {r['bucket']:<16} {str(r['sansio']):<8} "
                  f"{str(r['nonsansio']):<11} {str(r.get('bare', False))}")
        else:
            print(f"{r['run']:<5} {r['bucket']:<16} {'-':<8} {'-':<11} -")

    n_both = sum(1 for r in rows if r.get("bucket") == "both")
    n_sansio = sum(1 for r in rows if r.get("bucket") == "sansio-only")
    n_nonsansio = sum(1 for r in rows if r.get("bucket") == "nonsansio-only")
    n_neither = sum(1 for r in rows if r.get("bucket") == "neither")
    n_err = sum(1 for r in rows if r.get("bucket") not in
                ("both", "sansio-only", "nonsansio-only", "neither"))
    print()
    print(f"both: {n_both}/{N_RUNS}   sansio-only: {n_sansio}/{N_RUNS}   "
          f"nonsansio-only: {n_nonsansio}/{N_RUNS}   neither: {n_neither}/{N_RUNS}   "
          f"errors: {n_err}/{N_RUNS}")
    print()
    if n_both == 0 and (n_sansio + n_nonsansio + n_neither) > 0:
        print("0/N for 'both' -> likely a REAL capability limit at temperature=0 with")
        print("this model size, not a prompt-wording issue. The rule is consistent-failing;")
        print("prompt iteration is probably not the right lever (raise temperature /")
        print("larger model / two-pass synthesis instead).")
    elif n_both > 0 and n_both < N_RUNS:
        print(f"{n_both}/{N_RUNS} 'both' -> INCONSISTENT: the rule lands intermittently,")
        print("so prompt wording IS worth tuning (it CAN fire, just not reliably).")
    elif n_both == N_RUNS:
        print(f"{n_both}/{N_RUNS} 'both' -> the rule now lands every time (no gap).")
    print()
    print("done.")


if __name__ == "__main__":
    main()
