"""THROWAWAY exploratory demo for the standalone Planner (SDD §9.1, pre-LangGraph).

Not part of the application — a harness that feeds 5 real questions about Flask to
:func:`app.agents.planner.plan` and prints, for each, the RAW JSON the local model
actually produced plus whether it parsed and whether it's structurally a valid plan.
The point (per the build task) is to honestly measure how well the local model
handles *structured tool selection* before wiring up the full LangGraph flow — not
to assert it works, but to see where it breaks.

Two modes per question, to separate two distinct failure modes (SDD §18):

* **raw** — ``fmt=None``: no format constraint, measure the model's baseline. If it
  fails here, the question is "does the model emit JSON-shaped text at all?"
* **json** — ``fmt="json"``: Ollama forced to syntactically-valid JSON. If raw fails
  but json passes, the failure was *JSON syntax* and the §18 lever rescues it; if
  json still fails the *shape*, the failure is tool-selection / schema, which
  JSON-mode alone cannot fix (a schema constraint is the heavier lever — out of
  scope for this measurement, decided after seeing these numbers).

Both run at ``temperature=0`` and ``retries=0`` — the honest *unmitigated* baseline.
Retries and the JSON-Schema lever are intentionally left off here; the Planner
module exposes them, this demo keeps them open to measure raw behavior.

Model resolution: ``OLLAMA_MODEL`` env var if set; else probe ``GET /api/tags`` for
the first installed model; else fall back to the Planner's ``qwen3:8b`` default (and
let Ollama's error tell the user to `ollama pull` it if absent). ``OLLAMA_HOST`` env
var if set, else the Planner's default ``http://localhost:11434``. Passed EXPLICITLY
to :func:`plan` so the demo's resolved values win over the Planner's import-time
reads (env may differ from what was baked in at import).

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/planner_demo.py
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Allow running as `python scripts/planner_demo.py` from backend/ without a
# pre-set PYTHONPATH (mirrors tests/conftest.py and the other scripts).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents.planner import (  # noqa: E402
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OllamaError,
    plan,
)


# 5 real questions about Flask, chosen to span all three MVP tools and to vary in
# how unambiguous the tool choice is (the "find something related to sessions" one
# is deliberately vague — a good probe of the model's ambiguity handling).
QUESTIONS: list[str] = [
    "where is the Blueprint class defined",
    "what files mention URL rules",
    "find something related to sessions",
    "which file contains the Flask class",
    "where is before_request used",
]


def _resolve_model(host: str) -> str:
    """Pick a model to run against: ``OLLAMA_MODEL`` env if set, else the first
    model Ollama reports installed via ``GET /api/tags``, else the Planner default.

    The Planner already defaults ``OLLAMA_MODEL``→``qwen3:8b`` at import; this still
    probes ``/api/tags`` when the env var is unset so the demo runs against whatever
    the user actually has pulled (a LAPTOP with ``llama3.1:8b`` only shouldn't fail
    just because the default name is ``qwen3:8b``). Returns the chosen name; the
    caller passes it explicitly to :func:`plan`.
    """
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model.strip()
    # Probe /api/tags (best-effort — if Ollama is down, plan() will surface that).
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
        # Fall through to the Planner default; plan() will report Ollama unreachable.
        pass
    return OLLAMA_MODEL


def _print_result(label: str, res) -> None:
    """Print one PlannerResult: the raw model output, then the parse/validate
    verdicts, then the plan (if parsed). The raw text is the thing the task asks to
    see ("the raw JSON plan the model produces"), so it goes first and untruncated
    up to a sane cap (a runaway 400-token plan is unreadable; cap keeps it useful)."""
    print(f"  [{label}]  model={res.model}")
    print("  --- raw model output ---")
    raw = res.raw_response
    cap = 1200
    print("  " + (raw if len(raw) <= cap else raw[:cap] + f"  …<+{len(raw) - cap} chars>"))
    print("  --- verdict ---")
    print(f"  json_valid        : {res.json_valid}")
    if res.parse_error:
        print(f"  parse_error       : {res.parse_error}")
    print(f"  structurally_ok   : {res.validation.structurally_ok}")
    print(f"  structural issues : {res.validation.summary()}")
    print(f"  retries_used      : {res.retries_used}   wall_time={res.wall_time_s:.1f}s")
    if res.plan is not None:
        print("  --- parsed plan ---")
        print("  " + json.dumps(res.plan, indent=2).replace("\n", "\n  "))
    else:
        print("  --- parsed plan : (none — JSON parse failed) ---")
    # Surface the tool names chosen, for the summary even when shape is wrong.
    tools = []
    if isinstance(res.plan, dict) and isinstance(res.plan.get("steps"), list):
        for s in res.plan["steps"]:
            if isinstance(s, dict) and isinstance(s.get("tool"), str):
                tools.append(s["tool"])
    print(f"  tools chosen      : {tools}")


def main() -> None:
    # Windows consoles are often cp1252/cp437; force UTF-8 so non-ASCII in model
    # output (or paths) survives. Best-effort, mirrors the other driver scripts.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    model = _resolve_model(host)
    print("=" * 78)
    print("RepoLens Planner demo (SDD §9.1, standalone — pre-LangGraph)")
    print(f"  Ollama host : {host}")
    print(f"  model       : {model}")
    print(f"  questions   : {len(QUESTIONS)}  modes: raw (fmt=None) + json-forced (fmt='json')")
    print(f"  temperature : 0.0   retries: 0   (unmitigated baseline)")
    print("=" * 78)

    # Per-question, per-mode outcome rows for the summary table.
    rows: list[dict] = []

    for qi, q in enumerate(QUESTIONS, 1):
        print()
        print("#" * 78)
        print(f"Q{qi}: {q!r}")
        print("#" * 78)
        for mode_label, fmt in (("raw", None), ("json", "json")):
            print(f"\n  >>> mode={mode_label}")
            try:
                res = plan(q, host=host, model=model, fmt=fmt, retries=0, temperature=0.0)
            except OllamaError as exc:
                # Infra failure (Ollama down / model missing) — print and stop the
                # whole demo: no point retrying 9 more calls against a dead server.
                print(f"  OLLAMA ERROR ({mode_label}): {exc}")
                print()
                print("Ollama is unreachable or the model is not installed.")
                print("  1. start:  ollama serve")
                print("  2. pull:   ollama pull <model>")
                print("  (override host via OLLAMA_HOST, model via OLLAMA_MODEL)")
                raise SystemExit(2)
            _print_result(mode_label, res)
            # Tools chosen + step count, for the summary table's "tools/notes" col
            # and the "valid but empty" failure pattern. Recomputed from res.plan
            # (kept off PlannerResult to avoid bloating that dataclass for one demo).
            tools: list[str] = []
            n_steps = 0
            if isinstance(res.plan, dict) and isinstance(res.plan.get("steps"), list):
                n_steps = len(res.plan["steps"])
                for s in res.plan["steps"]:
                    if isinstance(s, dict) and isinstance(s.get("tool"), str):
                        tools.append(s["tool"])
            rows.append(
                {
                    "q": q,
                    "qi": qi,
                    "mode": mode_label,
                    "json_valid": res.json_valid,
                    "structurally_ok": res.validation.structurally_ok,
                    "n_steps": n_steps,
                    "tools": tools,
                    "issues": res.validation.summary(),
                }
            )

    # ─── summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'Q#':<3} {'mode':<5} {'json':<5} {'struct':<7} {'steps':<6} {'tools/notes'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        jv = "Y" if r["json_valid"] else "N"
        st = "OK" if r["structurally_ok"] else "FAIL"
        if r["structurally_ok"]:
            note = ",".join(r["tools"]) if r["tools"] else f"(empty: {r['issues']})"
        else:
            note = r["issues"][:44]
        print(f"{r['qi']:<3} {r['mode']:<5} {jv:<5} {st:<7} {r['n_steps']:<6} {note}")

    # The two headline numbers the build task asks for, per mode.
    print()
    for mode in ("raw", "json"):
        subset = [r for r in rows if r["mode"] == mode]
        valid = sum(1 for r in subset if r["json_valid"])
        sensible = sum(1 for r in subset if r["structurally_ok"])
        print(
            f"{mode:<5}: valid JSON = {valid}/{len(subset)}   "
            f"structurally sensible plan = {sensible}/{len(subset)}"
        )

    # Failure-pattern breakdown across all 10 rows: count each category so the
    # report names the *dominant* failure mode (the task asks for "any patterns in
    # how it failed: malformed JSON, wrong tool names, missing required args, etc.").
    # A single row can fall into more than one structural bucket (e.g. wrong tool +
    # missing arg), so these need not sum to 10 — they're counts of occurrences.
    print()
    print("failure patterns (across all 10 rows; a row can hit >1):")
    patterns: dict[str, int] = {
        "malformed JSON (no object found / parse error)": 0,
        "wrong/unknown tool name": 0,
        "missing required arg (query)": 0,
        "unexpected/extra arg": 0,
        "malformed step (not an object / bad key types)": 0,
        "valid JSON but empty steps (no tool calls)": 0,
        "structurally sensible (non-empty)": 0,
    }
    for r in rows:
        if not r["json_valid"]:
            patterns["malformed JSON (no object found / parse error)"] += 1
            continue
        if not r["structurally_ok"]:
            iss = r["issues"]
            if "unknown_tool" in iss:
                patterns["wrong/unknown tool name"] += 1
            if "missing" in iss:
                patterns["missing required arg (query)"] += 1
            if "unexpected" in iss:
                patterns["unexpected/extra arg"] += 1
            if "malformed" in iss:
                patterns["malformed step (not an object / bad key types)"] += 1
            continue
        # structurally OK
        if r["n_steps"] == 0:
            patterns["valid JSON but empty steps (no tool calls)"] += 1
        else:
            patterns["structurally sensible (non-empty)"] += 1
    for p, n in patterns.items():
        print(f"  {n:>2}  {p}")

    print()
    print("done.")


if __name__ == "__main__":
    main()
