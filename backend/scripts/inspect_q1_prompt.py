"""THROWAWAY inspection: print the EXACT prompt the Synthesizer builds for a question.

Reproduces one of the synthesizer_demo question paths up to (but NOT including)
the Ollama synthesis call: Planner (fmt=json, think=False, temperature=0) ->
Search Agent dispatch -> :func:`_format_hits` -> :func:`_chat_payload`, then
prints the LITERAL ``messages[0]`` (system) + ``messages[1]`` (user) that
:func:`app.agents.synthesizer.synthesize` would POST to Ollama. No answer is
generated (no second Ollama call) -- the point is to see the prompt text, not to
answer the question.

Usage -- the question is a CLI arg so this is reusable across the demo's
questions (not a Q1 one-off). Accepts either a 1-based index into the demo's
QUESTIONS list OR a literal question string, defaulting to Q1::

    PYTHONPATH=. python scripts/inspect_q1_prompt.py          # Q1 (default)
    PYTHONPATH=. python scripts/inspect_q1_prompt.py 3         # Q3 by index
    PYTHONPATH=. python scripts/inspect_q1_prompt.py "where is ..."  # literal

Purpose: the 3/3 demo answers were grounded (zero invented citations -- good)
but at first looked narrow (each cited only ~1 of the several real hits it was
given). On inspecting Q1's hits that proved mostly correct selectivity (the
model correctly excluded *different* same-prefix classes like BlueprintError),
with the one real gap being a dropped *same-name* match (blueprints.py vs
sansio/blueprints.py). This script lets us confirm per-question whether a narrow
answer is (a) correct exclusion of non-matching classes -- keep, or (b) a
dropped same-name real match -- the actual synthesis-quality issue worth
prompt-tuning. Kept general so we can compare Q1 vs Q3 against the SAME tool.

Preconditions: Ollama running with a model (override via OLLAMA_MODEL), Postgres
up with flask indexed (same as synthesizer_demo.py).
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import sqlalchemy as sa  # noqa: E402

from app.agents.planner import OLLAMA_HOST, OLLAMA_MODEL, OllamaError, plan  # noqa: E402
from app.agents.search_agent import SearchDispatchError, build_search_step_result  # noqa: E402
from app.agents.synthesizer import (  # noqa: E402
    DEFAULT_MAX_HITS,
    DEFAULT_MAX_TOKENS,
    _chat_payload,
    _format_hits,
)
from app.db import SessionLocal  # noqa: E402
from app.models import File, Repository, Symbol  # noqa: E402

# The SAME 3 questions as synthesizer_demo.py / planner_search_chain_demo.py,
# reused verbatim so this inspection is faithful to the demo run. Indexed 1-3
# from the CLI; default Q1 (the Blueprint question where the same-name-gap, if
# real, lives). A literal question string on the CLI bypasses the index.
QUESTIONS: list[str] = [
    "where is the Blueprint class defined",
    "what files mention URL rules",
    "which file contains the Flask class",
]

SEP = "=" * 78


def _resolve_model(host: str) -> str:
    """Pick a model: OLLAMA_MODEL env if set, else first installed via /api/tags,
    else the Planner default. Same resolution as synthesizer_demo._resolve_model."""
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
    """Readiness probe -- mirrors synthesizer_demo._flask_repo_id. Returns the
    flask repo_id or None (caller aborts cleanly)."""
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


def _box(title: str, text: str) -> None:
    """Print a labeled, delimited block so the prompt's boundaries/whitespace are
    visible (the whole point: see the EXACT bytes, including blank lines)."""
    print(SEP)
    print(title)
    print(SEP)
    print(text)
    print()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    # Resolve the question from argv: a 1-based index into QUESTIONS, OR a
    # literal question string, OR Q1 by default. An out-of-range index errors
    # loudly (not silently coerced) so a typo doesn't inspect the wrong question.
    arg = sys.argv[1] if len(sys.argv) > 1 else "1"
    if arg.strip().isdigit():
        idx = int(arg)
        if not (1 <= idx <= len(QUESTIONS)):
            print(f"index {idx} out of range (1-{len(QUESTIONS)}); questions:")
            for i, q in enumerate(QUESTIONS, 1):
                print(f"  {i}: {q!r}")
            raise SystemExit(2)
        question, label = QUESTIONS[idx - 1], f"Q{idx}"
    else:
        question, label = arg, "custom"

    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    model = _resolve_model(host)
    print(f"Inspecting the Synthesizer prompt for {label}: {question!r}")
    print(f"  Ollama host : {host}   model: {model}")
    print(f"  Planner cfg : fmt='json', think=False, temperature=0 (matches the demo)")
    print(f"  Synth cfg   : max_hits={DEFAULT_MAX_HITS}, max_tokens={DEFAULT_MAX_TOKENS}")
    print()

    repo_id = _flask_repo_id()
    if repo_id is None:
        print("flask repo is not indexed in Postgres. Run scripts/run_walker_once.py +")
        print("index_all_flask_symbols.py first. Aborting.")
        raise SystemExit(1)
    print(f"  flask repo_id: {repo_id}")
    print()

    # ── stage 1: Planner (same call as the demo) ──────────────────────────
    try:
        res = plan(question, host=host, model=model, fmt="json", retries=0, temperature=0.0)
    except OllamaError as exc:
        print(f"PLANNER OLLAMA ERROR: {exc}")
        print("(start `ollama serve`; the demo already ran, so Ollama should be up)")
        raise SystemExit(1) from exc

    print(SEP)
    print("Planner: parsed plan")
    print(SEP)
    print(json.dumps(res.plan, indent=2) if res.plan is not None else "(JSON parse failed)")
    print()

    if not res.json_valid or not isinstance(res.plan, dict):
        print("Planner produced no parseable plan -- nothing to inspect.")
        raise SystemExit(1)
    steps = res.plan.get("steps")
    if not isinstance(steps, list) or not steps:
        print("Planner emitted an empty steps list -- nothing to inspect.")
        raise SystemExit(1)
    step = steps[0]

    # ── stage 2: Search Agent dispatch (same call as the demo) ────────────
    try:
        with SessionLocal() as session:
            result = build_search_step_result(step, repo_id, session)
    except SearchDispatchError as exc:
        print(f"SEARCH AGENT DISPATCH ERROR: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"SEARCH AGENT TOOL ERROR ({type(exc).__name__}): {exc}")
        raise SystemExit(1) from exc

    print(SEP)
    print(f"Search Agent: tool={result.tool}  args={result.args}  total hits={len(result.hits)}")
    print(SEP)
    # The raw hit list -- so we can see the FULL grounding set (how many hits
    # the model was actually given), not just the formatted block.
    for i, h in enumerate(result.hits, 1):
        print(f"  hit {i}: {h!r}")
    print()

    # ── stage 3: build the EXACT prompt synthesize() would send ───────────
    hits_block, ground = _format_hits([result], max_hits=DEFAULT_MAX_HITS)
    payload = _chat_payload(
        question,
        hits_block,
        model=model,
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        think=False,
    )
    messages = payload["messages"]

    print(SEP)
    print(f"hits block -> {len(ground)} distinct ground paths: {ground}")
    print(SEP)
    print()

    # The literal system message -- verbatim, the exact bytes sent.
    _box("PROMPT: messages[0] (system) -- VERBATIM", messages[0]["content"])
    # The literal user message -- verbatim. This is "Search hits:\n{hits_block}
    # \n\nUser question:\n{question}", so its structure is what steers the model.
    _box("PROMPT: messages[1] (user) -- VERBATIM", messages[1]["content"])

    print("done.")


if __name__ == "__main__":
    main()
