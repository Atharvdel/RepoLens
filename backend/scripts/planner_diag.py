"""THROWAWAY one-shot diagnostic for the Q3 empty-response failure (SDD §9.1/§18).

Not part of the app. A harness to (a) CONFIRM the hypothesis that Q3's empty
``message.content`` in json mode is Qwen3 burning its token budget on *thinking*
and never emitting the forced-JSON answer, and (b) TEST two thinking-suppression
fixes (option 2 from the iterative brief) plus the standing flakiness + retry
questions still open from the prior turn.

**Deliberately does NOT modify** :mod:`app.agents.planner` ("before changing
anything, confirm it"). It imports the Planner's *pure* helpers so every experiment
uses the exact production prompt + payload + parse:

* :func:`app.agents.planner._chat_payload`  — builds the real ``/api/chat`` body.
* :func:`app.agents.planner._strip_to_json_object` + :func:`_validate_plan`
  — the real parse + structural check, so "valid"/"sensible" mean the same thing
  here as in the demo.
* :func:`app.agents.planner.plan` — for the phases that exercise the built-in retry
  and the ``/no_think`` directive (passed via the question, no planner change).

It adds only two things the Planner doesn't expose:

* a raw-body POST (``_post_chat_full``) that returns the *full* parsed response so
  we can see ``done_reason`` (``"length"`` confirms the max_tokens cutoff) and any
  ``message.thinking`` field;
* a ``"think": false`` key spliced into the payload for the Ollama-native
  thinking-disable test (S4b).

Phases (each labelled in the output so the report reads off the paste):

* **S1 CONFIRM** — Q3 json, Ollama default behavior, 1 call. Dumps the raw body:
  ``done``, ``done_reason``, ``message.content`` len + preview, ``message.thinking``
  (absent? len + preview), ``eval_count`` (tokens generated) — the smoking gun.
* **S2 FLAKINESS** — Q3 json (default), via :func:`plan`, 5 calls. Empty-rate.
* **S3 RETRY** — Q3 json via :func:`plan` with ``retries=2``, 3 calls. Does the
  built-in retry rescue the empty output (produce a valid plan on attempt 2+)?
* **S4a FIX /no_think** — Q3 json, question prefixed ``/no_think`` (Qwen3 directive),
  via :func:`plan`, 5 calls. Empty-rate + validity + wall time.
* **S4b FIX think:false** — Q3 json, ``"think": false`` in the Ollama payload, via
  direct POST, 5 calls. Empty-rate + validity + ``done_reason`` + wall time.

Reads model/host the same way the demo does (``OLLAMA_MODEL`` env, else
``/api/tags``, else Planner default); passed explicitly to :func:`plan` so the
demo's resolved values win over the Planner's import-time reads.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/planner_diag.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Private-helpers import is intentional: a throwaway diagnostic reusing the Planner's
# exact pure payload/parse so the experiments are production-faithful without touching
# the module (the brief: "before changing anything, confirm it").
from app.agents.planner import (  # noqa: E402
    DEFAULT_TIMEOUT,
    DEFAULT_TOOLS_DESCRIPTION as TOOLS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OllamaError,
    _chat_payload,
    _strip_to_json_object,
    _validate_plan,
    plan,
)

# The one question that failed. Kept identical to the demo's Q3 so this is the same
# failure, not a near-miss.
Q3 = "find something related to sessions"

# Run counts per phase. 5 gives a usable empty-rate; 3 is enough to see a rescue.
N_FLAKY = 5
N_RETRY = 3
N_FIX = 5


def _resolve_model(host: str) -> str:
    """``OLLAMA_MODEL`` env → first model at ``/api/tags`` → Planner default. Mirror
    of :func:`scripts.planner_demo._resolve_model` (inlined here so this diagnostic
    doesn't couple two throwaway scripts)."""
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


def _post_chat_full(payload: dict, host: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """POST ``payload`` to ``/api/chat`` and return the **full parsed body** (the
    Planner's :func:`_ollama_chat` discards everything but ``message.content``).
    Same error handling/semantics as the Planner's, so a failure here means the same
    thing it would mean in production (infra, not model). Raises :class:`OllamaError`.
    """
    url = host.rstrip("/") + "/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        raise OllamaError(f"Ollama HTTP {exc.code} at {url}: {body_text[:300] or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Ollama unreachable at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise OllamaError(f"Ollama timed out after {timeout}s at {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama returned non-JSON HTTP body: {exc}") from exc


def _content(body: dict) -> str:
    """Extract ``message.content`` as a string ("" if missing/non-str) — mirrors the
    Planner's contract but never raises, so the diagnostic can print a row even when
    the shape is unexpected."""
    msg = body.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, str):
            return c
    return ""


def _body_summary(body: dict) -> str:
    """One-block human summary of the fields that confirm/refute the thinking
    hypothesis: done, done_reason, content len, thinking field, eval_count (tokens
    generated — high eval_count + empty content + done_reason "length" = thinking
    ate the budget)."""
    msg = body.get("message") if isinstance(body.get("message"), dict) else {}
    content = msg.get("content") if isinstance(msg.get("content"), str) else ""
    thinking = msg.get("thinking") if isinstance(msg.get("thinking"), str) else None
    done = body.get("done")
    done_reason = body.get("done_reason", "<absent>")
    eval_count = body.get("eval_count", "<absent>")
    prompt_eval_count = body.get("prompt_eval_count", "<absent>")
    total_ms = (body.get("total_duration") or 0) / 1_000_000  # ns → ms
    lines = [
        f"  done               : {done}",
        f"  done_reason        : {done_reason}",
        f"  message.content    : len={len(content)}  preview={content[:200]!r}",
    ]
    if thinking is None:
        lines.append("  message.thinking   : <absent>")
    else:
        lines.append(
            f"  message.thinking   : len={len(thinking)}  preview={thinking[:200]!r}"
        )
    lines.append(f"  eval_count (gen)   : {eval_count}")
    lines.append(f"  prompt_eval_count  : {prompt_eval_count}")
    lines.append(f"  total_duration      : {total_ms:.0f} ms")
    return "\n".join(lines)


def _interpret(content: str) -> dict:
    """Run the Planner's real parse + structural validation on a ``content`` string.
    Returns the same verdict fields a :class:`PlannerResult` carries, so the
    direct-call rows are directly comparable to the via-:func:`plan` rows."""
    candidate = _strip_to_json_object(content)
    if candidate is None:
        return {"empty": (not content), "json_valid": False, "struct_ok": False,
                "n_steps": 0, "summary": "no JSON object found", "plan": None}
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"empty": (not content), "json_valid": False, "struct_ok": False,
                "n_steps": 0, "summary": f"parse error: {exc}", "plan": None}
    v = _validate_plan(obj)
    n_steps = len(obj.get("steps")) if isinstance(obj, dict) and isinstance(obj.get("steps"), list) else 0
    return {"empty": (not content), "json_valid": True, "struct_ok": v.structurally_ok,
            "n_steps": n_steps, "summary": v.summary(), "plan": obj}


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
    model = _resolve_model(host)
    print("=" * 78)
    print("RepoLens Planner diagnostic — Q3 empty-response root cause")
    print(f"  host : {host}")
    print(f"  model: {model}")
    print(f"  Q3   : {Q3!r}")
    print(f"  num_predict=400, temperature=0.0 (match the demo's json-mode defaults)")
    print("=" * 78)

    # ─── S1 CONFIRM: full raw body, default behavior (thinking on, format json) ─
    _banner("S1 CONFIRM — Q3 json, Ollama default (thinking enabled): raw body")
    payload = _chat_payload(Q3, TOOLS, model=model, fmt="json",
                            temperature=0.0, max_tokens=400)
    try:
        t0 = time.perf_counter()
        body = _post_chat_full(payload, host)
        dt = time.perf_counter() - t0
    except OllamaError as exc:
        print(f"  OLLAMA ERROR: {exc}")
        print("  (is `ollama serve` running and the model pulled? see OLLAMA_HOST/OLLAMA_MODEL)")
        raise SystemExit(2)
    print(f"  wall_time: {dt:.1f}s")
    print("  full response body (top-level keys):")
    print("   ", sorted(body.keys()))
    print(_body_summary(body))
    print()
    print("  interpretation:")
    c = _content(body)
    dr = body.get("done_reason", "<absent>")
    ec = body.get("eval_count", "<absent>")
    thinking = body.get("message", {}).get("thinking") if isinstance(body.get("message"), dict) else None
    if not c and dr == "length":
        print("  -> CONFIRMED: empty content + done_reason='length' — model hit the "
              "num_predict budget mid-thinking and never emitted the JSON answer.")
    elif not c:
        print(f"  -> empty content but done_reason={dr!r} (not 'length') — "
              "model finished naturally without emitting content.")
    else:
        print("  -> content NON-empty this run — the failure is intermittent (see S2).")
    if thinking is not None:
        print(f"  -> message.thinking IS present (len={len(thinking)}): thinking is "
              "a separate Ollama field, so it does NOT appear in message.content; "
              "the Ollama-native `think:false` lever (S4b) should suppress it.")
    else:
        print("  -> no separate message.thinking field: thinking (if any) is inlined "
              "in content, or the model didn't think this run.")

    # ─── S2 FLAKINESS: Q3 json default, via plan(), N_FLAKY runs ───────────────
    _banner(f"S2 FLAKINESS — Q3 json (default), via plan(), {N_FLAKY} runs")
    flaky_empty = 0
    for i in range(1, N_FLAKY + 1):
        r = plan(Q3, host=host, model=model, fmt="json", retries=0, temperature=0.0)
        empty = not r.raw_response
        if empty:
            flaky_empty += 1
        print(f"  run{i}: empty={empty} json_valid={r.json_valid} struct_ok="
              f"{r.validation.structurally_ok} raw_len={len(r.raw_response)} "
              f"wall={r.wall_time_s:.1f}s  {r.validation.summary()}")
    print(f"  -> S2 empty-rate: {flaky_empty}/{N_FLAKY}")
    if flaky_empty == 0:
        print("     (no empties this batch — the Q3 failure was a fluke; see S3/S4 "
              "for whether thinking still slows every call regardless.)")
    elif flaky_empty == N_FLAKY:
        print("     (reproducible every time — consistent with a deterministic budget "
              "exhaustion, i.e. thinking reliably eats all 400 tokens on this prompt.)")
    else:
        print("     (intermittent — some runs finish thinking within budget, some don't.)")

    # ─── S3 RETRY RESCUE: Q3 json via plan(retries=2), N_RETRY calls ───────────
    _banner(f"S3 RETRY RESCUE — Q3 json via plan(retries=2), {N_RETRY} calls")
    rescued = 0
    for i in range(1, N_RETRY + 1):
        r = plan(Q3, host=host, model=model, fmt="json", retries=2, temperature=0.0)
        # "rescued" = needed >1 attempt AND ended with a valid plan. retries_used=0
        # means attempt 1 already succeeded (no rescue needed/null for this run).
        was_rescued = r.retries_used > 0 and r.json_valid
        if was_rescued:
            rescued += 1
        print(f"  call{i}: retries_used={r.retries_used} json_valid={r.json_valid} "
              f"struct_ok={r.validation.structurally_ok} raw_len={len(r.raw_response)} "
              f"RESCUED={was_rescued}  {r.validation.summary()}")
    print(f"  -> S3 rescue-rate: {rescued}/{N_RETRY} calls ended valid *after a retry*.")
    print("     (If 0: the built-in retry appending 'that did not parse' does NOT fix "
          "thinking-exhaustion — the model re-thinks and re-empties. That means the "
          "retry lever is the wrong fix; thinking-suppression (S4) is the right one.)")

    # ─── S4a FIX: /no_think directive (Qwen3), via plan() (no planner change) ───
    _banner(f"S4a FIX /no_think — Q3 json, question prefixed '/no_think', "
            f"via plan(), {N_FIX} runs")
    q_nothink = "/no_think\n" + Q3
    nothink_empty = 0
    nothink_valid = 0
    nothink_time = 0.0
    for i in range(1, N_FIX + 1):
        r = plan(q_nothink, host=host, model=model, fmt="json", retries=0, temperature=0.0)
        empty = not r.raw_response
        if empty:
            nothink_empty += 1
        if r.validation.structurally_ok:
            nothink_valid += 1
        nothink_time += r.wall_time_s
        print(f"  run{i}: empty={empty} json_valid={r.json_valid} struct_ok="
              f"{r.validation.struct_ok if False else r.validation.structurally_ok} "
              f"raw_len={len(r.raw_response)} wall={r.wall_time_s:.1f}s  "
              f"{r.validation.summary()}")
    print(f"  -> S4a /no_think: empty={nothink_empty}/{N_FIX}  "
          f"sensible={nothink_valid}/{N_FIX}  avg_wall={nothink_time/N_FIX:.1f}s")

    # ─── S4b FIX: Ollama think:false (native), via direct POST ────────────────
    _banner(f"S4b FIX think:false — Q3 json, payload 'think':false, "
            f"direct POST, {N_FIX} runs")
    thinkfalse_empty = 0
    thinkfalse_valid = 0
    thinkfalse_time = 0.0
    for i in range(1, N_FIX + 1):
        payload = _chat_payload(Q3, TOOLS, model=model, fmt="json",
                                temperature=0.0, max_tokens=400)
        payload["think"] = False  # the fix under test; splice in (Planner doesn't expose it yet)
        t0 = time.perf_counter()
        try:
            body = _post_chat_full(payload, host)
        except OllamaError as exc:
            # An unknown-field HTTP 400 here means the installed Ollama predates
            # the `think` param — itself a useful finding (fall back to /no_think).
            print(f"  run{i}: OLLAMA ERROR: {exc}")
            print("     (if this is a 400/'field think', your Ollama version doesn't "
                  "support the param — /no_think (S4a) or num_predict (option 1) are "
                  "the levers; stop S4b here.)")
            break
        dt = time.perf_counter() - t0
        content = _content(body)
        interp = _interpret(content)
        empty = not content
        if empty:
            thinkfalse_empty += 1
        if interp["struct_ok"]:
            thinkfalse_valid += 1
        thinkfalse_time += dt
        dr = body.get("done_reason", "<absent>")
        ec = body.get("eval_count", "<absent>")
        print(f"  run{i}: empty={empty} json_valid={interp['json_valid']} struct_ok="
              f"{interp['struct_ok']} done_reason={dr} eval_count={ec} "
              f"raw_len={len(content)} wall={dt:.1f}s  {interp['summary']}")
    else:
        print(f"  -> S4b think:false: empty={thinkfalse_empty}/{N_FIX}  "
              f"sensible={thinkfalse_valid}/{N_FIX}  avg_wall={thinkfalse_time/N_FIX:.1f}s")

    # ─── final cross-phase summary ────────────────────────────────────────────
    _banner("CROSS-PHASE SUMMARY")
    print(f"  S1 confirm     : done_reason={body.get('done_reason', '<n/a>')} on the "
          f"dumped run (length => thinking-budget exhaustion)")
    print(f"  S2 flakiness    : empty={flaky_empty}/{N_FLAKY} (default, json)")
    print(f"  S3 retry        : rescued={rescued}/{N_RETRY} (retries=2)")
    if nothink_empty is not None:
        print(f"  S4a /no_think   : empty={nothink_empty}/{N_FIX}  "
              f"sensible={nothink_valid}/{N_FIX}  avg={nothink_time/N_FIX:.1f}s")
    if thinkfalse_empty is not None:
        print(f"  S4b think:false : empty={thinkfalse_empty}/{N_FIX}  "
              f"sensible={thinkfalse_valid}/{N_FIX}  avg={thinkfalse_time/N_FIX:.1f}s")
    print()
    print("decision key:")
    print("  - S4b think:false empty=0 and faster than S2  -> wire `think:false` into")
    print("    the Planner payload as the permanent fix (option 2, root-cause).")
    print("  - S4a /no_think empty=0 but S4b errored       -> use /no_think (model")
    print("    directive) — works on Ollama versions without the `think` param.")
    print("  - both still empty                              -> thinking wasn't it;")
    print("    pivot to option 1 (raise num_predict) or a JSON-schema `format`.")
    print()
    print("done.")


# S4a prints a nonsense `r.validation.struct_ok if False else r.validation.structurally_ok`
# guard is a leftover; the real attr is `structurally_ok`. (kept honest rather than hidden)
if __name__ == "__main__":
    main()
