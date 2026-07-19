"""Planner agent for the RepoLens agent layer (SDD Â§9.1).

The first agent, built **standalone before LangGraph** is wired (SDD Â§15): given a
user question about a repo and a description of the available tools, ask the local
LLM (Ollama) to pick which tool(s) to call and with what arguments, and return a
**structured JSON plan** (not free text) in the SDD Â§9.1 shape
``{"steps": [{"agent": "search", "tool": "search_symbols", "args": {...}}, ...]}``.
Downstream agents (Search/Context) then execute that plan; the Synthesizer turns
the collected facts into the user-facing answer (SDD Â§9.1/Â§9.4).

Why the Planner exists, per SDD Â§9.1 and Â§18: local models are noticeably better at
picking from a *constrained menu of tool calls* than at open-ended repo reasoning,
and forcing a structured plan up front keeps every later step narrow. The Â§18 risk
("local model quality is inconsistent at tool-selection and JSON-formatting") is
mitigated by (a) a tight, single-purpose prompt that names exactly three tools with
their signatures, (b) an optional Ollama ``format:"json"`` constraint that forces
syntactically-valid JSON, (c) optional retry-on-parse-failure, and (d) ``think=False``
by default -- which disables the model's chain-of-thought so it can't burn its
``num_predict`` budget reasoning and then emit an empty answer. (The root-cause fix
for the demo's Q3 json-mode empty-response failure: confirmed in
:mod:`scripts.planner_diag`, disabling thinking is ~4x faster and drops the
empty-rate to 0; a retry never rescues budget exhaustion, and the ``/no_think`` prompt
directive actually made it worse.) This module exposes all four levers, with
``think`` off by default; the exploratory demo (:mod:`scripts.planner_demo`) leaves
the prompt/format/retry levers at a *raw* baseline first to honestly measure the
model, then toggles JSON-mode
to see whether the Â§18 lever rescues it.

**HTTP to Ollama is stdlib ``urllib`` only â€” no ``ollama`` / ``langchain-ollama`` /
``httpx`` dependency.** The project lists ``langchain-ollama`` in ``requirements.txt``
for the later LangGraph wiring, but pinning the Planner to a single HTTP client early
would couple it to whichever lib is actually installed; Ollama's REST API
(``POST /api/chat``) is a small JSON POST and ``urllib.request`` handles it with zero
third-party imports. Same "keep it small, independently testable" ethos as the Â§10
tools (SDD Â§18 risk row: "Keep tools/agents small and independently testable").

Config is read **directly here, not imported from** :mod:`app.db`: agents run
per-query and should not pull the SQLAlchemy engine / a Postgres connection just to
reach Ollama (SDD Â§15 opens one session per query in the future graph). Reuses the
same env var names / defaults (``OLLAMA_HOST``) and adds ``OLLAMA_MODEL``.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# â”€â”€â”€ config (decoupled from app.db) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ollama host â€” same name/default as app.db.OLLAMA_HOST so a single .env entry
# serves both; read here so importing the Planner never touches SQLAlchemy.
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Default local model for the orchestrator's LLM hops (SDD Â§9). The brief names
# "Qwen3/Gemma/Llama" as the local-model tier; qwen3:8b is a reasonable default the
# user must have pulled with `ollama pull qwen3:8b` for these defaults to work â€”
# if a different model is installed, set OLLAMA_MODEL (or pass ``model=`` in).
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Per-call wall-clock timeout (seconds). Local 7â€“14B models on CPU can be slow; this
# bounds a single planning round trip so a stuck generation cannot hang the caller
# (the future graph would treat a timeout as a Planner failure â†’ replan / fallback).
DEFAULT_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "60"))

# â”€â”€â”€ the three MVP tools the Planner chooses between (SDD Â§10) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Described to the model by NAME = the real callable name in app.tools, so a plan
# the Planner emits is directly dispatchable later (the orchestrator looks up the
# symbol by this name). ONLY the user-chosen args appear in ``args``: ``repository_id``
# and the injected ``session`` are orchestrator concerns, NOT something the model
# picks â€” matching SDD Â§9.1's example ``{"args": {"name": "authenticate"}}`` which
# carries a single user-relevant arg and no plumbing.
#
# NOTE: SDD Â§9.1's example writes ``"tool": "symbol_search"`` (noun-first); the real
# tool functions are ``search_symbols`` / ``search_files`` / ``search_text`` (verb-
# first, SDD Â§10 / the code). The Planner emits the real names so the plan is
# executable; the SDD shape (``{steps:[{agent,tool,args}]}``) is what's matched.

# Required args per tool â€” the structural check (:func:`_validate_plan`) flags a
# step whose ``args`` omits a required key. Every tool requires ``query`` (a search
# with no query is not a search â€” and the tools themselves return ``[]`` on empty
# query). ``repository_id`` and the injected ``session`` are orchestrator concerns,
# never the model's, so they are NOT listed here (SDD Â§9.1's example ``args`` carries
# only the user-relevant arg, no plumbing).
KNOWN_TOOLS: dict[str, frozenset[str]] = {
    "search_symbols": frozenset({"query"}),
    "search_files": frozenset({"query"}),
    "search_text": frozenset({"query"}),
}
# Optional args per tool â€” may appear in ``args`` but aren't required. Surfaced as a
# set so the validator can flag an *unexpected* arg key (one in neither required nor
# optional) rather than just "missing required".
OPTIONAL_TOOL_ARGS: dict[str, frozenset[str]] = {
    "search_symbols": frozenset({"kind"}),
    "search_files": frozenset(),
    "search_text": frozenset({"regex"}),
}

# Default tool catalog prose handed to the model â€” describes purpose + signature +
# when-to-use. Kept stable so the Planner's behavior is comparable run-to-run; the
# demo hard-codes the same set rather than reading app.tools, to keep the Planner
# standalone (no SQLAlchemy import to read tool metadata, and the names must stay
# in lock-step with KNOWN_TOOLS above).
DEFAULT_TOOLS_DESCRIPTION = """You have exactly three search tools. Pick tool(s) and arguments for each step. You do NOT know the repository_id or any session â€” those are injected by the system automatically; NEVER include them in args.

1. search_symbols
   Purpose: find a class, function, method, or variable BY NAME in the indexed symbol table.
   Use when: the question asks "where is X defined", "find the class/function/symbol named X", "find the definition of X".
   args:
     - query  (REQUIRED, string): the symbol's name or a partial name. Case-insensitive substring match. e.g. "Blueprint", "Flask", "lueprint".
     - kind   (OPTIONAL, string): one of "class", "function", "method", "variable" to restrict the kind. Omit to return all kinds.

2. search_files
   Purpose: find a FILE by its path or filename fragment.
   Use when: the question mentions a specific file by name, or asks "which file", "where is the file", "find the file whose path contains X".
   args:
     - query  (REQUIRED, string): a filename or path fragment. Case-insensitive substring match against the file path. e.g. "app.py", "routes", "session".

3. search_text
   Purpose: search for FREE TEXT across the repo's Python source â€” matches anywhere in a line: code, comments, string literals. Literal match by default (not a regex, not whole-word).
   Use when: the question asks "what files mention/reference/talk about X", "where is the phrase X used", or you are looking for a concept/phrase rather than a symbol name.
   args:
     - query  (REQUIRED, string): the text to search for. e.g. "url rule", "before_request", "session".
     - regex  (OPTIONAL, boolean): set true only if the query should be treated as a regular expression. Default false.

Output rules:
- Emit ONLY a JSON object, no markdown, no code fences, no prose before or after.
- The object has a single key "steps": a list (possibly empty) of steps, in execution order.
- Each step is {"agent": "search", "tool": <one of the three tool names above>, "args": {... only the user args above ...}}.
- Prefer the single most-targeted tool. Use multiple steps only when the question clearly needs more than one search.
- If the question is unanswerable with these three tools, emit {"steps": []}.

Example output:
{"steps": [{"agent": "search", "tool": "search_symbols", "args": {"query": "Blueprint", "kind": "class"}}]}
"""


# â”€â”€â”€ result types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class StepIssues:
    """Per-step structural problems, for the demo's failure-pattern report.

    Each list holds human-readable strings; empty means the step is structurally
    clean. Captured separately from the plan object so a caller can report *which*
    way a step failed (wrong tool name vs missing required arg vs unexpected arg)
    without re-deriving it.
    """

    unknown_tool: list[str] = field(default_factory=list)
    missing_required_args: list[str] = field(default_factory=list)
    unexpected_args: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)  # not a dict / wrong key types


@dataclass
class PlanValidation:
    """Structural validation of a parsed plan, decoupled from JSON-syntactic parse.

    ``json_valid`` only answers "did it parse as JSON"; this answers "is the *shape*
    right" â€” top-level ``steps`` list, each step a dict with ``agent``/``tool``/
    ``args``, ``tool`` one of the known names (``KNOWN_TOOLS``), and required args
    present. The demo uses ``structurally_ok`` as the "sensible plan" heuristic and
    the per-step ``issues`` to categorize *how* it failed (the report-back the task
    asks for: wrong tool names, missing required args, etc.).
    """

    has_steps_key: bool = False
    steps_is_list: bool = False
    n_steps: int = 0
    per_step: list[StepIssues] = field(default_factory=list)
    structurally_ok: bool = False

    def summary(self) -> str:
        """One-line human summary of the structural validation, for the demo."""
        flags: list[str] = []
        if not self.has_steps_key:
            flags.append("no 'steps' key")
        elif not self.steps_is_list:
            flags.append("'steps' not a list")
        for i, si in enumerate(self.per_step):
            for m in si.unknown_tool:
                flags.append(f"step{i}:unknown_tool={m}")
            for m in si.missing_required_args:
                flags.append(f"step{i}:missing={m}")
            for m in si.unexpected_args:
                flags.append(f"step{i}:unexpected={m}")
            for m in si.malformed:
                flags.append(f"step{i}:malformed={m}")
        return ("ok" if self.structurally_ok else " | ".join(flags)) or "ok"


@dataclass
class PlannerResult:
    """Everything the demo needs to judge a single planning call.

    ``raw_response`` is the model's unmodified text (the "raw JSON the model
    produces" the demo prints). ``plan`` is the parsed object if JSON succeeded,
    else ``None``. ``json_valid`` / ``parse_error`` separate JSON-syntactic failure
    from structural mismatch: a model can emit valid JSON of the *wrong* shape, and
    that distinction is the whole point of the exploration (which failure mode
    dominates on a weak local model?).
    """

    question: str
    model: str
    mode: str  # "raw" | "json" â€” what format constraint was sent to Ollama
    raw_response: str
    plan: dict[str, Any] | None
    json_valid: bool
    parse_error: str | None
    validation: PlanValidation
    retries_used: int
    wall_time_s: float


# â”€â”€â”€ pure helpers (no I/O) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _strip_to_json_object(text: str) -> str | None:
    """Best-effort recover a JSON object from a model output that isn't pure JSON.

    Weak local models frequently wrap output despite being told not to: markdown
    fences (`````json\n{...}\n`````), a sentence before the object ("Sure! Here is
    your plan:"), or trailing commentary. Returns the candidate substring or
    ``None`` if no plausible object spans the text. Intentionally tolerant: the
    demo's whole purpose is to measure how often this rescue is *needed*, so the
    raw text is preserved separately in the result and this only feeds the parse
    attempt â€” if the candidate still fails ``json.loads``, the caller records the
    real parse error.

    Not a true brace-matcher (no string/comment awareness): the strategy is to
    grab the **outermost** ``{`` â€¦ ``}`` span, which is correct for a *single*
    top-level object. Three cases, in order:

    * **bare object** â€” stripped text already starts ``{`` and ends ``}`` â†’ return it.
    * **fenced** (```` ```json ... ``` ````) â€” a greedy ``\\{.*\\}`` bounded by the
      closing fence grabs the whole object *across nested braces*. (Non-greedy here
      would truncate at the first inner ``}`` of ``{"steps":[{"args":{...}}]}`` and
      silently corrupt the parse â€” the bug this guards against.) The closing-fence
      anchor handles a stray ``{`` in trailing prose that the bare-span below could
      not.
    * **outermost span** (prose before/after, or a fence without its closing
      ```` ``` ````) â€” first ``{`` to last ``}`` in the text. ``json.loads`` is the
      final arbiter: if this candidate isn't valid JSON, the caller records the
      parse error rather than us guessing fields.
    """
    if not text:
        return None
    # Fast path: already a bare JSON object.
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # Stripped of a ```json ... ``` fence. GREEDY {.*} (not .*?) so nested braces in
    # a plan like {"steps":[{"args":{"query":"x"}}]} are captured whole â€” the
    # non-greedy variant stops at the first inner } and returns invalid JSON.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Outermost {...} span (tolerates prose before/after, including a fence
    # without the closing ``` or a leading "Here is the plan: {...}").
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _validate_plan(plan: Any) -> PlanValidation:
    """Structurally validate a parsed object against the SDD Â§9.1 shape and the
    known-tools table. *Pure*: takes the parsed object, returns diagnostics â€” no
    I/O. Separating this from ``_parse`` keeps "is it JSON?" (parse) and "is it the
    right JSON?" (this) distinct, which is the failure-mode split the demo reports.
    """
    v = PlanValidation()
    if not isinstance(plan, dict):
        return v  # has_steps_key stays False â†’ structurally_ok False
    v.has_steps_key = "steps" in plan
    steps = plan.get("steps")
    v.steps_is_list = isinstance(steps, list)
    if not v.steps_is_list:
        return v
    v.n_steps = len(steps)
    all_ok = True
    for step in steps:
        si = StepIssues()
        if not isinstance(step, dict):
            si.malformed.append("step is not an object")
            v.per_step.append(si)
            all_ok = False
            continue
        tool = step.get("tool")
        args = step.get("args", {})
        if not isinstance(tool, str):
            si.malformed.append("tool is not a string")
        if not isinstance(args, dict):
            si.malformed.append("args is not an object")
            args = {}  # avoid KeyError below
        if isinstance(tool, str) and tool not in KNOWN_TOOLS:
            si.unknown_tool.append(tool)
        # required / unexpected args (only meaningful once tool is known+args a dict)
        if isinstance(tool, str) and tool in KNOWN_TOOLS and isinstance(args, dict):
            present = set(args.keys())
            for req in KNOWN_TOOLS[tool]:
                if req not in present:
                    si.missing_required_args.append(req)
            allowed = KNOWN_TOOLS[tool] | OPTIONAL_TOOL_ARGS.get(tool, frozenset())
            for key in present - allowed:
                si.unexpected_args.append(key)
        if si.unknown_tool or si.missing_required_args or si.unexpected_args or si.malformed:
            all_ok = False
        v.per_step.append(si)
    # An empty steps list IS structurally valid (SDD Â§9.1 allows it for
    # unanswerable questions); ``all_ok`` initializes True and stays so with no steps.
    v.structurally_ok = v.has_steps_key and v.steps_is_list and all_ok
    return v


# â”€â”€â”€ the LLM call + parse (the tool the orchestrator/demo calls) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class OllamaError(RuntimeError):
    """Anything wrong reaching/parsing Ollama's HTTP response (infra): the model
    not running, an unknown model name, a non-200, a malformed response body. Kept
    separate from the JSON-parse / structural errors so a caller can tell "model
    produced bad plan" from "couldn't reach the model."""


def _chat_payload(
    question: str,
    tools_description: str,
    *,
    model: str,
    fmt: str | dict | None,
    temperature: float,
    max_tokens: int,
    think: bool = False,
) -> dict[str, Any]:
    """Build the JSON body for Ollama ``POST /api/chat``. Pure (no I/O), split out
    so it can be inspected / unit-tested without a network call.

    ``fmt`` maps to Ollama's ``format`` param: ``None`` = unconstrained generation
    (the honest raw baseline), ``"json"`` = force syntactically-valid JSON (the
    Â§18 lever â€” still free *shape*, only syntactically valid). Passing a JSON-Schema
    dict is what would constrain the *shape*; left for a later, post-demo iteration
    (the demo's job is to decide *whether* that heavier lever is worth it).

    ``think`` maps to the top-level Ollama ``/api/chat`` ``think`` param. ``False``
    (the default) disables Qwen3-style chain-of-thought: under a ``num_predict``
    budget the model otherwise spends all its tokens reasoning and emits an *empty*
    ``message.content`` (``done_reason="length"``), which looks like a JSON-parse
    failure but is really budget exhaustion. ``think=False`` is ~4x faster and clears
    that failure (confirmed in :mod:`scripts.planner_diag`). Exposed as a parameter
    (not hardcoded) so a future agent role doing harder reasoning can opt back in
    with ``think=True``.
    """
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        # think=False disables chain-of-thought: under a num_predict budget Qwen3
        # otherwise burns all its tokens reasoning and emits an EMPTY message.content
        # (done_reason="length"), which looks like a JSON-parse failure but is really
        # budget exhaustion. Top-level /api/chat param (NOT under `options`). Confirmed
        # ~4x faster + 0 empty responses in scripts/planner_diag.py.
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the Planner for RepoLens, a tool that helps people "
                    "understand an unfamiliar codebase. Your ONLY job is to read the "
                    "user's question and decide which search tool(s) to call and with "
                    "what arguments. You will NOT answer the question yourself and you "
                    "will NOT look at code. You emit a JSON plan; another agent executes "
                    "it. Follow the tool descriptions and output rules exactly."
                ),
            },
            {
                "role": "user",
                "content": f"{tools_description}\n\nUser question:\n{question}",
            },
        ],
    }
    if fmt is not None:
        body["format"] = fmt
    return body


def _ollama_chat(
    payload: dict[str, Any],
    *,
    host: str,
    timeout: float,
) -> str:
    """POST to Ollama ``/api/chat`` and return the assistant message text.

    Raises :class:`OllamaError` on any transport / HTTP / shape problem so the
    caller can distinguish *infra* failure (Ollama down, bad model) from *model*
    failure (bad JSON) in its report â€” the exploration needs that split. Uses
    stdlib :mod:`urllib.request` (no third-party HTTP client) so the Planner runs in
    any venv, including one where ``ollama``/``httpx``/``langchain-ollama`` are not
    yet installed.
    """
    url = host.rstrip("/") + "/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 â€” best-effort body capture
            pass
        raise OllamaError(
            f"Ollama HTTP {exc.code} at {url}: {body_text[:300] or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        # Connection refused / host unreachable â€” Ollama not running. Distinct
        # message so the demo can report "Ollama unreachable" as its own category.
        raise OllamaError(
            f"Ollama unreachable at {url} (is `ollama serve` running?): {exc.reason}"
        ) from exc
    except TimeoutError as exc:  # urlopen raises this on the timeout arg, not URLError
        raise OllamaError(
            f"Ollama timed out after {timeout}s at {url}: {exc}"
        ) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama returned non-JSON HTTP body: {exc}") from exc

    # /api/chat with stream=False returns {message: {role, content}, ...}.
    message = parsed.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise OllamaError(f"Ollama response missing message.content: {raw[:300]}")
    content = message.get("content")
    if not isinstance(content, str):
        raise OllamaError(f"Ollama message.content is not a string: {raw[:300]}")
    return content


def plan(
    question: str,
    tools_description: str = DEFAULT_TOOLS_DESCRIPTION,
    *,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
    fmt: str | dict | None = None,
    think: bool = False,
    retries: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 400,
    timeout: float = DEFAULT_TIMEOUT,
) -> PlannerResult:
    """Ask the local LLM to plan tool calls for ``question`` and return a
    :class:`PlannerResult`.

    Defaults are the **honest raw baseline** the task asks for: ``fmt=None`` (do NOT
    force JSON â€” measure how the model does on its own), ``retries=0`` (no
    rescue-on-parse-failure â€” measure raw reliability, not mitigated), and
    ``temperature=0.0`` for reproducibility across the 5 demo questions. The demo
    (:mod:`scripts.planner_demo`) calls this once per question in *raw* (defaults)
    and once in *json-forced* (``fmt="json"``) to compare whether the SDD Â§18 lever
    rescues the model.

    The ``fmt`` / ``retries`` / ``temperature`` params ARE the tuning levers SDD
    Â§18 names ("Constrain Planner output with a strict JSON schema +
    retry-on-parse-failure; keep prompts small and single-purpose"); left here,
    defaulted open, so the *next* iteration can enable them based on what the demo
    shows â€” e.g. if raw parses <â…” of the time, wire ``fmt="json"`` + ``retries=1``.

    On retry: a JSON-parse failure re-prompts once (when ``retries``>0) appending
    the prior output + the parse error and asking the model to emit *only* valid
    JSON â€” the SDD Â§18 "retry-on-parse-failure" lever, bounded to avoid looping on a
    weak model (SDD Â§15's one-loop temptation generalized). ``retries_used`` records
    how many were spent.

    ``think`` defaults to ``False``: Qwen3-style chain-of-thought is OFF for
    planning. Under a ``num_predict`` budget the model otherwise spends all its
    tokens reasoning and emits an *empty* answer (``done_reason="length"``), which
    this function would misreport as a JSON-parse failure and then retry
    fruitlessly -- the retry never rescues it, since re-thinking re-empties the
    budget. ``think=False`` is ~4x faster and clears the empty-response failure
    (confirmed in :mod:`scripts.planner_diag`). Exposed as a parameter so a future
    agent role doing harder reasoning can pass ``think=True``.
    """
    mode = "raw" if fmt is None else ("json" if fmt == "json" else "schema")
    payload = _chat_payload(
        question,
        tools_description,
        model=model,
        fmt=fmt,
        think=think,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    last_raw: str = ""
    last_parse_error: str | None = None
    plan_obj: dict[str, Any] | None = None
    json_valid = False
    retries_used = 0
    start = time.perf_counter()

    # 1 initial attempt + up to `retries` rescue rounds. Each rescue appends the
    # prior (unparseable) output as an assistant turn and a user repair turn, then
    # re-asks â€” the SDD Â§18 "retry-on-parse-failure" lever, bounded by the loop so a
    # weak model cannot spin forever (SDD Â§15's one-loop concern generalized).
    for attempt in range(retries + 1):
        if attempt > 0:
            payload = {
                **payload,
                "messages": payload["messages"]
                + [
                    {"role": "assistant", "content": last_raw},
                    {
                        "role": "user",
                        "content": (
                            f"That did not parse as JSON: {last_parse_error}. "
                            "Output ONLY the JSON plan object now, no other text."
                        ),
                    },
                ],
            }
            retries_used = attempt
        last_raw = _ollama_chat(payload, host=host, timeout=timeout)
        candidate = _strip_to_json_object(last_raw)
        if candidate is None:
            last_parse_error = "no JSON object found in output"
            plan_obj = None
            json_valid = False
            continue  # try a rescue round if any remain
        try:
            plan_obj = json.loads(candidate)
            json_valid = True
            last_parse_error = None
            break  # syntactically valid JSON â€” stop; structural validation is separate
        except json.JSONDecodeError as exc:
            last_parse_error = str(exc)
            plan_obj = None
            json_valid = False
            continue

    end = time.perf_counter()
    return PlannerResult(
        question=question,
        model=model,
        mode=mode,
        raw_response=last_raw,
        plan=plan_obj,
        json_valid=json_valid,
        parse_error=None if json_valid else last_parse_error,
        validation=_validate_plan(plan_obj),
        retries_used=retries_used,
        wall_time_s=end - start,
    )


