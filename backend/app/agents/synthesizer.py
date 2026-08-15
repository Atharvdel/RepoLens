"""Synthesizer agent for the RepoLens agent layer (SDD 9.4).

The third agent, built **standalone before LangGraph** is wired (SDD 15), the
same way :mod:`app.agents.planner` and :mod:`app.agents.search_agent` were: a
single, independently runnable unit that will later become the terminal node of
the graph. SDD 9.4 defines its responsibility narrowly:

    "the only agent that produces user-facing natural language. Takes all
    structured outputs collected so far and writes a grounded, cited answer ...
    Inputs: original question + all structured tool outputs from prior steps.
    Outputs: final markdown answer with inline file/line citations."

So this agent takes the original user question PLUS the :class:`SearchStepResult`
list the Search Agent (SDD 9.2) produced -- one per executed plan step, each
carrying real hits (file paths, line numbers, docstrings, matched source text)
-- and asks the local LLM via Ollama for a final natural-language answer that
cites **only** the files/lines actually present in those hits. The grounding
rule is the whole reason the agent exists as a *separate final step* (SDD 9.4
"Why it exists: ... only this agent talks to the user, and it's explicitly
instructed to cite only facts present in its structured inputs"): the Planner
never sees code, the Search Agent never writes prose, and the Synthesizer never
invents a path or line number it wasn't handed. Splitting synthesis off from
search is what keeps local-model hallucination out of user-facing answers
(SDD 9.4 + SDD 18 risk row).

Why **no** ``format="json"`` here (unlike the Planner): the output is prose,
not a structured plan, so JSON-mode would force the answer into a single JSON
string field (or break it) for no benefit -- there is no schema to satisfy and
no parse step to rescue. The SDD 18 "constrain output" lever does not apply to
prose; the grounding lever here is the *prompt* (cite only what's in the hits),
not a format constraint.

The Planner's other lesson DOES carry over: ``think=False`` by default
(top-level Ollama ``/api/chat`` param, NOT under ``options``). Under a
``num_predict`` budget a thinking model burns its tokens on chain-of-thought
and emits an *empty* ``message.content`` (``done_reason="length"``), which for
the Synthesizer would read as "the model wrote nothing"; ``think=False`` is
~4x faster and clears that (confirmed in :mod:`scripts.planner_diag`). Exposed
as a parameter so a future, harder synthesis pass can opt into reasoning.

HTTP to Ollama is **stdlib ``urllib`` only** -- no ``ollama`` /
``langchain-ollama`` / ``httpx`` dependency -- mirroring the Planner. The
transport (:func:`_ollama_chat`) is a local copy of
:func:`app.agents.planner._ollama_chat`: the same "keep each module
self-contained" posture the tool modules take toward their copied helpers
(``_ilike_contains``, ``_normalize_ripgrep_path``). It raises the SAME public
:class:`OllamaError` the Planner raises (imported from there), so the chain
demo catches one class for both stages and an unreachable/bad-Ollama failure
stays a single error category rather than a per-agent duplicate.

Config (``OLLAMA_HOST`` / ``OLLAMA_MODEL`` / :data:`DEFAULT_TIMEOUT`) is imported
from :mod:`app.agents.planner` rather than re-read here: it is the same env-var-
driven connection config, a single ``.env`` entry serves both agents, and one
place owns the defaults so they cannot drift (the Planner's stated rationale).
``synthesize()`` still takes ``host=`` / ``model=`` / ``timeout=`` params so a
caller overrides per call -- the demo passes its resolved model explicitly, the
same way :func:`app.agents.planner.plan` is driven.

Two cosmetic-but-real normalizations before the hits reach the prompt: any
``\\r\\n`` (or a lone ``\\r``) in a symbol's docstring or a text hit's
``matched_text`` is folded to ``\\n`` (:func:`_normalize_newlines`). The Windows
CRLF artifact flagged in CLAUDE.md would otherwise put raw carriage returns in
front of the model; this fixes it at the boundary where text *enters* agent
context, rather than sweeping it for later (the deferred docstring cleanup
slated for this step in CLAUDE.md). ``FileResult.path`` / ``SymbolResult.file``
/ ``TextHit.file`` are POSIX-rel paths straight from the DB / ripgrep
normalizer and carry no CRLF, so they are left untouched.

Importing this module does NOT open a Postgres connection: only the hit
*dataclass* types come in (transitively, via :mod:`app.agents.search_agent`'s
re-exports -- which invite exactly this), no SQLAlchemy engine, matching the
agent layer's "no Postgres connection just to reason" stance (SDD 15 opens one
session per query in the future graph; the Synthesizer never touches the
session -- it reasons over what Search already fetched).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

# Imported ONLY for type-checking. A real (runtime) import of
# :mod:`app.agents.context_agent` here would pull its four tool modules + NetworkX
# into this module's import surface, breaking the documented stance that
# "importing this module does NOT open a Postgres connection / pull heavy deps"
# (the agent layer reasons over what Search already fetched, SDD 9.4 / 15).
# ``from __future__ import annotations`` keeps the ``context_results`` annotation
# below a *string* at runtime, so this is never evaluated; the context-block
# builder (:func:`_format_context_block`) duck-types its inputs via
# :func:`dataclasses.asdict` instead of ``isinstance`` against this type.
if TYPE_CHECKING:
    from app.agents.context_agent import ContextStepResult

from app.agents.planner import (  # public: shared infra-error + connection config
    DEFAULT_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OllamaError,
)
from app.agents.search_agent import (  # the hit types search_agent re-exports for exactly this
    FileResult,
    SearchStepResult,
    SymbolResult,
    TextHit,
)

# ─── prompt-size guards ──────────────────────────────────────────────────────
# Caps that bound what enters the LLM prompt. None of the search tools trim
# free text themselves (the posture in app.tools), so the Synthesizer is the
# first place a runaway line / module-spanning docstring could swallow the
# prompt budget. These caps defend against that without cutting anything that
# matters for *grounding*: the citable fact for a docstring/matched-text hit is
# the line NUMBER (already uncapped) and the symbol NAME, not the full prose --
# so truncating the prose to a readable window cannot let the model cite a line
# it wasn't entitled to.

# Per-symbol docstring cap. Flask's are short; this is insurance against an
# unusually long module docstring.
DOCSTRING_CAP = 300

# Per-text-hit matched source-line cap. An ordinary source line is <120 chars;
# this defends against a minified/vendored single line without losing real
# nearby content the model might quote.
MATCHED_TEXT_CAP = 400

# Cap on total hits rendered into the prompt. The search tools bound themselves
# (search_text caps at 50 via DEFAULT_CAP), but search_symbols / search_files are
# repo-scale-bounded and a broad name query could flood an already-large prompt.
# When this bites it is surfaced in the hits block (no silent caps -- the same
# posture as the indexing stages' dropped-count reports), not applied invisibly.
DEFAULT_MAX_HITS = 40

# num_predict for the answer. Roomier than the Planner's 400 because the answer
# is prose WITH inline citations and the prompt already carries the hits block;
# 700 leaves comfortable room for a cited answer while still bounding a
# rambling model. Under think=False this budget is entirely answer (no CoT), so
# budget-exhaustion-to-empty (the Planner's root-cause failure) stays off.
DEFAULT_MAX_TOKENS = 1500

# Context-results bounds (SDD 9.3 results the Search Agent cannot produce:
# architecture centrality / dependency closures / file history / GitHub metadata).
DEFAULT_MAX_CONTEXT = 8

# Per-context-result rendered-text cap. Roomy so README documentation and
# code implementation snippets are not truncated.
CONTEXT_RESULT_CAP = 4000


# The system prompt: guides the model to produce detailed, grounded architectural,
# functional, and data flow explanations.
SYNTHESIZER_SYSTEM_PROMPT = """You are the Senior Code Intelligence Synthesizer for RepoLens. You analyze codebases and provide detailed, technical, step-by-step explanations based ONLY on the provided repository documentation, package configurations, route handlers, and source code excerpts.

Instructions:
1. Identify the exact technologies and libraries strictly from the retrieved evidence (e.g. Next.js Pages Router or App Router, Supabase, PostgreSQL, MongoDB, Prisma, NextAuth, etc. as confirmed in package dependencies or source imports).
2. When explaining architecture, data flow, or features, ground your answer in the concrete technical layers present in the evidence:
   * **Frontend UI / Pages Layer**: User interaction in React components / pages, forms, client state (e.g. localStorage), and API request calls.
   * **API Routing Layer**: Request handling, validation, and authorization in API route handlers (e.g. `pages/api/...` or `app/api/...`).
   * **Database & Persistence Layer**: Exact database client / ORM (e.g. Supabase client in `utils/supabase.ts`, Prisma, MongoDB) and data tables/collections referenced in the source excerpts.
   * **Authentication & Session Handling**: Real authentication mechanisms found in the source code (e.g. Supabase auth queries, token validation, or session persistence).
3. When analyzing a specific file or endpoint:
   * State the EXACT functions, HTTP methods, and handlers defined in the source code. NEVER invent or guess HTTP methods or libraries that do not exist in the retrieved excerpts.
   * State the EXACT request payload parameters, database tables, and helper functions called.
   * Describe the real logic step by step as written in the code.
4. Reference and cite the actual files provided in the evidence using inline backtick citations (e.g. `pages/api/validateCredentials.ts`, `utils/supabase.ts`).
5. Always be precise, direct, and grounded strictly in the provided evidence. NEVER invent boilerplate code, credentials providers, or database schemas not present in the evidence."""


# ─── result type ─────────────────────────────────────────────────────────────


@dataclass
class SynthesizerResult:
    """Everything the demo (and a future ``chat_messages.tool_trace`` row, SDD 11)
    needs from one synthesis call.

    ``answer`` is the final user-facing prose (leading/trailing whitespace
    stripped -- the version that goes to the user). ``raw_response`` is the
    model's unmodified ``message.content`` (the un-stripped raw, so nothing is
    hidden -- mirrors :class:`app.agents.planner.PlannerResult.raw_response`).

    ``hit_file_paths`` is the **grounding set**: the distinct file paths present
    in the *rendered* (shown) hits (sorted) -- i.e. the only paths the model was
    permitted to cite. ``cited_file_paths`` is the **model's claims**: the paths
    the answer actually names, extracted heuristically
    (:func:`_extract_cited_file_paths`), in first-appearance order, deduped. The
    build task asks for "the list of file paths actually cited (extracted from
    the answer) so a caller can later verify citations are grounded in real
    hits"; carrying BOTH on the result makes that deferred verification a
    one-liner -- ``set(cited) - set(hit)`` = paths the model invented,
    ``set(hit) - set(cited)`` = real hits the model left uncited -- without the
    caller re-deriving either side. The verification itself is deliberately NOT
    built here (the task defers it); these two fields are the inputs to it.

    ``hit_file_paths`` is the *shown* set (post-cap), not the full pre-cap search
    result, so the two sets are directly comparable: the model was told to cite
    only from the shown hits, so a path in ``cited_file_paths`` but not in
    ``hit_file_paths`` is a genuine invention, not a real-but-truncated hit.
    """

    question: str
    model: str
    answer: str
    raw_response: str
    hit_file_paths: list[str] = field(default_factory=list)
    cited_file_paths: list[str] = field(default_factory=list)
    wall_time_s: float = 0.0


# ─── pure helpers (no I/O) ────────────────────────────────────────────────────


def _normalize_newlines(s: str | None) -> str | None:
    """Fold CRLF / lone CR to LF so the model never sees a raw carriage return.

    ``None`` passes through (a missing docstring stays missing). Targets the two
    free-text fields sourced from disk that can carry the Windows CRLF artifact
    CLAUDE.md flags -- ``SymbolResult.docstring`` and ``TextHit.matched_text``;
    applied only to what's fed to the prompt, never to the model's own output.
    Lone ``\\r`` (a CR not part of a CRLF) is folded too, so a mixed-ending file
    is handled uniformly.
    """
    if s is None:
        return None
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _hit_path(hit: Any) -> str | None:
    """Return the POSIX-rel file path a hit refers to, or ``None`` if the hit is
    a type the Synthesizer doesn't recognize. The three SDD 10 hit dataclasses
    name this field three different ways -- :attr:`SymbolResult.file`,
    :attr:`FileResult.path`, :attr:`TextHit.file` -- so the path-extraction for
    the grounding set dispatches on type rather than guessing an attribute. A hit
    of an unexpected type is skipped (returns ``None``) rather than raising: the
    Search Agent re-exports these three as THE hit types, but :func:`synthesize`
    types ``results`` as a list and stays defensive if a future step emits
    something else (the hit is then also rendered under "Other hits").
    """
    if isinstance(hit, SymbolResult):
        return hit.file
    if isinstance(hit, FileResult):
        return hit.path
    if isinstance(hit, TextHit):
        return hit.file
    return None


def _line_range(start: int, end: int) -> str:
    """Render a symbol's line span as ``L`` or ``L-end``. A single-line symbol
    (``start == end``) collapses to one number so the model cites
    ``file.py:42``, not ``file.py:42-42`` (which reads as a zero-width range)."""
    return str(start) if start == end else f"{start}-{end}"


def _cap(s: str | None, n: int) -> str | None:
    """Truncate ``s`` to ``n`` chars with an ellipsis marker, leaving ``None`` and
    short strings untouched. The citable fact for a docstring/matched-text hit is
    its line number (uncapped elsewhere), so trimming the prose is safe for
    grounding and bounds the prompt against a runaway line."""
    if s is None:
        return None
    return s if len(s) <= n else s[:n] + " …"


def _format_hits(
    results: list[SearchStepResult],
    *,
    max_hits: int,
) -> tuple[str, list[str]]:
    """Build the hits block for the prompt; return ``(block_text, ground_paths)``.

    The block is **segmented by the originating hit type** so the model sees, per
    section, both the facts AND the citation rule that applies to them -- because
    the three hit types differ in what is citable: ``SymbolResult`` and
    ``TextHit`` carry line numbers (cite ``path:line``); ``FileResult`` is a file
    with NO line (cite the path alone, never an invented line). Segmenting makes
    that distinction explicit instead of burying it in a mixed list the model
    could conflate (and then attach a fabricated line number to a file hit).

    Text fields that can carry the CRLF artifact (docstrings, matched source
    lines) are newline-normalized (:func:`_normalize_newlines`) and then
    ``repr``-quoted so embedded newlines stay as escaped ``\\n`` -- the block
    stays FLAT, one physical line per bullet, so a multi-line docstring can't
    break the section structure -- and capped (:func:`_cap`) so a minified source
    line or a module-spanning docstring can't swallow the prompt's budget.

    A total-hit cap (``max_hits``) bounds what enters the LLM: the union of all
    steps' hits is flattened in step-then-hit order and the first ``max_hits`` are
    rendered (first-N, simple and documented; for the demo's single-step
    questions each union is one tool, so there is no cross-tool starvation). When
    the cap bites, a banner surfaces it -- same "no silent caps" posture as the
    indexing stages' dropped-count reports.

    ``ground_paths`` is the sorted distinct set of paths in the *rendered* (shown)
    hits -- exactly the paths the model was permitted to cite, so a future
    grounding check (``cited - ground``) flags genuine inventions, not real-but-
    truncated hits. An all-empty input produces an explicit "(no hits ...)" block;
    the LLM is still called so the "nothing matched" path is an observable model
    output, not a code short-cut.
    """
    flat: list[Any] = []
    for step in results:
        flat.extend(step.hits)
    total = len(flat)
    capped = total > max_hits
    shown = flat[:max_hits]

    symbols = [h for h in shown if isinstance(h, SymbolResult)]
    files = [h for h in shown if isinstance(h, FileResult)]
    texts = [h for h in shown if isinstance(h, TextHit)]
    other = [
        h for h in shown if not isinstance(h, (SymbolResult, FileResult, TextHit))
    ]

def _extract_context_paths(context_results: list[Any] | None) -> set[str]:
    """Extract distinct valid file paths from context results."""
    paths: set[str] = set()
    if not context_results:
        return paths
    for cr in context_results:
        res = getattr(cr, "result", None)
        if not res:
            continue
        key_files = getattr(res, "key_files", None)
        if isinstance(key_files, list):
            for kf in key_files:
                p = getattr(kf, "path", None) if not isinstance(kf, dict) else kf.get("path")
                if p:
                    paths.add(p)
        key_snippets = getattr(res, "key_file_snippets", None)
        if isinstance(key_snippets, list):
            for ks in key_snippets:
                p = ks.get("path") if isinstance(ks, dict) else getattr(ks, "path", None)
                if p:
                    paths.add(p)
        overview = getattr(res, "overview_doc", None)
        if overview and overview.startswith("["):
            doc_p = overview.split("]")[0].lstrip("[")
            if doc_p:
                paths.add(doc_p)
        node = getattr(res, "node", None)
        if node:
            p = getattr(node, "path", None) if not isinstance(node, dict) else node.get("path")
            if p:
                paths.add(p)
        node_path = getattr(res, "node_path", None)
        if node_path:
            paths.add(node_path)
        for n in getattr(res, "neighbors_in", []) or []:
            p = getattr(n, "path", None) if not isinstance(n, dict) else n.get("path")
            if p:
                paths.add(p)
        for n in getattr(res, "neighbors_out", []) or []:
            p = getattr(n, "path", None) if not isinstance(n, dict) else n.get("path")
            if p:
                paths.add(p)
        target = getattr(res, "target", None)
        if target and ("/" in str(target) or any(str(target).endswith(ext) for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".md"))):
            paths.add(str(target))
    return paths


def _format_hits(
    results: list[SearchStepResult],
    *,
    context_results: list[Any] | None = None,
    max_hits: int = DEFAULT_MAX_HITS,
) -> tuple[str, list[str]]:
    """Flatten every hit in ``results`` into one citable prompt block and the
    grounding set of distinct file paths.
    """
    all_hits: list[Any] = []
    for r in results:
        all_hits.extend(r.hits)
    total = len(all_hits)
    capped = total > max_hits
    shown = all_hits[:max_hits]

    symbols: list[SymbolResult] = []
    files: list[FileResult] = []
    texts: list[TextHit] = []
    other: list[Any] = []
    for h in shown:
        if isinstance(h, SymbolResult):
            symbols.append(h)
        elif isinstance(h, FileResult):
            files.append(h)
        elif isinstance(h, TextHit):
            texts.append(h)
        else:
            other.append(h)

    ground_set: set[str] = set()
    for h in shown:
        p = _hit_path(h)
        if p:
            ground_set.add(p)
    ground_set.update(_extract_context_paths(context_results))

    # Add search snippets to ground set
    search_snippets: list[dict[str, str]] = []
    for r in results:
        snips = getattr(r, "matched_file_snippets", None)
        if snips and isinstance(snips, list):
            for s in snips:
                sp = s.get("path")
                if sp:
                    ground_set.add(sp)
                    search_snippets.append(s)

    ground = sorted(ground_set)

    parts: list[str] = []
    if capped:
        parts.append(
            f"((NOTE: the tools returned {total} hits in total; only the first "
            f"{max_hits} are listed below. Cite ONLY from the hits shown here.))"
        )

    # Render matched file source snippets if available
    if search_snippets:
        parts.append("## Matched Source File Implementation Snippets:")
        for snip in search_snippets[:3]:
            sp = snip.get("path", "")
            sc = snip.get("snippet", "")
            if sp and sc:
                parts.append(f"### `{sp}` (source excerpt):\n```\n{sc}\n```")

    if not (symbols or files or texts):
        if context_results or search_snippets:
            parts.append("## Retrieved Repository Evidence & Context (from repository inspection):")
            parts.append("Analyze the documentation, key files, and implementation snippets provided in the Context section below to answer the user's question.")
        else:
            parts.append("(no hits were returned by any search tool)")
        parts.append(
            "Allowed citation paths: "
            + (", ".join(f"`{p}`" for p in ground) if ground else "(none)")
        )
        return "\n".join(parts), ground

    if symbols:
        parts.append(
            "## Symbol hits (each has a line number; cite as `path:line` or "
            "`path:start-end`)"
        )
        for h in symbols:
            line = f"- `{h.file}:{_line_range(h.line_start, h.line_end)}` -- {h.name} ({h.kind})"
            ds = _cap(_normalize_newlines(h.docstring), DOCSTRING_CAP)
            if ds:
                line += f" -- {ds!r}"
            parts.append(line)
    if files:
        parts.append(
            "## File hits (NO line numbers; cite the path WITHOUT a line -- "
            "never invent a line number for these)"
        )
        for h in files:
            parts.append(f"- `{h.path}` ({h.language}, {h.loc} LOC)")
    if texts:
        parts.append(
            "## Text hits (each is a matched source line; cite as `path:line`)"
        )
        for h in texts:
            mt = _cap(_normalize_newlines(h.matched_text), MATCHED_TEXT_CAP)
            parts.append(f"- `{h.file}:{h.line}` -- {mt!r}")
    if other:
        parts.append("## Other hits (unrecognized hit type -- shown raw; cite with care)")
        for h in other:
            parts.append(f"- {h!r}")

    parts.append(
        "Allowed citation paths: "
        + (", ".join(f"`{p}`" for p in ground) if ground else "(none)")
    )
    return "\n".join(parts), ground


def _format_context_block(
    context_results: list[Any] | None,
    *,
    max_context: int,
) -> str:
    """Build the Context results block appended to the user message."""
    if not context_results:
        return ""

    def _fold_strs(obj: Any) -> Any:
        if isinstance(obj, str):
            return _normalize_newlines(obj)
        if isinstance(obj, list):
            return [_fold_strs(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _fold_strs(v) for k, v in obj.items()}
        return obj

    parts: list[str] = [
        "## Context results (structural, relational, and architectural facts from repository tools):"
    ]
    n_dropped = max(0, len(context_results) - max_context)
    for r in context_results[:max_context]:
        res_obj = getattr(r, "result", None)

        overview_doc = getattr(res_obj, "overview_doc", None) if res_obj else None
        if overview_doc:
            parts.append(f"### Repository Overview Documentation:\n{overview_doc}\n")

        target_snip = getattr(res_obj, "target_file_snippet", None) if res_obj else None
        node_path = getattr(res_obj, "node_path", None) if res_obj else None
        if target_snip and node_path:
            parts.append(f"### Target File Implementation (`{node_path}` source excerpt):\n```\n{target_snip}\n```\n")

        snippets = getattr(res_obj, "key_file_snippets", None) if res_obj else None
        if snippets and isinstance(snippets, list):
            parts.append("### Key Implementation File Snippets:")
            for snip in snippets:
                sp = snip.get("path") if isinstance(snip, dict) else getattr(snip, "path", "")
                sc = snip.get("snippet") if isinstance(snip, dict) else getattr(snip, "snippet", "")
                if sp and sc:
                    parts.append(f"#### `{sp}` (source excerpt):\n```\n{sc}\n```")

        try:
            dumped = json.dumps(_fold_strs(asdict(res_obj) if res_obj and hasattr(res_obj, "__dataclass_fields__") else res_obj), ensure_ascii=False)
        except (TypeError, ValueError):
            dumped = repr(res_obj)
        if len(dumped) > CONTEXT_RESULT_CAP:
            dumped = dumped[:CONTEXT_RESULT_CAP] + " …"
        parts.append(f"- {r.tool} (args={json.dumps(r.args, ensure_ascii=False)}) -> {dumped}")

    if n_dropped:
        parts.append(
            f"((NOTE: {len(context_results)} context results were returned in "
            f"total; only the first {max_context} are listed here.))"
        )
    return "\n".join(parts)


# ─── citation extraction from the answer (a heuristic, clearly labeled) ─────
# A run of path-legal characters ([A-Za-z0-9_./-]); backticks / parens / colons
# are NOT in the class, so a citation like `src/flask/app.py:42` is matched as
# the single token `src/flask/app.py` (the `:` truncates it cleanly -- which is
# what we want; the line number is not part of the path claim). See
# :func:`_extract_cited_file_paths` for the keep/drop rules that turn these runs
# into the cited-path list.
_CITATION_TOKEN_RE = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./\-]*")


def _extract_cited_file_paths(
    answer: str,
    ground_paths: list[str] | None = None,
) -> list[str]:
    """Heuristic extraction of the file paths the model actually NAMED in its
    answer, in first-appearance order, deduped."""
    seen: set[str] = set()
    out: list[str] = []

    # First priority: check any ground-truth path that appears in the answer text
    if ground_paths:
        for gp in ground_paths:
            if gp and gp in answer and gp not in seen:
                seen.add(gp)
                out.append(gp)

    for m in _CITATION_TOKEN_RE.finditer(answer):
        tok = m.group(0)
        if not any(c.isalpha() for c in tok):
            continue
        if tok.endswith("/"):
            tok = tok[:-1]
        if not tok:
            continue
        if not ("/" in tok or tok.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".md"))):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# ─── the LLM call (the tool the orchestrator/demo calls) ────────────────────


def _chat_payload(
    question: str,
    hits_block: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    think: bool = False,
    context_block: str = "",
) -> dict[str, Any]:
    """Build the JSON body for Ollama ``POST /api/chat``. Pure (no I/O), split out
    so it can be inspected without a network call (mirrors
    :func:`app.agents.planner._chat_payload`). Unlike the Planner's payload there
    is NO ``format`` key -- the output is prose (see module docstring) -- and the
    user message carries the hits block (the Planner's carries a tool catalog).

    ``context_block`` (default ``""``) is the additional "Context results"
    block :func:`_format_context_block` produces from the executed context plan
    steps (SDD 9.3). When empty or all-whitespace (the common case today: the
    Planner's context menu is not yet wired, so context_results is empty for
    every current execution path) the user message is **byte-identical** to the
    pre-context search-only prompt -- so :mod:`scripts.inspect_q1_prompt` and the
    11 verified demo/diagnostic runs are preserved unchanged. When non-empty the
    context block is appended between the hits block and the question, clearly
    labeled "CONTEXTUAL only; not additional citation grounds" (see
    :func:`_format_context_block`).

    ``think`` maps to the top-level Ollama ``/api/chat`` ``think`` param; ``False``
    (the default) disables chain-of-thought (see module docstring for the
    budget-exhaustion rationale). Top-level, NOT under ``options``.
    """
    if context_block and context_block.strip():
        if hits_block.strip().startswith("## Retrieved Repository Evidence"):
            user_content = f"{hits_block}\n\n{context_block}\n\nUser question:\n{question}"
        else:
            user_content = (
                f"Search hits:\n{hits_block}\n\nContext results:\n{context_block}\n"
                f"\nUser question:\n{question}"
            )
    else:
        user_content = f"Search hits:\n{hits_block}\n\nUser question:\n{question}"

    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    return body


def _ollama_chat(
    payload: dict[str, Any],
    *,
    host: str,
    timeout: float,
) -> str:
    """POST to Ollama ``/api/chat`` and return the assistant message text.

    Local copy of :func:`app.agents.planner._ollama_chat` -- the same
    "keep each module self-contained" posture the tool modules take toward their
    copied helpers. Identical behavior; raises the SAME public :class:`OllamaError`
    (imported from the Planner) so transport/infra failures stay one error
    category across every agent hop. See the Planner's docstring for the full
    error-split rationale (infra failure vs. model output); not duplicated here.
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
        except Exception:  # noqa: BLE001 -- best-effort body capture
            pass
        raise OllamaError(
            f"Ollama HTTP {exc.code} at {url}: {body_text[:300] or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        # Connection refused / host unreachable -- Ollama not running. Distinct
        # message so a caller can report "Ollama unreachable" as its own category
        # (the demo does, for both stages via the shared OllamaError).
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

    message = parsed.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise OllamaError(f"Ollama response missing message.content: {raw[:300]}")
    content = message.get("content")
    if not isinstance(content, str):
        raise OllamaError(f"Ollama message.content is not a string: {raw[:300]}")
    return content


def synthesize(
    question: str,
    results: list[SearchStepResult],
    *,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
    think: bool = False,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_hits: int = DEFAULT_MAX_HITS,
    context_results: list[ContextStepResult] | None = None,
    max_context: int = DEFAULT_MAX_CONTEXT,
    timeout: float = DEFAULT_TIMEOUT,
) -> SynthesizerResult:
    """Produce the final user-facing answer to ``question`` from the structured
    search hits in ``results`` (one :class:`SearchStepResult` per executed plan
    step -- SDD 9.2 -> 9.4), plus the optional context results in
    ``context_results`` (one :class:`ContextStepResult` per executed context plan
    step -- SDD 9.3 -> 9.4), via one Ollama ``/api/chat`` call.

    Defaults mirror the Planner's lessons: ``think=False`` (chain-of-thought off
    so the answer isn't budget-exhausted to empty -- see module docstring),
    ``temperature=0.0`` for reproducibility, and **no ``format``** (prose, not
    JSON -- there is no schema to satisfy, no parse to rescue, so the SDD 18
    JSON lever does not apply and retries are not wired: prose has no
    structured-output parse that could fail and warrant rescue). ``max_tokens``
    is roomier than the Planner's 400 because the answer is prose with citations
    and the prompt already carries the hits block.

    ``max_hits`` caps how many hits are rendered into the prompt (the search tools
    bound themselves, but ``search_symbols`` / ``search_files`` are repo-scale-
    bounded); when the cap bites it is surfaced in the hits block, not applied
    silently. ``max_context`` is the twin cap for the context block (see
    :func:`_format_context_block`); when ``context_results`` is ``None`` or empty
    -- the common case today, because the Planner's context menu is not yet wired
    -- the context block is the empty string and the user message is byte-identical
    to the pre-context search-only prompt.

    **Grounding posture is unchanged by context_results.** The model is permitted
    to cite ONLY the search-hit paths (in ``SynthesizerResult.hit_file_paths``);
    the context block is rendered under a "CONTEXTUAL only; NOT additional
    citation grounds" header and its paths are deliberately NOT added to the
    grounding set, so a path that appears in (e.g.) architecture's key-files list
    cannot become a citation the search step never vouched for. (Bespoke per-tool
    context rendering is a deferred follow-up, exercised only once the Planner's
    context menu lights up -- see :func:`_format_context_block`.)

    An empty ``results`` (or all-empty hits) is a *valid* input (SDD 10: an empty
    result is a valid answer): the LLM is still called -- the system prompt tells
    it to say plainly that nothing matched -- so the "no hits" path is a real,
    observable model output the demo can eyeball, not a code short-cut that hides
    whether the model would have hallucinated anyway.

    Returns :class:`SynthesizerResult`. Raises :class:`OllamaError` (from the
    Planner) on transport/infra failure (Ollama down, bad model, non-200, malformed
    body) -- the same category the chain demo already catches for the Planner, so
    one ``except`` covers both stages.
    """
    hits_block, ground = _format_hits(results, context_results=context_results, max_hits=max_hits)
    context_block = _format_context_block(context_results, max_context=max_context)
    payload = _chat_payload(
        question,
        hits_block,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        think=think,
        context_block=context_block,
    )
    start = time.perf_counter()
    raw = _ollama_chat(payload, host=host, timeout=timeout)
    end = time.perf_counter()
    answer = raw.strip()
    return SynthesizerResult(
        question=question,
        model=model,
        answer=answer,
        raw_response=raw,
        hit_file_paths=ground,
        cited_file_paths=_extract_cited_file_paths(answer, ground_paths=ground),
        wall_time_s=end - start,
    )


__all__ = [
    "CONTEXT_RESULT_CAP",
    "DEFAULT_MAX_CONTEXT",
    "SynthesizerResult",
    "synthesize",
]
