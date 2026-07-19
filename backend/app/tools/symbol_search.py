"""Symbol-search tool for the RepoLens agent layer (SDD §10).

The first of the thin Postgres-backed tools the agent orchestrator (SDD §9) will
call: given a repository and a name (or partial name), return the repo's symbols
whose ``name`` matches as structured data — never prose, never free text — in the
SDD §10 Symbol Search output shape ``[{name, kind, file, line_start, line_end,
docstring}]``. ``file`` is the symbol's POSIX-rel path (``files.path`` joined
through the ``symbols.file_id`` FK), because agents cite file paths, not surrogate
``file_id`` ints — the SDD §10 shape names this field ``file`` deliberately.

Pure read, no side effects, injected session — same posture as the indexing
``index_*`` stages: :func:`search_symbols` takes the ``Session`` it reads through
and owns no transaction (no open, no commit, no close). The future agent layer
opens one session per query and passes it down, exactly as the throwaway driver
scripts already drive the indexer; tests pass their own. "Pure query function" is
this stage's invariant, not a claim of no I/O — it touches Postgres, but writes
nothing and changes no state.

Matching:

* **Case-insensitive + partial.** ``symbols.name ILIKE '%query%'`` — a substring
  match, so ``"widget"`` finds ``"Widget"`` (case) and ``"lueprint"`` finds
  ``"Blueprint"`` (mid-name substring). This is name-textual only: it is **not**
  a scope/owner-resolved search. Two symbols sharing a name in different files, or
  a method and a same-named top-level function, both surface — resolution beyond
  the name is a later concern, the same "honestly weaker, clearly labeled" posture
  as the reference-index stage's reference-graph-not-call-graph note (SDD §8/§13).
* **LIKE metacharacters are escaped** (``%``, ``_``, ``\\``) via
  :func:`_ilike_contains` + the ``ESCAPE '\\'`` clause, so a literal ``__init__``
  matches its underscores, not "any-char init any-char" — the SQL LIKE analog of
  the reference-index stage's ripgrep ``--fixed-strings`` belt-and-suspenders
  against accidental wildcard interpretation.
* **Optional ``kind`` filter** (SDD §10: "optional kind filter: class/function/
  method") restricts to one of the parser's stored lowercase kinds (``class`` /
  ``function`` / ``method`` / ``variable``); ``None`` returns every kind. An
  unknown ``kind`` simply matches nothing (empty result, not an error), consistent
  with the search contract that an empty result is a valid answer.

No result cap: the SDD §10 Symbol Search row lists none (unlike Text Search, which
caps), and a single repo's symbol set is bounded by indexing scale (hundreds–
thousands at MVP). The ``symbols.name`` B-tree cannot serve a leading-``%`` LIKE
prefix, so the match is a scan at MVP scale — acceptable for the target repo sizes
(SDD §7); a trigram index / cap are natural later refinements if large repos
arrive.

Results are ordered by ``(name, file, line_start, id)`` for deterministic output
(stable tests, stable agent display).
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import File, Symbol

# Escape char for the ILIKE patterns (emitted into the SQL as `ESCAPE '\'`).
# Escaping LIKE metacharacters in the query keeps a literal `__init__` matching
# its underscores rather than the single-char wildcard `_` — see
# :func:`_ilike_contains`.
LIKE_ESCAPE: str = "\\"

# The symbol `kind` values the MVP parser emits (SDD §11): class | function |
# method | variable. Surfaced for callers/tests; the parser writes these lowercased.
SYMBOL_KINDS: frozenset[str] = frozenset({"class", "function", "method", "variable"})


@dataclass
class SymbolResult:
    """One matched symbol, in the SDD §10 Symbol Search output shape.

    Every field is a plain JSON type (str / int / None), so ``dataclasses.asdict``
    is directly ``json.dumps``-able — the whole point of SDD §10's "structured
    JSON, never free text" contract: agents and the Synthesizer work from these
    facts, not paraphrased summaries.

    ``file`` is the POSIX-rel path (``files.path``), not ``file_id``; ``docstring``
    is the parser-captured first-statement docstring (nullable).
    """

    name: str
    kind: str
    file: str
    line_start: int
    line_end: int
    docstring: str | None


def _ilike_contains(query: str) -> str:
    """Build a case-insensitive *substring* ILIKE pattern for ``query``.

    Wraps ``query`` in ``%…%`` so it matches anywhere in the column (partial /
    substring search — SDD §10 "name/partial name") and escapes LIKE
    metacharacters (``%``, ``_``, ``\\``) with :data:`LIKE_ESCAPE` so they match
    literally. Pair the result with ``Column.ilike(pattern, escape=LIKE_ESCAPE)``.

    The pattern is case-*preserving*: case-insensitivity comes from ``ILIKE`` at
    query time, not from mutating the query here (leaving the name's own case
    untouched keeps the match faithful and the helper a pure string transform).

    Examples::

        _ilike_contains("Widget")    -> "%Widget%"
        _ilike_contains("__init__")   -> r"%\_\_init\_\_%"   # _ escaped
        _ilike_contains("a%b")        -> r"%a\%b%"            # % escaped
        _ilike_contains("a\\b")       -> r"%a\\b%"            # backslash doubled
    """
    e = LIKE_ESCAPE
    escaped = query.replace(e, e + e).replace("%", e + "%").replace("_", e + "_")
    return f"%{escaped}%"


def search_symbols(
    repository_id: int,
    query: str,
    session: Session,
    *,
    kind: str | None = None,
) -> list[SymbolResult]:
    """Return the ``symbols`` of repository ``repository_id`` whose ``name``
    case-insensitively contains ``query``, as :class:`SymbolResult` (SDD §10 shape).

    *Pure read*: one ``SELECT`` over ``symbols`` joined to ``files`` (the join
    carries the ``repository_id`` filter and supplies ``file`` = ``files.path``),
    built into dataclasses, returned — no commit, no session ownership. See the
    module docstring for the matching, ``kind``-filter, no-cap, and ordering
    rationale.

    ``query`` is stripped first; an empty query returns ``[]`` (an empty search
    term is not a search, and avoids dumping the whole symbol table into agent
    context). ``kind`` is stripped + lowercased and, if empty afterwards, treated
    as ``None`` (no kind filter) — so a stray ``"  "`` kind does not silently zero
    out an otherwise-valid search.
    """
    query = query.strip()
    if not query:
        return []

    pattern = _ilike_contains(query)
    clauses = [
        File.repository_id == repository_id,
        Symbol.name.ilike(pattern, escape=LIKE_ESCAPE),
    ]
    if kind is not None:
        kind = kind.strip().lower()
        if kind:
            clauses.append(Symbol.kind == kind)

    stmt = (
        sa.select(
            Symbol.name,
            Symbol.kind,
            File.path,
            Symbol.line_start,
            Symbol.line_end,
            Symbol.docstring,
        )
        .join(File, Symbol.file_id == File.id)
        .where(*clauses)
        .order_by(Symbol.name, File.path, Symbol.line_start, Symbol.id)
    )
    return [
        SymbolResult(
            name=name,
            kind=kind_,
            file=path,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
        )
        for name, kind_, path, line_start, line_end, docstring in session.execute(stmt).all()
    ]
