"""File-search tool for the RepoLens agent layer (SDD §10).

The second thin Postgres-backed tool: given a repository and a filename or path
fragment, return the repo's ``files`` rows whose ``path`` matches, as structured
data in the SDD §10 File Search output shape ``[{path, language, loc,
last_modified}]``. The companion to :mod:`app.tools.symbol_search`; the two share
matching semantics (escaped substring ILIKE) but this tool queries the ``files``
table directly — no join, since ``files`` itself carries ``repository_id``.

Pure read, no side effects, injected session — same posture as the indexing
``index_*`` stages and as :func:`app.tools.symbol_search.search_symbols`: the
``Session`` is injected and the function owns no transaction (no open, no commit,
no close). "Pure query function" means it writes nothing and changes no state; it
still reads Postgres.

Matching: ``files.path ILIKE '%query%'`` with LIKE metacharacters escaped (see
:func:`_ilike_contains`), so ``"app.py"`` finds ``src/flask/app.py`` (substring),
``"APP.PY"`` finds it too (case-insensitive), and a literal ``__init__`` query
matches its underscores rather than the single-char wildcard.

``last_modified`` is serialized to an ISO-8601 string (or ``None``) in
:class:`FileResult` so the whole result is ``json.dumps``-able — ``datetime`` is
not JSON-serializable by default, and this tool is the serialization boundary
(SDD §10 "structured JSON, never free text"). The matching helper
:func:`_ilike_contains` is kept as a copy of :mod:`app.tools.symbol_search`'s
rather than shared, so each tool module is self-contained (the lifting is small
and stable; mirror the one in symbol_search if it ever changes).

No result cap (SDD §10 File Search lists none); the ``files.path`` UNIQUE index
serves exact lookups but not a leading-``%`` LIKE, so this is a scan at MVP scale
— fine for target repo sizes (SDD §7). Results ordered by ``path`` for
deterministic output.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import File

# Escape char for the ILIKE patterns (emitted into the SQL as `ESCAPE '\'`).
# Matches :mod:`app.tools.symbol_search` — escaping LIKE metacharacters in the
# query keeps a literal `__init__` (or any `_`-laden path fragment) matching its
# underscores rather than the single-char wildcard `_`.
LIKE_ESCAPE: str = "\\"


@dataclass
class FileResult:
    """One matched file, in the SDD §10 File Search output shape.

    Every field is a plain JSON type: ``last_modified`` is the ISO-8601 string of
    ``files.last_modified`` (or ``None``) — ``datetime`` is not
    ``json.dumps``-able by default, and this tool is the JSON boundary, so the
    timestamp is serialized here. ``dataclasses.asdict(result)`` is directly
    JSON-serializable.
    """

    path: str
    language: str
    loc: int
    last_modified: str | None


def _ilike_contains(query: str) -> str:
    """Case-insensitive substring ILIKE pattern for ``query`` with LIKE
    metacharacters (``%``, ``_``, ``\\``) escaped. Pair with
    ``File.path.ilike(pattern, escape=LIKE_ESCAPE)``. See the twin in
    :mod:`app.tools.symbol_search` for the full rationale; it is identical here.
    """
    e = LIKE_ESCAPE
    escaped = query.replace(e, e + e).replace("%", e + "%").replace("_", e + "_")
    return f"%{escaped}%"


def search_files(
    repository_id: int,
    query: str,
    session: Session,
) -> list[FileResult]:
    """Return the ``files`` of repository ``repository_id`` whose ``path``
    case-insensitively contains ``query``, as :class:`FileResult` (SDD §10 shape).

    *Pure read*: one ``SELECT`` over ``files`` (no join — ``files`` owns
    ``repository_id``), built into dataclasses with ``last_modified`` ISO-encoded,
    returned — no commit, no session ownership.

    ``query`` is stripped first; an empty query returns ``[]`` (an empty search
    term is not a search, and avoids dumping the whole file table into agent
    context).
    """
    query = query.strip()
    if not query:
        return []

    stmt = (
        sa.select(File.path, File.language, File.loc, File.last_modified)
        .where(
            File.repository_id == repository_id,
            File.path.ilike(_ilike_contains(query), escape=LIKE_ESCAPE),
        )
        .order_by(File.path)
    )
    return [
        FileResult(
            path=path,
            language=language,
            loc=loc,
            last_modified=lm.isoformat() if lm is not None else None,
        )
        for path, language, loc, lm in session.execute(stmt).all()
    ]
