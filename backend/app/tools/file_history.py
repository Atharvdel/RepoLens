"""History Search tool for the RepoLens agent layer (SDD §10).

The third context tool the §9.3 Context Agent dispatches. SDD §10 "History
Search" — "GitPython log parsing (cached at index time), file path, output
``{last_modified, top_contributors: [{author, commits}], recent_commits:
[{hash, message, date, author}]}``". Per SDD §11 it reads *cached* history —
the git log is walked once at index time (SDD §7 step 7) and the commits are
materialized as ``commits`` + ``file_commits`` rows; this tool is a pure read
that aggregates them again per query, never re-walking git log (the §11
"caching" rationale: avoid re-walking ``git log`` on every question).

**Backing status (when built): pending.** The ``commits`` / ``file_commits``
tables exist in the committed initial migration
(:mod:`app.alembic.versions.0001_initial_schema`) but are **not populated** —
the §7 step 7 git-history indexing stage is not yet built. So against the
indexed flask repo this tool returns ``last_modified`` (the walker wrote that in
§7 step 1) with empty contributor / commit lists: a clear, honest "no history
indexed yet" rather than a crash. The tool is *complete and correct* (proven by
synthetic-DB tests that insert commit rows and assert the aggregation); it lights
up untouched the moment §7 step 7 lands. This is the same "build the tool, flag
the pending backing, verify against synthetic rows" posture the architecture /
dependency-graph tools take toward the import graph — except here the *entire*
backing table is empty, not just a slice.

Design split (the project's pure-core / live-wrapper convention, mirrored from
:mod:`app.tools.dependency_graph` / the indexing stages):

* :func:`build_file_history` — **pure**: takes the file's path + ``last_modified``
  + a list of commit tuples ``(hash, message, date, author)`` (date a
  :class:`~datetime.datetime`) and returns a :class:`FileHistoryResult` with
  contributors aggregated (committed-count per author, ranked desc) + recent
  commits (date desc, capped). No DB, no session. Unit-testable against a
  synthetic commit list.
* :func:`query_file_history` — **live**: resolves ``target`` to a ``file_id``
  via :func:`app.tools._path_resolve.resolve_file_id`, reads ``files.last_modified``
  + the commits touching the file (``commits`` joined through ``file_commits``,
  SDD §11 join table), and delegates aggregation to the pure core. Injected
  ``Session``, owns no transaction.

Contributor aggregation: a commit's *author* is the "who" SDD §10 asks for
(``top_contributors: [{author, commits}]``). Git separates author from
committer; the §7 step 7 indexer is the one that picks (the committed
:meth:`app.models.commit.Commit` model stores only ``author``, the §11 spec
names only ``author``), so this tool reports by stored author. Ties in commit
count are broken by author name (stable, deterministic — same posture as the
search tools' ``order_by``).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Commit, File, file_commits
from app.tools._path_resolve import resolve_file_id

# Cap on "recent_commits" reported. SDD §10 names the shape but no count; a
# short recent-commits tail keeps the agent-readable history compact (the
# full log belongs in the file_commits table, not the agent context). The
# top_contributors list is unbounded below by a per-author cap — a handful of
# authors is typical for a student/OSS repo (SDD §7 target scale), so it stays
# short in practice. Both are the caller's to override once the Planner knows.
DEFAULT_RECENT_CAP: int = 10


# ─── result types ────────────────────────────────────────────────────────────


@dataclass
class ContributorView:
    """One contributor in the file's ``top_contributors`` list (SDD §10 shape).

    ``author`` is the stored commit author string; ``commits`` is the count of
    that file's commits attributed to them. JSON-serializable (plain types)."""

    author: str
    commits: int


@dataclass
class CommitView:
    """One commit in the file's ``recent_commits`` list (SDD §10 shape).

    Every field is a plain JSON type: ``date`` is the ISO-8601 string of the
    commit's ``commits.date`` (``datetime`` is not ``json.dumps``-able by
    default, and this tool is the serialization boundary — the same posture as
    :class:`app.tools.file_search.FileResult`'s ``last_modified``). ``hash`` is
    the full git SHA the indexer stored (the Commit model stores the full hash;
    a caller may shorten for display)."""

    hash: str
    message: str
    date: str
    author: str


@dataclass
class FileHistoryResult:
    """The SDD §10 History Search output shape, with an added ``file_path``.

    ``file_path`` is the POSIX-rel path resolved from ``target`` (or ``None``
    if unresolvable), added so the Synthesizer can cite "history of
    ``<path>``" without re-resolving. ``last_modified`` is the ISO-8601 string
    of ``files.last_modified`` (set by the walker in §7 step 1) — populated
    even before §7 step 7 lands, which is why a history query on flask today
    returns a populated ``last_modified`` with empty contributor/commit lists.
    An unresolvable ``target`` (the resolver returns ``None``) yields a clear
    empty result: ``file_path=None`` everywhere, empty lists, ``last_modified=None``
    — the same "don't guess, return clear empty" contract the graph tools hold."""

    file_path: str | None
    last_modified: str | None
    top_contributors: list[ContributorView] = field(default_factory=list)
    recent_commits: list[CommitView] = field(default_factory=list)


# ─── pure core (no DB) ───────────────────────────────────────────────────────


def build_file_history(
    file_path: str | None,
    last_modified: datetime | None,
    commits: Iterable[tuple[str, str, datetime, str]],
    *,
    recent_cap: int = DEFAULT_RECENT_CAP,
) -> FileHistoryResult:
    """Aggregate ``commits`` (``(hash, message, date, author)`` tuples) touching
    a file into a :class:`FileHistoryResult` (SDD §10 History Search shape).

    *Pure*: no DB, no session. ``last_modified`` (a ``datetime`` from
    ``files.last_modified``) is ISO-encoded at this boundary; ``commits`` are
    sorted newest-first and capped at ``recent_cap``; contributors are ranked
    by commit count desc (ties broken by author name asc). The whole result is
    ``json.dumps(dataclasses.asdict(result))``-able.

    ``file_path`` is passed straight through to the result (the live wrapper
    already resolved it); a ``None`` file_path with an empty ``commits``
    iterable is the unresolvable-target case the live wrapper hands us.
    """
    commit_list = list(commits)
    # Recent commits: newest first, capped. Stable sort by (date desc) then
    # (hash asc) so same-second commits are deterministic.
    recent = sorted(
        commit_list, key=lambda c: (c[2], c[0]), reverse=True
    )[: max(0, recent_cap)]

    # Contributors: count per author, rank by count desc then author asc.
    counts = Counter(c[3] for c in commit_list)
    contributors = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    return FileHistoryResult(
        file_path=file_path,
        last_modified=last_modified.isoformat() if last_modified is not None else None,
        top_contributors=[
            ContributorView(author=author, commits=count)
            for author, count in contributors
        ],
        recent_commits=[
            CommitView(
                hash=h,
                message=msg,
                date=d.isoformat(),
                author=a,
            )
            for (h, msg, d, a) in recent
        ],
    )


# ─── live wrapper (injected session) ─────────────────────────────────────────


def query_file_history(
    repository_id: int,
    target: str | None = None,
    session: Session | None = None,
    *,
    recent_cap: int = DEFAULT_RECENT_CAP,
) -> FileHistoryResult:
    """Return the cached git history for the file ``target`` resolves to, or for
    the entire repository if ``target`` is omitted / 'repository_root' / 'repo' / '.'
    (SDD §10 History Search).

    *Live read*: resolves ``target`` to a ``file_id`` via
    :func:`app.tools._path_resolve.resolve_file_id`, reads ``files.last_modified``
    plus the commits touching the file (``commits`` joined through the
    ``file_commits`` join table, SDD §11), and hands them to
    :func:`build_file_history`. Injected ``Session``, owns no transaction.
    """
    try:
        cap = max(0, int(recent_cap))
    except (TypeError, ValueError):
        cap = DEFAULT_RECENT_CAP

    if session is None:
        return FileHistoryResult(file_path=None, last_modified=None)

    target_clean = str(target or "").strip().lower()
    is_repo_wide = not target or target_clean in ("", ".", "repository_root", "repo", "all", "root", "*", "whole_repo")

    if is_repo_wide:
        commit_rows = (
            session.execute(
                sa.select(Commit.hash, Commit.message, Commit.date, Commit.author)
                .where(Commit.repository_id == repository_id)
                .order_by(Commit.date.desc())
            )
            .all()
        )
        commits = [(h, msg, d, a) for (h, msg, d, a) in commit_rows]
        return build_file_history(
            file_path="repository_root",
            last_modified=commits[0][2] if commits else None,
            commits=commits,
            recent_cap=cap,
        )

    file_id = resolve_file_id(repository_id, target, session)
    if file_id is None:
        # If target has "repo" or "root" in its name or is unresolvable generic query, fallback to repo-wide
        if "repo" in target_clean or "root" in target_clean or target_clean in ("git", "project"):
            commit_rows = (
                session.execute(
                    sa.select(Commit.hash, Commit.message, Commit.date, Commit.author)
                    .where(Commit.repository_id == repository_id)
                    .order_by(Commit.date.desc())
                )
                .all()
            )
            commits = [(h, msg, d, a) for (h, msg, d, a) in commit_rows]
            return build_file_history(
                file_path="repository_root",
                last_modified=commits[0][2] if commits else None,
                commits=commits,
                recent_cap=cap,
            )
        return FileHistoryResult(file_path=None, last_modified=None)

    row = session.execute(
        sa.select(File.path, File.last_modified).where(File.id == file_id)
    ).first()
    if row is None:
        return FileHistoryResult(file_path=None, last_modified=None)
    file_path, last_modified = row

    commit_rows = (
        session.execute(
            sa.select(Commit.hash, Commit.message, Commit.date, Commit.author)
            .join(file_commits, file_commits.c.commit_id == Commit.id)
            .where(file_commits.c.file_id == file_id)
            .order_by(Commit.date.desc())
        )
        .all()
    )
    commits = [(h, msg, d, a) for (h, msg, d, a) in commit_rows]

    return build_file_history(
        file_path=file_path,
        last_modified=last_modified,
        commits=commits,
        recent_cap=cap,
    )


__all__ = [
    "CommitView",
    "ContributorView",
    "DEFAULT_RECENT_CAP",
    "FileHistoryResult",
    "build_file_history",
    "query_file_history",
]
