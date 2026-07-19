"""GitHub Metadata Loader tool for the RepoLens agent layer (SDD §10).

The fourth context tool the §9.3 Context Agent dispatches. SDD §10 "GitHub
Metadata Loader" — "PyGithub (cached at index time), repo, optional file path,
output ``{issues: [{number, title, labels, state, url}], prs: [...]}``".
Per SDD §11 it reads *cached* metadata: the GitHub issues/PRs are pulled once at
index time (SDD §7 step 8, PyGithub) and materialized as ``issues`` rows; this
tool is a pure read, never hitting the GitHub API per question (the §11
"caching" rationale: avoid rate-limiting the GitHub API).

**Backing status (when built): pending.** The ``issues`` table exists in the
committed initial migration but is **not populated** — the §7 step 8 PyGithub
indexing stage is not yet built. So against any indexed repo this tool returns
empty lists: a clear, honest "no GitHub metadata indexed yet". The tool is
*complete and correct* (proven by synthetic-DB tests that insert issue/PR rows
and assert the split + optional file-link filter); it lights up untouched the
moment §7 step 8 lands. Same "build the tool, flag the pending backing, verify
against synthetic rows" posture as :mod:`app.tools.file_history`.

**Issue vs PR split (a documented heuristic).** SDD §10's output shape separates
``issues`` and ``prs``, but the §11 ``issues`` table has **no type discriminator
column** — it stores rows GitHub calls both issues and PRs (the
:meth:`app.models.issue.Issue` model has no ``kind`` field; §0 explicitly keeps
PRs in scope but only as metadata, so the indexer pulls both into one table).
Rather than require a schema change to split, this tool partitions rows by the
**GitHub URL convention** the §7 step 8 indexer will store: a PR's ``url``
contains ``/pull/``, an issue's contains ``/issues/``. This is deterministic and
faithful to the §10 shape; a row whose URL matches neither is reported under
``issues`` (the honest default when the index-time URL is missing/ambiguous).
If a later schema revision adds a real discriminator, this heuristic becomes a
one-line fallback — documented, not load-bearing.

**Optional file-path filter.** SDD §10 says the loader takes "repo, optional
file path" — so a plan step may pass ``target`` to restrict to issues/PRs whose
cached ``linked_files`` array names that file (the "linked files via PyGithub"
note, §0). When ``target`` is empty/``None``, all the repo's cached rows are
returned. Filtering by ``linked_files`` membership (vs URL substring) is the
reliable path — PyGithub links are explicit, not text-guessing (same "don't
guess" posture as the import graph's PEP 420 note).

Design split (the project's pure-core / live-wrapper convention, mirrored from
:mod:`app.tools.file_history` / the indexing stages):

* :func:`build_github_metadata` — **pure**: takes issue rows
  ``(number, title, state, labels, url, linked_files)`` + an optional target
  path, partitions by URL into issues/PRs, filters by file-link, sorts, and
  returns a :class:`GitHubMetadataResult`. No DB, no session.
* :func:`query_github_metadata` — **live**: optionally resolves ``target`` to a
  file path via :func:`app.tools._path_resolve.resolve_file_id`, reads the
  repo's ``issues`` rows (filtered by ``linked_files`` when a target path is
  given), and delegates to the pure core. Injected ``Session``, owns no
  transaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import File, Issue
from app.tools._path_resolve import resolve_file_id


# ─── result types ────────────────────────────────────────────────────────────


@dataclass
class IssueView:
    """One cached GitHub issue or PR, in the SDD §10 GitHub Metadata output
    shape.

    ``labels`` is the ``issues.labels`` text array as a plain list (or ``[]``);
    every field is a plain JSON type so ``dataclasses.asdict`` is directly
    ``json.dumps``-able. The same shape serves both the ``issues`` and ``prs``
    lists per §10 — a PR is an issue-shaped row whose ``url`` names ``/pull/``."""

    number: int
    title: str
    state: str
    labels: list[str]
    url: str
    linked_files: list[str]


@dataclass
class GitHubMetadataResult:
    """The SDD §10 GitHub Metadata output shape, with an added ``file_path``.

    ``file_path`` is the POSIX-rel path ``target`` resolved to (or ``None`` when
    no ``target`` was given — the whole-repo query — or when the target was
    unresolvable). The Synthesizer cites "N issues linked to ``<path>``" using
    it without re-resolving. ``issues`` and ``prs`` are URLs the
    :func:`build_github_metadata` / ``_is_pr`` heuristic split. An unresolvable
    ``target`` (resolver returns ``None`` when one was requested) yields a
    clear-empty result, consistent with the other context tools' "don't guess"
    posture."""

    file_path: str | None
    issues: list[IssueView] = field(default_factory=list)
    prs: list[IssueView] = field(default_factory=list)


# ─── pure core (no DB) ───────────────────────────────────────────────────────


def _is_pr(url: str | None) -> bool:
    """Heuristic: does this cached GitHub URL name a PR (``/pull/``) rather than
    an issue (``/issues/``)? Matches the GitHub URL convention the §7 step 8
    indexer stores. A ``None``/ambiguous URL is treated as an issue (the honest
    default — PRs are a subset, and a missing URL is more likely an issue row
    than a PR). Documented in the module docstring."""

    if not url:
        return False
    return "/pull/" in url


def _make_view(row: tuple) -> IssueView:
    """Materialize one DB row ``(number, title, state, labels, url, linked_files)``
    into an :class:`IssueView`, coercing ``labels`` / ``linked_files`` from the
    nullable Postgres arrays to plain lists (``None`` -> ``[]``)."""
    number, title, state, labels, url, linked = row
    return IssueView(
        number=number,
        title=title,
        state=state,
        labels=list(labels) if labels is not None else [],
        url=url,
        linked_files=list(linked) if linked is not None else [],
    )


def build_github_metadata(
    rows: Iterable[tuple],
    *,
    target_path: str | None = None,
) -> GitHubMetadataResult:
    """Partition ``rows`` (each ``(number, title, state, labels, url,
    linked_files)``) into issues / PRs and optionally filter by ``target_path``,
    returning a :class:`GitHubMetadataResult` (SDD §10 GitHub Metadata shape).

    *Pure*: no DB, no session. Filtering by ``target_path`` keeps a row iff
    ``target_path`` is a member of the row's ``linked_files`` array (PyGithub's
    explicit linked-file metadata; empty/``None`` ``target_path`` keeps all —
    the whole-repo query). Issues and PRs are each sorted by ``number`` desc
    (newest-first, matching the file-history tool's recent-first posture).
    """
    target = (target_path or "").strip() or None

    issues: list[IssueView] = []
    prs: list[IssueView] = []
    for raw in rows:
        view = _make_view(raw)
        if target is not None and target not in view.linked_files:
            continue
        bucket = prs if _is_pr(view.url) else issues
        bucket.append(view)

    issues.sort(key=lambda v: v.number, reverse=True)
    prs.sort(key=lambda v: v.number, reverse=True)
    return GitHubMetadataResult(file_path=target, issues=issues, prs=prs)


# ─── live wrapper (injected session) ─────────────────────────────────────────


def query_github_metadata(
    repository_id: int,
    target: str | None,
    session: Session,
) -> GitHubMetadataResult:
    """Return the cached GitHub issues/PRs for ``repository_id`` (SDD §10
    GitHub Metadata Loader), optionally restricted to those linked to ``target``.

    *Live read*: when ``target`` is given and resolvable, restricts to ``issues``
    rows whose ``linked_files`` array contains the resolved POSIX-rel path;
    otherwise returns all the repo's cached rows. Delegates the issues/PRs
    split + sorting to :func:`build_github_metadata`. Injected ``Session``, owns
    no transaction.

    An unresolvable ``target`` (resolver returns ``None``) yields a clear-empty
    :class:`GitHubMetadataResult` — the same "don't guess" posture the sibling
    context tools hold. An empty/``None`` ``target`` is the whole-repo query
    (no file-path filter), and ``file_path`` is ``None`` in that case.

    The ``linked_files`` membership uses :meth:`sqlalchemy.types.ARRAY`-aware
    ``Issue.linked_files.any(target_path)``; for rows where ``linked_files`` is
    NULL the predicate evaluates to NULL (filtered out by WHERE), so a
    target-filtered query never misreads a NULL linked list as a match.
    """
    target_s = (target or "").strip()
    target_path: str | None = None
    if target_s:
        file_id = resolve_file_id(repository_id, target_s, session)
        if file_id is None:
            return GitHubMetadataResult(file_path=None)
        # Resolve to the canonical POSIX-rel path the indexer would have written
        # into `linked_files` — re-resolving (rather than trusting the Planner's
        # raw `target`) so a module-name or substring plan arg still matches
        # path-form `linked_files` entries.
        target_path = session.scalar(
            sa.select(File.path).where(File.id == file_id)
        )
        if target_path is None:
            return GitHubMetadataResult(file_path=None)

    stmt = sa.select(
        Issue.number,
        Issue.title,
        Issue.state,
        Issue.labels,
        Issue.url,
        Issue.linked_files,
    ).where(Issue.repository_id == repository_id)
    if target_path is not None:
        stmt = stmt.where(Issue.linked_files.any(target_path))

    rows = list(session.execute(stmt).all())
    return build_github_metadata(rows, target_path=target_path)


__all__ = [
    "GitHubMetadataResult",
    "IssueView",
    "build_github_metadata",
    "query_github_metadata",
]
