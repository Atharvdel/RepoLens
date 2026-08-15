"""GitHub metadata sync stage of the RepoLens indexing pipeline (SDD §7 step 8, §10).

Uses PyGithub to fetch issues and pull requests metadata (numbers, titles,
labels, states, URLs, and linked files) and caches them in the `issues` table.

Follows the project's design principles:
- Caching: Pulled once at index time so queries do not hit GitHub API (SDD §11).
- Graceful degradation: If no token is provided, token is invalid, or rate limits
  are exceeded, logs a warning and returns gracefully without failing the pipeline (SDD §18).
- Does NOT commit: Caller owns the transaction (SDD §7 step 9).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import File, Issue

logger = logging.getLogger(__name__)

DEFAULT_MAX_ISSUES: int = 100


@dataclass
class ExtractedIssue:
    """GitHub issue or PR metadata extracted via PyGithub."""

    number: int
    title: str
    state: str
    labels: list[str]
    url: str
    linked_files: list[str] | None


def extract_linked_files(text: str | None, known_file_paths: set[str]) -> list[str] | None:
    """Find mentions of known repository files in an issue/PR description or comment.

    Pure function: checks word/path tokens against known repo files.
    """
    if not text or not known_file_paths:
        return None

    matched: set[str] = set()

    # Match tokens or backticked expressions `path/to/file.py`
    for path in known_file_paths:
        # Check if path or basename is explicitly mentioned
        if path in text:
            matched.add(path)
        else:
            base = os.path.basename(path)
            if len(base) > 4 and f"`{base}`" in text:
                matched.add(path)

    return sorted(matched) if matched else None


def fetch_github_metadata(
    owner: str,
    repo_name: str,
    token: str | None = None,
    known_files: set[str] | None = None,
    max_items: int = DEFAULT_MAX_ISSUES,
) -> list[ExtractedIssue]:
    """Connect to GitHub API via PyGithub and fetch issues and PRs.

    Returns empty list on missing credentials, network failure, or rate limit.
    """
    try:
        from github import Auth, Github, GithubException
    except ImportError:
        logger.warning("PyGithub not installed; skipping GitHub metadata sync")
        return []

    auth_token = token or os.getenv("GITHUB_TOKEN")
    try:
        if auth_token:
            auth = Auth.Token(auth_token)
            gh = Github(auth=auth)
        else:
            gh = Github()

        repo = gh.get_repo(f"{owner}/{repo_name}")
        items = repo.get_issues(state="all", sort="updated", direction="desc")
    except Exception as e:
        logger.warning("Failed to access GitHub repository %s/%s: %s", owner, repo_name, e)
        return []

    results: list[ExtractedIssue] = []
    known = known_files or set()

    try:
        count = 0
        for issue in items:
            if count >= max_items:
                break

            labels = [lbl.name for lbl in issue.labels] if issue.labels else []
            body = issue.body or ""
            linked = extract_linked_files(body, known)

            results.append(
                ExtractedIssue(
                    number=issue.number,
                    title=issue.title or f"Issue #{issue.number}",
                    state=issue.state or "open",
                    labels=labels,
                    url=issue.html_url or f"https://github.com/{owner}/{repo_name}/issues/{issue.number}",
                    linked_files=linked,
                )
            )
            count += 1
    except Exception as e:
        logger.warning("Error reading issues from GitHub for %s/%s: %s", owner, repo_name, e)

    return results


def sync_github_metadata(
    repository_id: int,
    owner: str | None,
    repo_name: str | None,
    session: Session,
    token: str | None = None,
    max_items: int = DEFAULT_MAX_ISSUES,
) -> int:
    """Fetch GitHub issues/PRs and populate the `issues` table for ``repository_id``.

    Does NOT commit -- caller owns the transaction.
    Returns number of issues indexed.
    """
    if not owner or not repo_name:
        return 0

    # Read known files for linking
    paths = session.scalars(
        sa.select(File.path).where(File.repository_id == repository_id)
    ).all()
    known_files = set(paths)

    extracted = fetch_github_metadata(
        owner=owner,
        repo_name=repo_name,
        token=token,
        known_files=known_files,
        max_items=max_items,
    )

    count = 0
    for item in extracted:
        row = Issue(
            repository_id=repository_id,
            number=item.number,
            title=item.title,
            state=item.state,
            labels=item.labels,
            url=item.url,
            linked_files=item.linked_files,
        )
        session.add(row)
        count += 1

    return count
