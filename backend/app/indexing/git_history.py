"""Git history indexing stage of the RepoLens pipeline (SDD §7 step 7).

Walks the git commit history using GitPython, extracts commit metadata
(hash, author, date, message), and maps each commit to the files it touched via
the `commits` table and `file_commits` join table (SDD §11).

This backs `app.tools.file_history` so queries about file contributors,
last modified timestamps, and recent commits are answered from cached rows
without re-walking `git log` on every question.

Follows the project's indexing convention:
- Pure/disk-inspection helper `extract_commits_and_touches(root, path_to_id_map)`
- Persistence helper `index_git_history(root, repository_id, session)`
- Caller owns the transaction (does not commit).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Commit, File, file_commits

logger = logging.getLogger(__name__)

# Default cap on commits scanned during indexing to maintain low single-digit minute latency (SDD §7, §17)
DEFAULT_MAX_COMMITS: int = 500


@dataclass
class ExtractedCommit:
    """Commit record extracted from Git repository."""

    hash: str
    author: str
    date: datetime
    message: str
    touched_file_ids: list[int]


def extract_commits_and_touches(
    root: Path | str,
    path_to_id: dict[str, int],
    max_commits: int = DEFAULT_MAX_COMMITS,
) -> list[ExtractedCommit]:
    """Inspect the git repository at ``root`` and return commits mapped to known file IDs.

    Returns an empty list if ``root`` is not a git repository or has no commits.
    """
    root_path = Path(root).resolve()
    try:
        import git
        from git.exc import InvalidGitRepositoryError, NoSuchPathError
    except ImportError:
        logger.warning("GitPython not available; skipping git history scan")
        return []

    try:
        repo = git.Repo(root_path)
    except (InvalidGitRepositoryError, NoSuchPathError, Exception):
        logger.info("Path %s is not a valid git repository; skipping git history scan", root_path)
        return []

    try:
        # Check if repo has any commits (e.g. empty repo has no HEAD)
        if not repo.heads and not repo.branches:
            try:
                _ = repo.head.commit
            except Exception:
                return []
    except Exception:
        return []

    extracted: list[ExtractedCommit] = []
    seen_hashes: set[str] = set()

    try:
        commits = list(repo.iter_commits(max_count=max_commits))
    except Exception as e:
        logger.warning("Failed to iterate commits in %s: %s", root_path, e)
        return []

    for commit in commits:
        if commit.hexsha in seen_hashes:
            continue
        seen_hashes.add(commit.hexsha)

        author_name = commit.author.name if commit.author and commit.author.name else (commit.author.email if commit.author else "Unknown")
        # Commit date converted to naive UTC datetime for SQLAlchemy DateTime column
        committed_dt = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc).replace(tzinfo=None)
        message = commit.message.strip() if commit.message else ""

        touched_file_ids: list[int] = []

        try:
            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit)
                for diff in diffs:
                    paths = [diff.b_path, diff.a_path]
                    for p in paths:
                        if p:
                            norm_p = p.replace("\\", "/")
                            if norm_p in path_to_id:
                                fid = path_to_id[norm_p]
                                if fid not in touched_file_ids:
                                    touched_file_ids.append(fid)
            else:
                # Root commit: all blobs in tree
                for item in commit.tree.traverse():
                    if item.type == "blob":
                        norm_p = item.path.replace("\\", "/")
                        if norm_p in path_to_id:
                            fid = path_to_id[norm_p]
                            if fid not in touched_file_ids:
                                touched_file_ids.append(fid)
        except Exception as e:
            logger.debug("Failed to extract touched files for commit %s: %s", commit.hexsha, e)

        extracted.append(
            ExtractedCommit(
                hash=commit.hexsha,
                author=author_name,
                date=committed_dt,
                message=message,
                touched_file_ids=touched_file_ids,
            )
        )

    return extracted


def index_git_history(
    root: Path | str,
    repository_id: int,
    session: Session,
    max_commits: int = DEFAULT_MAX_COMMITS,
) -> tuple[int, int]:
    """Scan git log at ``root``, populate `commits` and `file_commits` in ``session``.

    Does NOT commit -- caller owns the transaction.
    Returns `(commits_indexed, file_commit_links_indexed)`.
    """
    # Fetch file map from DB for this repository
    file_rows = session.execute(
        sa.select(File.id, File.path).where(File.repository_id == repository_id)
    ).all()
    path_to_id: dict[str, int] = {row.path: row.id for row in file_rows}

    if not path_to_id:
        return 0, 0

    extracted = extract_commits_and_touches(root, path_to_id, max_commits=max_commits)
    if not extracted:
        return 0, 0

    commits_count = 0
    links_count = 0

    for item in extracted:
        commit_row = Commit(
            repository_id=repository_id,
            hash=item.hash,
            author=item.author,
            date=item.date,
            message=item.message,
        )
        session.add(commit_row)
        session.flush()  # assign commit_row.id
        commits_count += 1

        for fid in item.touched_file_ids:
            session.execute(
                file_commits.insert().values(file_id=fid, commit_id=commit_row.id)
            )
            links_count += 1

    return commits_count, links_count
