"""Automated indexing pipeline orchestrator (SDD §7, §16).

Coordinates the 9-step batch indexing process for a repository:
1. Clone / Pull (git URL or local filesystem path)
2. Walk & Filter files (app.indexing.walker)
3. AST Parse for symbols (app.indexing.parser + app.indexing.ts_js_parser)
4. Build Import Graph (app.indexing.import_graph)
5. Build Reference Index (app.indexing.reference_index via ripgrep)
6. Parse Documentation (app.indexing.doc_indexer)
7. Git History Scan (app.indexing.git_history)
8. GitHub Metadata Sync (app.indexing.github_sync)
9. Transactional Commit & Status Update (indexing -> ready / failed)

Provides both synchronous `run_indexing_pipeline` and background task runners.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.indexing.doc_indexer import index_documents
from app.indexing.git_history import index_git_history
from app.indexing.github_sync import sync_github_metadata
from app.indexing.import_graph import ImportResolver, index_file_imports, parse_imports
from app.indexing.parser import parse_and_index_file
from app.indexing.reference_index import find_references, index_symbol_references
from app.indexing.walker import walk_repository
from app.models import (
    Commit,
    Document,
    Edge,
    File,
    Issue,
    Repository,
    Symbol,
    file_commits,
)

logger = logging.getLogger(__name__)

# Base directory where remote repositories are cloned locally
MANAGED_REPOS_DIR = Path(os.getenv("MANAGED_REPOS_DIR", "data/repos"))


@dataclass
class PipelineProgress:
    """Tracks progress through the 9 indexing pipeline stages."""

    stage: str = "initialized"  # "cloning" | "walking" | "parsing" | "imports" | "references" | "docs" | "git" | "github" | "ready" | "failed"
    step: int = 0
    total_steps: int = 9
    files_indexed: int = 0
    symbols_indexed: int = 0
    import_edges_indexed: int = 0
    reference_edges_indexed: int = 0
    docs_indexed: int = 0
    commits_indexed: int = 0
    issues_indexed: int = 0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


def _parse_github_owner_repo(url_or_path: str) -> tuple[str | None, str | None]:
    """Extract (owner, repo) from a GitHub URL if applicable."""
    # Matches https://github.com/owner/repo or git@github.com:owner/repo
    m = re.search(r"github\.com[/:]([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+?)(?:\.git)?$", url_or_path.strip())
    if m:
        return m.group(1), m.group(2)
    return None, None


def _is_git_url(url_or_path: str) -> bool:
    s = url_or_path.strip().lower()
    return s.startswith("http://") or s.startswith("https://") or s.startswith("git@") or s.startswith("ssh://")


def clear_repository_data(repository_id: int, session: Session) -> None:
    """Clear all derived indexing data for a repository in foreign-key safe order."""
    # 1. file_commits join rows
    file_ids = session.scalars(
        sa.select(File.id).where(File.repository_id == repository_id)
    ).all()
    if file_ids:
        session.execute(file_commits.delete().where(file_commits.c.file_id.in_(file_ids)))

    # 2. symbols
    if file_ids:
        session.execute(sa.delete(Symbol).where(Symbol.file_id.in_(file_ids)))

    # 3. edges
    session.execute(sa.delete(Edge).where(Edge.repository_id == repository_id))

    # 4. documents
    session.execute(sa.delete(Document).where(Document.repository_id == repository_id))

    # 5. commits
    session.execute(sa.delete(Commit).where(Commit.repository_id == repository_id))

    # 6. issues
    session.execute(sa.delete(Issue).where(Issue.repository_id == repository_id))

    # 7. files
    session.execute(sa.delete(File).where(File.repository_id == repository_id))

    session.flush()


def resolve_or_clone_repository(
    url_or_path: str,
    repository_id: int,
    repo_name: str,
) -> Path:
    """Ensure repository source files are available locally.

    If local directory, returns Path. If remote git URL, clones/pulls into MANAGED_REPOS_DIR.
    """
    if not _is_git_url(url_or_path):
        local_p = Path(url_or_path).resolve()
        if not local_p.is_dir():
            raise FileNotFoundError(f"Local repository path does not exist: {local_p}")
        return local_p

    # Remote git repository
    import git

    MANAGED_REPOS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_name)
    target_dir = MANAGED_REPOS_DIR / f"{repository_id}_{safe_name}"

    if target_dir.is_dir() and (target_dir / ".git").is_dir():
        logger.info("Pulling latest changes in %s", target_dir)
        try:
            repo = git.Repo(target_dir)
            repo.remotes.origin.pull()
            return target_dir
        except Exception as e:
            logger.warning("Pull failed; attempting fresh clone: %s", e)
            shutil.rmtree(target_dir, ignore_errors=True)

    logger.info("Cloning %s into %s", url_or_path, target_dir)
    git.Repo.clone_from(url_or_path, target_dir)
    return target_dir


def run_indexing_pipeline(
    repository_id: int,
    session: Session,
    github_token: str | None = None,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
) -> PipelineProgress:
    """Execute the complete 9-step indexing pipeline for a repository.

    Manages transaction boundaries: clears old data, runs all indexing stages,
    updates status to 'ready' and commits. On failure, updates status to 'failed'.
    """
    progress = PipelineProgress()

    def _notify(stage: str, step: int):
        progress.stage = stage
        progress.step = step
        if progress_callback:
            progress_callback(progress)

    # Fetch repository record
    repo = session.get(Repository, repository_id)
    if not repo:
        raise ValueError(f"Repository with id {repository_id} not found")

    try:
        # Mark indexing
        repo.status = "indexing"
        session.commit()

        # Step 1: Clone / Pull
        _notify("cloning", 1)
        root_path = resolve_or_clone_repository(repo.url_or_path, repo.id, repo.name)

        # Infer owner/repo if not already set
        if not repo.github_owner or not repo.github_repo:
            owner, gh_repo = _parse_github_owner_repo(repo.url_or_path)
            if owner and gh_repo:
                repo.github_owner = owner
                repo.github_repo = gh_repo
                session.flush()

        # Clear existing data for idempotency / re-indexing
        clear_repository_data(repository_id, session)

        # Step 2: File Walk & Filter
        _notify("walking", 2)
        n_files = walk_repository(root_path, repository_id, session)
        session.flush()
        progress.files_indexed = n_files

        # Step 3: Parse AST for Symbols
        _notify("parsing", 3)
        file_rows = session.scalars(
            sa.select(File).where(File.repository_id == repository_id)
        ).all()

        total_symbols = 0
        for f in file_rows:
            full_file_path = root_path / f.path
            if full_file_path.is_file():
                try:
                    added = parse_and_index_file(full_file_path, f.id, session)
                    total_symbols += added
                except Exception as e:
                    logger.debug("Parse error on %s: %s", f.path, e)
        session.flush()
        progress.symbols_indexed = total_symbols

        # Step 4: Build Import Graph
        _notify("imports", 4)
        path_to_id = {f.path: f.id for f in file_rows}
        resolver_rows = [(f.id, f.path) for f in file_rows]
        resolver = ImportResolver(resolver_rows)

        total_import_edges = 0
        for f in file_rows:
            full_file_path = root_path / f.path
            if full_file_path.is_file():
                try:
                    parsed_imps = parse_imports(full_file_path)
                    res = index_file_imports(
                        f.id, f.path, parsed_imps, resolver, repository_id, session
                    )
                    total_import_edges += res.added
                except Exception as e:
                    logger.warning("Import index error on %s: %s", f.path, e)
        session.flush()
        progress.import_edges_indexed = total_import_edges

        # Step 5: Build Reference Index
        _notify("references", 5)
        # Fetch symbols joined with file path
        sym_query = (
            sa.select(Symbol, File.path)
            .join(File, Symbol.file_id == File.id)
            .where(File.repository_id == repository_id)
        )
        symbols_with_paths = session.execute(sym_query).all()

        total_ref_edges = 0
        for sym, file_path in symbols_with_paths:
            try:
                res = index_symbol_references(
                    root=root_path,
                    symbol=sym,
                    def_file_path=file_path,
                    path_to_id=path_to_id,
                    repository_id=repository_id,
                    session=session,
                )
                total_ref_edges += res.edges_created
            except Exception as e:
                logger.debug("Reference index error for %s: %s", sym.name, e)
        session.flush()
        progress.reference_edges_indexed = total_ref_edges

        # Step 6: Parse Documentation
        _notify("docs", 6)
        docs_count = index_documents(root_path, repository_id, session)
        session.flush()
        progress.docs_indexed = docs_count

        # Step 7: Git History Scan
        _notify("git", 7)
        commits_cnt, _ = index_git_history(root_path, repository_id, session)
        session.flush()
        progress.commits_indexed = commits_cnt

        # Step 8: GitHub Metadata Sync
        _notify("github", 8)
        issues_cnt = sync_github_metadata(
            repository_id=repository_id,
            owner=repo.github_owner,
            repo_name=repo.github_repo,
            session=session,
            token=github_token,
        )
        session.flush()
        progress.issues_indexed = issues_cnt

        # Step 9: Transactional Commit & Mark Ready
        _notify("ready", 9)
        repo.status = "ready"
        repo.indexed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        session.commit()

        progress.completed_at = datetime.now(timezone.utc)
        logger.info(
            "Successfully indexed repo %s: %d files, %d symbols, %d import edges, %d ref edges, %d docs, %d commits, %d issues",
            repo.name,
            n_files,
            total_symbols,
            total_import_edges,
            total_ref_edges,
            docs_count,
            commits_cnt,
            issues_cnt,
        )
        return progress

    except Exception as e:
        session.rollback()
        err_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("Indexing failed for repo %s (id=%d): %s\n%s", repo.name, repository_id, err_msg, traceback.format_exc())
        progress.stage = "failed"
        progress.error = err_msg
        try:
            # Mark failed status in separate transaction
            repo.status = "failed"
            session.commit()
        except Exception:
            session.rollback()
        raise e
