"""File detail, documents, and issues metadata endpoints (SDD §12).

Routes:
- GET /repositories/{id}/files/{path:path}   File detail (symbols, imports, referencing files, history, content)
- GET /repositories/{id}/issues              Cached GitHub issues and PRs (optionally linked to file)
- GET /repositories/{id}/documents           List and retrieve cached documentation files
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_session
from app.indexing.pipeline import resolve_or_clone_repository
from app.models import Document, Edge, File, Issue, Repository, Symbol
from app.tools.file_history import query_file_history
from app.tools.github_metadata import query_github_metadata

router = APIRouter(prefix="/repositories/{repo_id}", tags=["files"])


# ─── Pydantic schemas ────────────────────────────────────────────────────────


class SymbolItem(BaseModel):
    id: int
    name: str
    kind: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    parent_symbol_id: Optional[int] = None


class FileDetailResponse(BaseModel):
    id: int
    repository_id: int
    path: str
    language: str
    loc: int
    last_modified: Optional[str] = None
    symbols: list[SymbolItem] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    referenced_by: list[str] = Field(default_factory=list)
    top_contributors: list[dict[str, Any]] = Field(default_factory=list)
    recent_commits: list[dict[str, Any]] = Field(default_factory=list)
    content: Optional[str] = None


class DocumentItem(BaseModel):
    id: int
    path: str
    title: Optional[str] = None
    content: str


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.get("/files/{file_path:path}", response_model=FileDetailResponse)
def get_file_detail(
    repo_id: int,
    file_path: str,
    include_content: bool = Query(True, description="Include raw file content"),
    session: Session = Depends(get_session),
):
    """Retrieve detailed file metadata, symbols, imports, referencing files, and git history."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    # 1. Clean line numbers and slashes
    norm_path = file_path.strip().replace("\\", "/").split(":")[0].strip().rstrip("/")

    # 2. Try exact path match
    file_row = session.scalars(
        sa.select(File)
        .where(File.repository_id == repo_id)
        .where(File.path == norm_path)
    ).first()

    # 3. Try suffix match (e.g. "app.py" -> "src/flask/app.py")
    if not file_row:
        file_row = session.scalars(
            sa.select(File)
            .where(File.repository_id == repo_id)
            .where(sa.or_(File.path.endswith(f"/{norm_path}"), File.path.endswith(norm_path)))
        ).first()

    # 4. Try directory/module match (e.g. "src/flask" or "examples/tutorial/flaskr")
    if not file_row:
        dir_files = session.scalars(
            sa.select(File)
            .where(File.repository_id == repo_id)
            .where(File.path.like(f"{norm_path}/%"))
            .order_by(File.loc.desc())
        ).all()

        if dir_files:
            # Pick entrypoint: __init__.py, index.ts, app.py, main.py, or largest file
            file_row = next(
                (
                    f
                    for f in dir_files
                    if f.path.endswith("__init__.py")
                    or f.path.endswith("index.ts")
                    or f.path.endswith("app.py")
                    or f.path.endswith("main.py")
                ),
                dir_files[0],
            )

    # 5. Try substring match (e.g. "sansio/blueprints" -> "src/flask/sansio/blueprints.py")
    if not file_row:
        file_row = session.scalars(
            sa.select(File)
            .where(File.repository_id == repo_id)
            .where(File.path.like(f"%{norm_path}%"))
        ).first()

    if not file_row:
        raise HTTPException(status_code=404, detail=f"File or module '{norm_path}' not found in repository")

    # Symbols in this file
    sym_rows = session.scalars(
        sa.select(Symbol)
        .where(Symbol.file_id == file_row.id)
        .order_by(Symbol.line_start.asc())
    ).all()
    symbols = [
        SymbolItem(
            id=s.id,
            name=s.name,
            kind=s.kind,
            line_start=s.line_start,
            line_end=s.line_end,
            docstring=s.docstring,
            parent_symbol_id=s.parent_symbol_id,
        )
        for s in sym_rows
    ]

    # Outgoing imports
    import_edges = session.execute(
        sa.select(Edge, File.path)
        .outerjoin(File, Edge.target_id == File.id)
        .where(Edge.repository_id == repo_id)
        .where(Edge.source_id == file_row.id)
        .where(Edge.edge_type == "imports")
    ).all()
    imports_list = [
        target_path or e.target_label or "external"
        for e, target_path in import_edges
    ]

    # Incoming references (who imports or references this file)
    ref_edges = session.execute(
        sa.select(File.path)
        .join(Edge, Edge.source_id == File.id)
        .where(Edge.repository_id == repo_id)
        .where(Edge.target_id == file_row.id)
        .distinct()
    ).scalars().all()

    # Git history & contributors
    history = query_file_history(
        target=file_row.path,
        repository_id=repo_id,
        session=session,
    )
    contributors = [dataclasses.asdict(c) for c in history.top_contributors]
    recent_commits = [dataclasses.asdict(cm) for cm in history.recent_commits]

    # Read content from disk if requested
    raw_content = None
    if include_content:
        try:
            root_path = resolve_or_clone_repository(repo.url_or_path, repo.id, repo.name)
            disk_path = root_path / file_row.path
            if disk_path.is_file():
                raw_content = disk_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            raw_content = None

    return FileDetailResponse(
        id=file_row.id,
        repository_id=file_row.repository_id,
        path=file_row.path,
        language=file_row.language or "unknown",
        loc=file_row.loc or 0,
        last_modified=file_row.last_modified.isoformat() if file_row.last_modified else None,
        symbols=symbols,
        imports=sorted(set(imports_list)),
        referenced_by=sorted(set(ref_edges)),
        top_contributors=contributors,
        recent_commits=recent_commits,
        content=raw_content,
    )


@router.get("/issues")
def get_repository_issues(
    repo_id: int,
    target: Optional[str] = Query(None, description="Optional file path to filter linked issues/PRs"),
    session: Session = Depends(get_session),
):
    """Retrieve cached GitHub issues and pull requests for the repository."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    result = query_github_metadata(
        target=target,
        repository_id=repo_id,
        session=session,
    )
    return dataclasses.asdict(result)


@router.get("/documents", response_model=list[DocumentItem])
def list_documents(repo_id: int, session: Session = Depends(get_session)):
    """Retrieve all parsed documentation files for this repository (README, guides, etc.)."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    docs = session.scalars(
        sa.select(Document)
        .where(Document.repository_id == repo_id)
        .order_by(Document.path.asc())
    ).all()

    return [
        DocumentItem(
            id=d.id,
            path=d.path,
            title=d.title,
            content=d.content,
        )
        for d in docs
    ]
