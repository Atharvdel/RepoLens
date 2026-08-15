"""Structured search endpoints (SDD §12).

Routes:
- GET /repositories/{id}/search     Search symbols, files, or ripgrep text in repository
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_session
from app.indexing.pipeline import resolve_or_clone_repository
from app.models import Repository
from app.tools.file_search import search_files
from app.tools.symbol_search import search_symbols
from app.tools.text_search import search_text

router = APIRouter(prefix="/repositories/{repo_id}", tags=["search"])


# ─── Pydantic schemas ────────────────────────────────────────────────────────


class SearchHit(BaseModel):
    type: str  # "symbol" | "file" | "text"
    item: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    search_type: str
    total_hits: int
    hits: list[dict[str, Any]]


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.get("/search", response_model=SearchResponse)
def execute_search(
    repo_id: int,
    q: str = Query(..., min_length=1, description="Search query string"),
    type: str = Query("symbol", pattern="^(symbol|file|text)$", description="Type of search to execute"),
    kind: Optional[str] = Query(None, description="Optional symbol kind filter ('class', 'function', 'method', 'variable')"),
    regex: bool = Query(False, description="Treat query as regular expression (text search only)"),
    cap: int = Query(50, ge=1, le=200, description="Max results cap"),
    session: Session = Depends(get_session),
):
    """Execute a structured symbol search, file search, or ripgrep text search."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    hits: list[dict[str, Any]] = []

    if type == "symbol":
        sym_results = search_symbols(
            query=q,
            repository_id=repo_id,
            session=session,
            kind=kind,
        )
        hits = [dataclasses.asdict(s) for s in sym_results[:cap]]

    elif type == "file":
        file_results = search_files(
            query=q,
            repository_id=repo_id,
            session=session,
        )
        hits = [dataclasses.asdict(f) for f in file_results[:cap]]

    elif type == "text":
        try:
            root_path = resolve_or_clone_repository(repo.url_or_path, repo.id, repo.name)
            text_hits = search_text(
                query=q,
                repository_id=repo_id,
                repo_root=root_path,
                regex=regex,
                cap=cap,
            )
            hits = [dataclasses.asdict(t) for t in text_hits]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text search error: {str(e)}")

    return SearchResponse(
        query=q,
        search_type=type,
        total_hits=len(hits),
        hits=hits,
    )
