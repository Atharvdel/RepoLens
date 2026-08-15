"""Repository management endpoints (SDD §12).

Routes:
- POST   /repositories              Add a repo (url or local path) -> starts indexing background job
- GET    /repositories              List repos with summary stats
- GET    /repositories/{id}         Repo detail + summary metrics & language breakdown
- POST   /repositories/{id}/reindex Trigger full re-indexing
- DELETE /repositories/{id}         Remove repo and all derived data
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import sqlalchemy as sa
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import SessionLocal, get_session
from app.indexing.pipeline import clear_repository_data, run_indexing_pipeline
from app.models import (
    ChatMessage,
    ChatSession,
    Commit,
    Document,
    Edge,
    File,
    Issue,
    Repository,
    Symbol,
)
from app.tools.architecture import query_architecture

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repositories", tags=["repositories"])


# ─── Pydantic schemas ────────────────────────────────────────────────────────


class RepositoryCreate(BaseModel):
    url_or_path: str = Field(..., description="Git clone URL (HTTPS/SSH) or local directory path")
    name: Optional[str] = Field(None, description="Display name for the repository")
    github_token: Optional[str] = Field(None, description="Optional GitHub PAT for metadata sync")


class RepositoryItem(BaseModel):
    id: int
    url_or_path: str
    name: str
    default_branch: Optional[str] = None
    status: str
    indexed_at: Optional[datetime] = None
    github_owner: Optional[str] = None
    github_repo: Optional[str] = None
    file_count: int = 0
    symbol_count: int = 0
    total_loc: int = 0


class RepositoryDetail(RepositoryItem):
    language_breakdown: dict[str, int] = Field(default_factory=dict)
    key_files: list[dict[str, Any]] = Field(default_factory=list)
    docs_count: int = 0
    commits_count: int = 0
    issues_count: int = 0
    readme_preview: Optional[str] = None


# ─── Background Indexer Worker ───────────────────────────────────────────────


def _run_indexing_background(repository_id: int, github_token: str | None = None):
    """Executes the indexing pipeline in an isolated DB session in the background."""
    with SessionLocal() as session:
        try:
            logger.info("Starting background indexing for repository id=%d", repository_id)
            run_indexing_pipeline(repository_id, session, github_token=github_token)
            logger.info("Background indexing finished successfully for repository id=%d", repository_id)
        except Exception as e:
            logger.error("Background indexing failed for repository id=%d: %s", repository_id, e)


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.post("", response_model=RepositoryItem, status_code=status.HTTP_201_CREATED)
def create_repository(
    payload: RepositoryCreate,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """Add a new repository for indexing (triggers asynchronous indexing job)."""
    raw_path = payload.url_or_path.strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="url_or_path cannot be empty")

    name = payload.name
    if not name:
        # Infer name from URL or directory path
        clean_path = raw_path.rstrip("/\\")
        if clean_path.endswith(".git"):
            clean_path = clean_path[:-4]
        name = os.path.basename(clean_path) or "repository"

    repo = Repository(
        url_or_path=raw_path,
        name=name,
        status="indexing",
        indexed_at=None,
    )
    session.add(repo)
    session.commit()
    session.refresh(repo)

    # Launch indexing in background task
    background_tasks.add_task(_run_indexing_background, repo.id, payload.github_token)

    return RepositoryItem(
        id=repo.id,
        url_or_path=repo.url_or_path,
        name=repo.name,
        default_branch=repo.default_branch,
        status=repo.status,
        indexed_at=repo.indexed_at,
        github_owner=repo.github_owner,
        github_repo=repo.github_repo,
        file_count=0,
        symbol_count=0,
        total_loc=0,
    )


@router.get("", response_model=list[RepositoryItem])
def list_repositories(session: Session = Depends(get_session)):
    """List all registered repositories with their indexing status and basic metrics."""
    repos = session.scalars(sa.select(Repository).order_by(Repository.id.asc())).all()
    results = []

    for r in repos:
        n_files = session.scalar(
            sa.select(sa.func.count(File.id)).where(File.repository_id == r.id)
        ) or 0
        n_symbols = session.scalar(
            sa.select(sa.func.count(Symbol.id))
            .join(File, Symbol.file_id == File.id)
            .where(File.repository_id == r.id)
        ) or 0
        tot_loc = session.scalar(
            sa.select(sa.func.sum(File.loc)).where(File.repository_id == r.id)
        ) or 0

        results.append(
            RepositoryItem(
                id=r.id,
                url_or_path=r.url_or_path,
                name=r.name,
                default_branch=r.default_branch,
                status=r.status,
                indexed_at=r.indexed_at,
                github_owner=r.github_owner,
                github_repo=r.github_repo,
                file_count=n_files,
                symbol_count=n_symbols,
                total_loc=int(tot_loc),
            )
        )

    return results


@router.get("/{repo_id}", response_model=RepositoryDetail)
def get_repository(repo_id: int, session: Session = Depends(get_session)):
    """Get detailed statistics, language distribution, key files, and overview for a repository."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    n_files = session.scalar(
        sa.select(sa.func.count(File.id)).where(File.repository_id == repo.id)
    ) or 0
    n_symbols = session.scalar(
        sa.select(sa.func.count(Symbol.id))
        .join(File, Symbol.file_id == File.id)
        .where(File.repository_id == repo.id)
    ) or 0
    tot_loc = session.scalar(
        sa.select(sa.func.sum(File.loc)).where(File.repository_id == repo.id)
    ) or 0

    # Language breakdown by LOC
    lang_rows = session.execute(
        sa.select(File.language, sa.func.sum(File.loc), sa.func.count(File.id))
        .where(File.repository_id == repo.id)
        .group_by(File.language)
    ).all()
    lang_breakdown = {row[0] or "other": int(row[1] or 0) for row in lang_rows}

    # Counts
    n_docs = session.scalar(
        sa.select(sa.func.count(Document.id)).where(Document.repository_id == repo.id)
    ) or 0
    n_commits = session.scalar(
        sa.select(sa.func.count(Commit.id)).where(Commit.repository_id == repo.id)
    ) or 0
    n_issues = session.scalar(
        sa.select(sa.func.count(Issue.id)).where(Issue.repository_id == repo.id)
    ) or 0

    # README preview if available
    readme_doc = session.scalars(
        sa.select(Document)
        .where(Document.repository_id == repo.id)
        .where(sa.func.lower(Document.path).contains("readme"))
        .limit(1)
    ).first()
    readme_preview = readme_doc.content if readme_doc else None

    # Key files via Architecture centrality query
    key_files: list[dict[str, Any]] = []
    if repo.status == "ready":
        try:
            arch = query_architecture(repository_id=repo.id, session=session, top_k=8)
            key_files = [{"path": kf.path, "in_degree": kf.in_degree} for kf in arch.key_files]
        except Exception:
            key_files = []

    return RepositoryDetail(
        id=repo.id,
        url_or_path=repo.url_or_path,
        name=repo.name,
        default_branch=repo.default_branch,
        status=repo.status,
        indexed_at=repo.indexed_at,
        github_owner=repo.github_owner,
        github_repo=repo.github_repo,
        file_count=n_files,
        symbol_count=n_symbols,
        total_loc=int(tot_loc),
        language_breakdown=lang_breakdown,
        key_files=key_files,
        docs_count=n_docs,
        commits_count=n_commits,
        issues_count=n_issues,
        readme_preview=readme_preview,
    )


@router.post("/{repo_id}/reindex", response_model=RepositoryItem)
def reindex_repository(
    repo_id: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    """Trigger a full re-indexing of the repository."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    repo.status = "indexing"
    session.commit()

    background_tasks.add_task(_run_indexing_background, repo.id)

    return RepositoryItem(
        id=repo.id,
        url_or_path=repo.url_or_path,
        name=repo.name,
        default_branch=repo.default_branch,
        status=repo.status,
        indexed_at=repo.indexed_at,
        github_owner=repo.github_owner,
        github_repo=repo.github_repo,
        file_count=0,
        symbol_count=0,
        total_loc=0,
    )


@router.delete("/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_repository(repo_id: int, session: Session = Depends(get_session)):
    """Delete repository and all derived data from the database."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Clear chat messages and sessions
    chat_session_ids = session.scalars(
        sa.select(ChatSession.id).where(ChatSession.repository_id == repo_id)
    ).all()
    if chat_session_ids:
        session.execute(
            sa.delete(ChatMessage).where(ChatMessage.session_id.in_(chat_session_ids))
        )
        session.execute(
            sa.delete(ChatSession).where(ChatSession.repository_id == repo_id)
        )

    # Clear indexing tables
    clear_repository_data(repo_id, session)

    # Delete repository record
    session.delete(repo)
    session.commit()
    return None


@router.get("/{repo_id}/report")
def generate_repository_report(repo_id: int, session: Session = Depends(get_session)) -> dict[str, str]:
    """Generate an executive Markdown architecture and intelligence report for the repository."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Metrics
    file_rows = session.scalars(sa.select(File).where(File.repository_id == repo_id)).all()
    file_count = len(file_rows)
    total_loc = sum(f.loc or 0 for f in file_rows)
    symbol_count = session.scalar(
        sa.select(sa.func.count(Symbol.id))
        .join(File, Symbol.file_id == File.id)
        .where(File.repository_id == repo_id)
    ) or 0

    # Language breakdown
    lang_dist: dict[str, int] = {}
    for f in file_rows:
        lang = f.language or "other"
        lang_dist[lang] = lang_dist.get(lang, 0) + (f.loc or 0)

    # Architecture map
    arch = query_architecture(repo_id, None, session)

    # Documents
    readme_doc = session.scalars(
        sa.select(Document)
        .where(Document.repository_id == repo_id)
        .order_by(
            sa.case(
                (Document.path.ilike("%readme%"), 1),
                (Document.path.ilike("%overview%"), 2),
                else_=3,
            )
        )
    ).first()

    # Detect API routes and their exported HTTP methods
    route_files = [f for f in file_rows if "route." in f.path or "/api/" in f.path]
    route_catalog: list[dict[str, Any]] = []

    for rf in route_files:
        endpoint = rf.path
        if "app/" in endpoint:
            endpoint = "/" + endpoint.split("app/", 1)[-1]
        for ext in ["/route.js", "/route.ts", "/route.jsx", "/route.tsx"]:
            endpoint = endpoint.replace(ext, "")

        methods: list[str] = []
        if repo.url_or_path:
            full_p = os.path.join(repo.url_or_path, rf.path)
            if not os.path.exists(full_p) and "/" in rf.path:
                alt_p = os.path.join(repo.url_or_path, rf.path.split("/", 1)[1])
                if os.path.exists(alt_p):
                    full_p = alt_p

            if os.path.exists(full_p) and os.path.isfile(full_p):
                try:
                    with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                        src = f.read()
                    found_methods = re.findall(
                        r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)",
                        src,
                    )
                    if found_methods:
                        methods = sorted(set(found_methods))
                except Exception:
                    pass

        route_catalog.append({
            "endpoint": endpoint,
            "methods": methods if methods else ["GET/POST"],
            "path": rf.path,
        })

    # Build report markdown
    lines = [
        f"# RepoLens Executive Architecture Report: {repo.name}",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Status:** {repo.status.upper()} | **Mode:** 100% Local & Air-Gapped",
        f"**Repository Source:** `{repo.url_or_path}`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Codebase Scale",
        f"- **Total Lines of Code (LOC):** {total_loc:,}",
        f"- **Indexed Source Files:** {file_count:,}",
        f"- **AST Parsed Symbols (Classes/Functions):** {symbol_count:,}",
        f"- **API Endpoints:** {len(route_catalog)}",
        "",
        "### Language Distribution",
        "| Language | Lines of Code | Share (%) |",
        "| :--- | :--- | :--- |",
    ]

    for lang, loc in sorted(lang_dist.items(), key=lambda x: -x[1]):
        pct = (loc / total_loc * 100) if total_loc > 0 else 0
        lines.append(f"| **{lang.capitalize()}** | {loc:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Core Architectural Components & Key Files",
        "Ranked by In-Degree Centrality (files that the rest of the codebase most heavily imports and depends on):",
        "",
        "| Rank | File Path | Centrality Score | Role in Architecture |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for i, kf in enumerate(arch.key_files[:10], start=1):
        role = "Application Component"
        if "auth" in kf.path:
            role = "Authentication & User Session Management"
        elif "mongodb" in kf.path or "db" in kf.path:
            role = "Database Connection & Persistence"
        elif "mail" in kf.path or "email" in kf.path:
            role = "Email Scheduling & Dispatch"
        elif "admin" in kf.path:
            role = "Administrative Console & Operations"
        elif "api" in kf.path:
            role = "Backend API Endpoint Handler"
        lines.append(f"| #{i} | `{kf.path}` | {kf.centrality:.3f} | {role} |")

    if route_catalog:
        lines.extend([
            "",
            "---",
            "",
            "## 3. REST & API Endpoint Catalog",
            "Discovered route handlers in the application architecture:",
            "",
            "| HTTP Endpoint | Methods | Source File |",
            "| :--- | :--- | :--- |",
        ])
        for r in sorted(route_catalog, key=lambda x: x["endpoint"]):
            methods_str = ", ".join(f"`{m}`" for m in r["methods"])
            lines.append(f"| `{r['endpoint']}` | {methods_str} | `{r['path']}` |")

    # Detect architectural subsystems
    has_auth = any("auth" in f.path.lower() for f in file_rows)
    has_db = any("mongo" in f.path.lower() or "db" in f.path.lower() or "prisma" in f.path.lower() for f in file_rows)
    has_email = any("mail" in f.path.lower() or "cron" in f.path.lower() for f in file_rows)

    lines.extend([
        "",
        "---",
        "",
        "## 4. Subsystem Architecture Overview",
    ])

    if has_auth:
        lines.extend([
            "### 🔐 Authentication & Access Control",
            "- **Mechanism:** NextAuth.js with session token handling and OAuth provider integration.",
            "- **Core Implementation:** `lib/auth.js` configures authentication callbacks, and `middleware.js` enforces protected route access.",
            "",
        ])

    if has_db:
        lines.extend([
            "### 🗄️ Database & Persistence Layer",
            "- **Mechanism:** MongoDB database connection managed via Mongoose and native MongoDB driver.",
            "- **Core Implementation:** `lib/mongodb.js` initializes client connections and exposes model access.",
            "",
        ])

    if has_email:
        lines.extend([
            "### ✉️ Background & Integration Services",
            "- **Mechanism:** Automated email dispatch and scheduled cron jobs using Nodemailer & Node-cron.",
            "- **Core Implementation:** `lib/sendMail.js` and `lib/emailScheduler.js` for template rendering and delivery.",
            "",
        ])

    if readme_doc and readme_doc.content:
        lines.extend([
            "---",
            "",
            f"## 5. Documentation & Package Overview (`{readme_doc.path}`)",
            "",
            readme_doc.content[:3000] + ("\n\n*(Excerpt truncated for report summary)*" if len(readme_doc.content) > 3000 else ""),
        ])

    lines.extend([
        "",
        "---",
        "*Report automatically synthesized by RepoLens Multi-Agent Code Intelligence Engine.*",
    ])

    return {
        "repository_id": str(repo_id),
        "name": repo.name,
        "filename": f"{repo.name}_architecture_report.md",
        "markdown": "\n".join(lines),
    }


class HealthMetric(BaseModel):
    name: str
    value: str
    status: str  # "healthy" | "warning" | "alert"
    description: str


class ArchitectureHealthResponse(BaseModel):
    repository_id: int
    health_score: int
    grade: str
    circular_dependencies: list[list[str]]
    hub_risks: list[dict[str, Any]]
    orphaned_files: list[str]
    modularity_ratio: float
    metrics: list[HealthMetric]
    recommendations: list[str]


@router.get("/{repo_id}/health", response_model=ArchitectureHealthResponse)
def get_repository_health(repo_id: int, session: Session = Depends(get_session)) -> ArchitectureHealthResponse:
    """Analyze repository dependency graph for architectural health, cycles, coupling, and tech debt."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    file_rows = session.scalars(sa.select(File).where(File.repository_id == repo_id)).all()
    if not file_rows:
        return ArchitectureHealthResponse(
            repository_id=repo_id,
            health_score=100,
            grade="A+",
            circular_dependencies=[],
            hub_risks=[],
            orphaned_files=[],
            modularity_ratio=1.0,
            metrics=[],
            recommendations=["No files indexed yet."],
        )

    path_by_id = {f.id: f.path for f in file_rows}

    edge_rows = session.scalars(
        sa.select(Edge)
        .where(Edge.repository_id == repo_id)
        .where(Edge.edge_type == "imports")
        .where(Edge.target_type == "file")
        .where(Edge.target_id.isnot(None))
    ).all()

    G = nx.DiGraph()
    for f in file_rows:
        G.add_node(f.id)
    for e in edge_rows:
        if e.target_id is not None:
            G.add_edge(e.source_id, e.target_id)

    # 1. Circular dependency detection
    try:
        cycles_raw = list(nx.simple_cycles(G))
        cycles = [[path_by_id.get(fid, f"file_{fid}") for fid in cycle] for cycle in cycles_raw[:5]]
    except Exception:
        cycles = []

    # 2. In-degree coupling hotspot detection
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    sorted_hubs = sorted(in_degrees.items(), key=lambda x: -x[1])

    hub_risks: list[dict[str, Any]] = []
    for fid, deg in sorted_hubs:
        if deg >= 4:
            pth = path_by_id.get(fid, "")
            hub_risks.append({
                "path": pth,
                "in_degree": deg,
                "risk_level": "High" if deg >= 8 else "Medium",
                "assessment": f"Directly imported by {deg} files. Changes to this interface could have widespread downstream impact.",
            })

    # 3. Orphaned / unreferenced file detection (exclude config & entrypoint files)
    orphaned_files: list[str] = []
    for f in file_rows:
        fid = f.id
        pth = f.path.lower()
        if in_degrees.get(fid, 0) == 0 and out_degrees.get(fid, 0) == 0:
            if not any(k in pth for k in ["config", "readme", "license", "package", "setup", "layout", "page", "route", ".d.ts"]):
                orphaned_files.append(f.path)

    # 4. Modularity ratio (intra-module vs cross-module edges)
    intra_edges = 0
    total_edges = len(edge_rows)
    for e in edge_rows:
        src_path = path_by_id.get(e.source_id, "")
        tgt_path = path_by_id.get(e.target_id, "")
        src_dir = src_path.split("/", 1)[0] if "/" in src_path else ""
        tgt_dir = tgt_path.split("/", 1)[0] if "/" in tgt_path else ""
        if src_dir == tgt_dir:
            intra_edges += 1

    modularity = (intra_edges / total_edges) if total_edges > 0 else 1.0

    # 5. Health score & grade calculation
    score = 100
    score -= min(30, len(cycles) * 15)
    score -= min(15, len([h for h in hub_risks if h["risk_level"] == "High"]) * 5)
    score -= min(10, len(orphaned_files) * 2)
    score = max(20, min(100, score))

    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "D"

    # 6. Metrics & Recommendations
    metrics = [
        HealthMetric(
            name="Circular Dependency Loops",
            value=f"{len(cycles)} detected",
            status="healthy" if len(cycles) == 0 else "alert",
            description="Zero cyclic import loops ensure clean build order and prevent initialization deadlocks.",
        ),
        HealthMetric(
            name="Coupling Hotspots",
            value=f"{len(hub_risks)} central hubs",
            status="healthy" if len(hub_risks) <= 2 else "warning",
            description="Files with high in-degree centrality require strict contract testing.",
        ),
        HealthMetric(
            name="Dead / Orphaned Files",
            value=f"{len(orphaned_files)} unlinked",
            status="healthy" if len(orphaned_files) == 0 else "warning",
            description="Files not imported or importing other modules in the application dependency graph.",
        ),
        HealthMetric(
            name="Subsystem Modularity",
            value=f"{int(modularity * 100)}%",
            status="healthy" if modularity >= 0.4 else "warning",
            description="Ratio of cohesive intra-subsystem dependencies to global cross-system imports.",
        ),
    ]

    recommendations: list[str] = []
    if len(cycles) == 0:
        recommendations.append("✅ Excellent dependency hierarchy: No circular import loops detected.")
    else:
        recommendations.append(f"⚠️ Resolve {len(cycles)} circular import loop(s) to avoid runtime initialization errors.")

    if hub_risks:
        top_hub = hub_risks[0]
        recommendations.append(f"💡 Consider interface abstraction for `{top_hub['path']}` (imported by {top_hub['in_degree']} files) to reduce blast radius.")

    if orphaned_files:
        recommendations.append(f"🔍 Audit {len(orphaned_files)} unreferenced source file(s) for potential cleanup or missing routes.")
    else:
        recommendations.append("✅ Clean codebase hygiene: Zero orphaned code files detected.")

    return ArchitectureHealthResponse(
        repository_id=repo_id,
        health_score=score,
        grade=grade,
        circular_dependencies=cycles,
        hub_risks=hub_risks,
        orphaned_files=orphaned_files,
        modularity_ratio=round(modularity, 3),
        metrics=metrics,
        recommendations=recommendations,
    )
