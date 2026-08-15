"""Graph and architecture visualizer endpoints (SDD §12).

Routes:
- GET /repositories/{id}/graph           Materializes node/edge graph for Cytoscape / React Flow
- GET /repositories/{id}/architecture    Returns key files by centrality and package module breakdown
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional

import networkx as nx
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Edge, File, Repository
from app.tools.architecture import query_architecture
from app.tools.dependency_graph import query_dependency_graph

router = APIRouter(prefix="/repositories/{repo_id}", tags=["graph"])


# ─── Pydantic schemas ────────────────────────────────────────────────────────


class GraphNode(BaseModel):
    id: str
    file_id: int
    path: str
    label: str
    language: str
    loc: int
    in_degree: int = 0
    out_degree: int = 0
    module: str = ""


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "imports"


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    scope: str
    target: Optional[str] = None


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.get("/graph", response_model=GraphResponse)
def get_repository_graph(
    repo_id: int,
    scope: str = Query("whole", pattern="^(whole|file|module)$", description="Scope of graph view"),
    target: Optional[str] = Query(None, description="Optional target file path or dotted module"),
    depth: int = Query(1, ge=1, le=5, description="Hop depth for focused neighborhood"),
    session: Session = Depends(get_session),
):
    """Retrieve node/edge graph data for interactive dependency visualization."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    if scope == "file" and target:
        dep_result = query_dependency_graph(
            target=target,
            repository_id=repo_id,
            session=session,
            depth=depth,
        )
        if dep_result.node is None:
            return GraphResponse(nodes=[], edges=[], scope="file", target=target)

        # Collect all file paths in in/out closures
        all_paths = {dep_result.node.path}
        for n in dep_result.neighbors_in:
            all_paths.add(n.path)
        for n in dep_result.neighbors_out:
            all_paths.add(n.path)

        file_rows = session.scalars(
            sa.select(File)
            .where(File.repository_id == repo_id)
            .where(File.path.in_(list(all_paths)))
        ).all()
        path_to_file = {f.path: f for f in file_rows}

        nodes: list[GraphNode] = []
        for pth in sorted(all_paths):
            f = path_to_file.get(pth)
            if f:
                mod = pth.rsplit("/", 1)[0] if "/" in pth else ""
                nodes.append(
                    GraphNode(
                        id=f"file_{f.id}",
                        file_id=f.id,
                        path=f.path,
                        label=os.path.basename(f.path),
                        language=f.language or "unknown",
                        loc=f.loc or 0,
                        module=mod,
                    )
                )

        # Build edges among selected nodes
        fids = [f.id for f in file_rows]
        edge_rows = session.scalars(
            sa.select(Edge)
            .where(Edge.repository_id == repo_id)
            .where(Edge.edge_type == "imports")
            .where(Edge.source_id.in_(fids))
            .where(Edge.target_id.in_(fids))
        ).all()

        edges: list[GraphEdge] = [
            GraphEdge(
                id=f"e_{e.id}",
                source=f"file_{e.source_id}",
                target=f"file_{e.target_id}",
                type=e.edge_type,
            )
            for e in edge_rows
            if e.target_id is not None
        ]

        return GraphResponse(nodes=nodes, edges=edges, scope="file", target=target)

    # Whole-repo graph view
    file_rows = session.scalars(
        sa.select(File).where(File.repository_id == repo_id)
    ).all()
    if not file_rows:
        return GraphResponse(nodes=[], edges=[], scope="whole", target=None)

    edge_rows = session.scalars(
        sa.select(Edge)
        .where(Edge.repository_id == repo_id)
        .where(Edge.edge_type == "imports")
        .where(Edge.target_type == "file")
        .where(Edge.target_id.isnot(None))
    ).all()

    # Build DiGraph for degrees
    G = nx.DiGraph()
    for f in file_rows:
        G.add_node(f.id)
    for e in edge_rows:
        if e.target_id is not None:
            G.add_edge(e.source_id, e.target_id)

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    nodes = []
    for f in file_rows:
        mod = f.path.rsplit("/", 1)[0] if "/" in f.path else ""
        nodes.append(
            GraphNode(
                id=f"file_{f.id}",
                file_id=f.id,
                path=f.path,
                label=os.path.basename(f.path),
                language=f.language or "unknown",
                loc=f.loc or 0,
                in_degree=in_degrees.get(f.id, 0),
                out_degree=out_degrees.get(f.id, 0),
                module=mod,
            )
        )

    edges = [
        GraphEdge(
            id=f"e_{e.id}",
            source=f"file_{e.source_id}",
            target=f"file_{e.target_id}",
            type=e.edge_type,
        )
        for e in edge_rows
        if e.target_id is not None
    ]

    return GraphResponse(nodes=nodes, edges=edges, scope="whole", target=None)


@router.get("/architecture")
def get_architecture_overview(
    repo_id: int,
    target: Optional[str] = Query(None, description="Optional target file path or dotted module"),
    top_k: int = Query(10, ge=1, le=50, description="Number of key files to surface"),
    radius: int = Query(2, ge=1, le=5, description="Radius for focused architecture"),
    session: Session = Depends(get_session),
):
    """Retrieve high-level module map, in-degree centrality key files, and architecture overview."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    result = query_architecture(
        target=target,
        repository_id=repo_id,
        session=session,
        top_k=top_k,
        radius=radius,
    )
    return dataclasses.asdict(result)
