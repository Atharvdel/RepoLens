"""Architecture Query tool for the RepoLens agent layer (SDD §10).

The second of the two graph-leaning context tools the §9.3 Context Agent
dispatches. SDD §10 "Architecture Query" — "NetworkX subgraph + centrality,
file/module or none (whole-repo), ``{modules: [...], key_files: [...] (by
degree centrality), edges: [...]}``". Two scopes per the spec row:

* **whole-repo** (``target`` empty): a repo-wide module map + the top files by
  degree centrality — the "what are the important files / how is this codebase
  laid out" overview.
* **file/module** (``target`` set): a scoped subgraph around that node — the
  architecture view centered on one file, complementary to the dependency-graph
  tool's tight neighborhood (architecture = the *structure* around X and the
  central files in that region; dependency_graph = the precise in/out
  neighborhood of X).

Like :mod:`app.tools.dependency_graph`, this reads the import-graph slice the
import-index stage built (:mod:`app.indexing.import_graph`; ``edges`` with
``edge_type="imports"``, ``source_type="file"``, ``target_type="file"`` —
internal only; external edges name no file node and are dropped, same rationale)
and runs NetworkX on it at query time. Degree centrality is the SDD §10
named measure; for a directed file→file import graph, the in-degree ("how many
files import this one") is the natural "key file" signal — a file many others
depend on. We report in-degree centrality (the standard NetworkX
:func:`networkx.in_degree_centrality` on the directed graph); for the undirected
whole-repo view we use total degree. Plain, documented, no bespoke weighting.

Design split (the project's pure-core / live-wrapper convention, mirrored from
:mod:`app.tools.dependency_graph` and the indexing stages):

* :func:`build_architecture_map` — **pure**: takes edges as ``(src, dst)``
  int pairs + a ``path_by_id`` map + an optional focus node id + a top-k,
  runs NetworkX (in-degree centrality) + a module grouping (:mod:`_path_resolve`'s
  package-dir derivation) + a scoped-subgraph extraction, and returns an
  :class:`ArchitectureResult`. No DB, no session. Unit-testable against an
  in-memory synthetic graph.
* :func:`query_architecture` — **live**: reads the repo's internal import
  edges + file-id→path map, resolves ``target`` (when given) via
  :func:`app.tools._path_resolve.resolve_file_id`, delegates to the pure core.
  Injected ``Session``, owns no transaction.

Why this tool and the dependency-graph tool are separate despite sharing a
load query: they answer different questions (overview-of-structure vs
precise-neighborhood-of-X), the SDD §10 lists them as two tools, and the §9.3
Context Agent dispatches either per plan step. Duplicating the ~10-line load
across two modules is the same self-containment posture as the duplicated
:func:`_ilike_contains` across the search tools — a copy per module beats a
shared live-boundary dependency between two independently-evolving tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Edge, File
from app.tools._path_resolve import (
    _enclosing_module,
    _package_dirs,
    resolve_file_id,
)

# How many "key files" to surface in a whole-repo architecture map. SDD §10
# lists "key_files (by degree centrality)" without a count; a small top-k keeps
# the agent-readable summary short and the centrality ranking meaningful (the
# tail is a long flat plateau on real repos). The caller may override it once
# the Planner knows to emit ``top_k``.
DEFAULT_TOP_K: int = 10

# Subgraph hop radius for a focused (file/module) architecture view. A tighter
# radius than the dependency-graph tool's neighborhood: the architecture view is
# "the region around X," and going beyond a couple of hops usually reaches most
# of the graph on a connected import graph (flask's import graph is small and
# well-connected), washing out the "region" structure. For whole-repo this is
# unused (everything is included).
DEFAULT_FOCUS_RADIUS: int = 2


# ─── result types ────────────────────────────────────────────────────────────


@dataclass
class KeyFile:
    """One "key file" of the architecture map, ranked by degree centrality.

    ``centrality`` is the float NetworkX returns (:func:`networkx.in_degree_centrality`
    on the directed import graph — normalized to ``[0, 1]``, the fraction of
    files that import this one). ``path`` is the POSIX-rel ``files.path`` for
    citation. JSON-serializable (plain types), same posture as
    :class:`app.tools.symbol_search.SymbolResult`.
    """

    file_id: int
    path: str | None
    centrality: float


@dataclass
class EdgeView:
    """One internal import edge in the reported subgraph, in citation form.

    ``label`` is ``A -> B`` over POSIX-rel paths (or the file_id when a path is
    missing — the stale-edge case the reference-index stage documents), so the
    Synthesizer can cite "X imports Y" in prose without re-deriving the
    direction. JSON-serializable.
    """

    source: str
    target: str


@dataclass
class ModuleView:
    """One top-level module/package in the repo's module map.

    ``name`` is the package dir (e.g. ``"src/flask"`` or ``""`` for a bare
    script dir); ``file_count`` is how many indexed files live under it. The
    whole-repo map lists modules with at least one file so the Synthesizer can
    say "the codebase is organized into N modules, the largest being …".
    """

    name: str
    file_count: int


@dataclass
class ArchitectureResult:
    """The SDD §10 Architecture Query output shape.

    ``scope`` is one of ``"whole"`` (no focus; full module map + repo-wide
    centrality + all internal edges) or ``"file"`` (focused on the resolved
    ``target``; locality-scoped modules/key_files/edges). JSON-serializable:
    every field is a plain type, so ``json.dumps(dataclasses.asdict(result))``
    lands cleanly in a ``tool_trace`` row (SDD §11) or the Synthesizer's
    context.

    ``focus_path`` is ``None`` for scope ``whole`` and when a ``target`` was
    given but unresolvable (the resolver returned ``None`` -> the result is a
    clear-empty "no architecture region for that file", not a crash — the same
    contract the dependency-graph tool holds for an unresolvable node).
    """

    scope: str
    focus_path: str | None
    modules: list[ModuleView] = field(default_factory=list)
    key_files: list[KeyFile] = field(default_factory=list)
    edges: list[EdgeView] = field(default_factory=list)
    overview_doc: str | None = None
    key_file_snippets: list[dict[str, Any]] = field(default_factory=list)


# ─── pure core (no DB) ───────────────────────────────────────────────────────


def _centrality_ranked(
    g: nx.DiGraph,
    subgraph_nodes: set[int],
    path_by_id: dict[int, str],
    top_k: int,
) -> list[KeyFile]:
    """The top ``top_k`` nodes in ``subgraph_nodes`` by in-degree centrality on
    ``g`` (the directed import graph), as :class:`KeyFile`.

    Uses :func:`networkx.in_degree_centrality` (normalized to ``[0, 1]`` =
    fraction of other nodes that import this one), the natural "key file" signal
    on an import graph ("how many depend on it"). For a node with no in-edges in
    the subgraph, centrality is ``0.0`` — it still may rank if the whole subgraph
    is sparse, which is honest: a leaf-only repo has genuinely low key files.

    Ties are broken by ``path`` (then ``file_id``) for stable, deterministic
    output (stable tests + stable agent display), the same posture as the
    search tools' ``order_by``.
    """
    if not subgraph_nodes:
        return []
    # in_degree_centrality over the whole graph, then filter to the subgraph
    # node set — cheaper on a small graph than subgraph-then-recompute, and
    # keeps the normalization denominator stable (the whole repo's node count),
    # so a focused view's centrality is comparable to the whole-repo view's.
    centrality = nx.in_degree_centrality(g)
    items = [
        (nid, centrality.get(nid, 0.0), path_by_id.get(nid))
        for nid in subgraph_nodes
    ]
    # Rank: centrality desc, then path asc (None sorts last), then file_id asc.
    items.sort(key=lambda t: (-t[1], t[2] is None, t[2] or "", t[0]))
    top = items[: max(0, top_k)]
    return [
        KeyFile(file_id=nid, path=path, centrality=cent)
        for nid, cent, path in top
    ]


def _module_map(
    node_ids: set[int],
    path_by_id: dict[int, str],
    package_dirs: set[str],
) -> list[ModuleView]:
    """Group the given ``node_ids`` by their nearest enclosing package
    directory (the module map), returning :class:`ModuleView` per package —
    the per-scope "modules" list. Files under no package (bare scripts) group
    under ``""``.

    ``package_dirs`` is **repo-wide** (every ``__init__.py`` marker across all
    known files), NOT derived from ``node_ids`` here. A package's marker
    ``__init__.py`` can sit outside the focus radius — an island marker that the
    import graph's edges never touched, or a central package whose ``__init__.py``
    is common to the rest. Deriving ``package_dirs`` from only the subgraph's
    paths would drop that package boundary and collapse every focused file to
    ``""``. Using the repo-wide set keeps "module" meaning the same here as in
    the import graph's resolution (the import resolver builds its package set
    from the entire repo's files, not the slice a single query happens to touch)
    — the invariant the old per-subgraph derivation silently broke for the
    focused scope.
    """
    counts: dict[str, int] = {}
    for n in node_ids:
        p = path_by_id.get(n)
        if p is None:
            continue
        mod = _enclosing_module(p, package_dirs) if package_dirs else ""
        counts[mod] = counts.get(mod, 0) + 1
    # Sort by file_count desc, then name asc, for a stable, useful ordering.
    return [
        ModuleView(name=mod, file_count=cnt)
        for mod, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _edge_views(
    edges: Iterable[tuple[int, int]],
    path_by_id: dict[int, str],
) -> list[EdgeView]:
    """Render ``edges`` as citation-form :class:`EdgeView` (``A -> B`` over
    POSIX-rel paths). A source/target whose path is missing (swept file, stale
    edge) falls back to its ``file_id`` as a string so no edge is silently
    dropped — the reference-index stage documents the same stale-edge
    possibility."""
    out: list[EdgeView] = []
    for src, dst in edges:
        sp = path_by_id.get(src)
        dp = path_by_id.get(dst)
        out.append(
            EdgeView(
                source=sp if sp is not None else f"#{src}",
                target=dp if dp is not None else f"#{dst}",
            )
        )
    return out


def build_architecture_map(
    edges: Iterable[tuple[int, int]],
    path_by_id: dict[int, str],
    *,
    focus_node_id: int | None = None,
    radius: int = DEFAULT_FOCUS_RADIUS,
    top_k: int = DEFAULT_TOP_K,
) -> ArchitectureResult:
    """Build an SDD §10 Architecture Query result over the directed import graph
    given by ``edges`` (``(source_file_id, target_file_id)`` pairs).

    *Pure*: no DB, no session. Builds a NetworkX :class:`~networkx.DiGraph`
    from ``edges``, computes in-degree centrality, groups files under their
    enclosing package dirs (the module map), and — when ``focus_node_id`` is
    given — restricts the reported modules / key_files / edges to the
    ``radius``-hop induced subgraph around the focus.

    ``focus_node_id=None`` -> ``scope="whole"``: the whole-repo map. A non-``None``
    focus that is **not** present in the graph (an isolated leaf the import
    edges never touched, or a node the resolver handed back but the edges slice
    doesn't mention) still yields ``scope="file"`` with the focus's path set and
    the modules/key_files/edges computed over a one-node subgraph (the focus
    itself) — the honest "this file is an island in the import graph" reading,
    not an empty crash. An unresolvable focus (``None`` from the caller, handled
    in the live wrapper) never reaches here; the live wrapper returns the
    clear-empty result directly.

    Edges are de-duplicated to an induced-subgraph edge set (no parallel edges,
    which the import dedup already enforced at index time — but NetworkX's
    DiGraph would coalesce them anyway) and rendered in determinant order
    (sorted by (source path, target path) — the same stability posture).
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)

    # Package boundaries are a property of the WHOLE repo's file layout, not the
    # radius-bounded subgraph: an __init__.py that marks a package can sit outside
    # the focus radius (e.g. pkg/__init__.py two hops from a focused pkg/utils.py),
    # and deriving package_dirs from only the subgraph's paths would drop that
    # package boundary — collapsing every focused file's module to "". Build it
    # once from all known files, mirroring the import graph's package detection.
    package_dirs = _package_dirs(list(path_by_id.values()))

    if focus_node_id is None:
        scope = "whole"
        focus_path = None
        # Every internal file node — from both edges and the path map — so an
        # isolated leaf with no edges still appears in the module map + centrality.
        node_ids: set[int] = set(g.nodes()) | set(path_by_id.keys())
        sub_edges = list(g.edges())
    else:
        scope = "file"
        focus_path = path_by_id.get(focus_node_id)
        node_ids = {focus_node_id}
        if focus_node_id in g and radius > 0:
            undirected = g.to_undirected()
            try:
                shells = list(nx.bfs_layers(undirected, focus_node_id))
            except nx.NetworkXError:
                shells = [[focus_node_id]]
            for shell in shells[: radius + 1]:
                node_ids.update(shell)
        # Restrict reported edges to the induced subgraph (both endpoints in node_ids).
        sub_edges = [
            (s, t) for s, t in g.edges() if s in node_ids and t in node_ids
        ]

    return ArchitectureResult(
        scope=scope,
        focus_path=focus_path,
        modules=_module_map(node_ids, path_by_id, package_dirs),
        key_files=_centrality_ranked(g, node_ids, path_by_id, top_k),
        edges=_edge_views(sorted(sub_edges), path_by_id),
    )


# ─── live wrapper (injected session) ─────────────────────────────────────────


def _load_internal_import_edges(
    repository_id: int,
    session: Session,
) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """Load the repo's internal file→file import edges + a ``file_id -> path``
    map — the twin of :func:`app.tools.dependency_graph._load_internal_import_edges`.
    Kept a copy (not a shared import) for module self-containment; the load is
    ~10 lines and duplicating it keeps the two tools' live boundaries
    independent, exactly as the search tools duplicate :func:`_ilike_contains`."""
    edge_rows = session.execute(
        sa.select(Edge.source_id, Edge.target_id)
        .where(
            Edge.repository_id == repository_id,
            Edge.edge_type == "imports",
            Edge.source_type == "file",
            Edge.target_type == "file",
            Edge.target_id.is_not(None),
        )
    ).all()
    edges = [(int(src), int(dst)) for src, dst in edge_rows]

    path_rows = session.execute(
        sa.select(File.id, File.path).where(File.repository_id == repository_id)
    ).all()
    path_by_id = {int(fid): pth for fid, pth in path_rows}
    return edges, path_by_id


def query_architecture(
    repository_id: int,
    target: str | None,
    session: Session,
    *,
    top_k: int = DEFAULT_TOP_K,
    radius: int = DEFAULT_FOCUS_RADIUS,
) -> ArchitectureResult:
    """Return the SDD §10 Architecture Query result for ``repository_id``.

    *Live read*: loads the repo's internal import edges + file-id→path map,
    resolves ``target`` (when non-empty) to a ``file_id`` via
    :func:`app.tools._path_resolve.resolve_file_id`, and delegates to
    :func:`build_architecture_map`. Injected ``Session``, owns no transaction.

    ``target`` empty/``None`` -> whole-repo scope (``scope="whole"``). A given
    ``target`` that resolves to ``None`` (the identifier maps to no indexed file)
    yields a clear-empty result: ``scope="file"``, ``focus_path=None``, empty
    modules/key_files/edges — the same "don't guess, return a clear empty"
    contract :func:`app.tools.dependency_graph.query_dependency_graph` holds for
    an unresolvable node. ``top_k`` and ``radius`` are int-coerced (a weak
    model's ``"10"`` survives) and floored at sensible minimums.
    """
    edges, path_by_id = _load_internal_import_edges(repository_id, session)
    target_s = (target or "").strip()
    if not target_s:
        result = build_architecture_map(
            edges, path_by_id, top_k=_coerce_int(top_k, DEFAULT_TOP_K, 1)
        )
    else:
        focus = resolve_file_id(repository_id, target_s, session)
        if focus is None:
            return ArchitectureResult(scope="file", focus_path=None)
        result = build_architecture_map(
            edges,
            path_by_id,
            focus_node_id=focus,
            radius=_coerce_int(radius, DEFAULT_FOCUS_RADIUS, 0),
            top_k=_coerce_int(top_k, DEFAULT_TOP_K, 1),
        )

    # 1. Fetch README / documentation for overview context
    from pathlib import Path
    from app.models.document import Document
    from app.models.repository import Repository

    doc_row = session.execute(
        sa.select(Document.path, Document.content)
        .where(Document.repository_id == repository_id)
        .order_by(
            sa.case(
                (Document.path.ilike("%readme%"), 1),
                (Document.path.ilike("%overview%"), 2),
                else_=3,
            )
        )
        .limit(1)
    ).first()
    if doc_row:
        doc_path, doc_content = doc_row
        result.overview_doc = f"[{doc_path}]:\n{doc_content[:2000]}"

    # 2. Fetch code snippets for top central key files
    from app.tools._path_resolve import resolve_repo_root
    repo_root = resolve_repo_root(repository_id, session)

    if repo_root and repo_root.is_dir():
        snippets = []
        paths_to_read = []
        if result.focus_path:
            paths_to_read.append(result.focus_path)
        for kf in result.key_files:
            if kf.path and kf.path not in paths_to_read:
                paths_to_read.append(kf.path)

        for p in paths_to_read[:5]:
            full_path = repo_root / p
            if not full_path.is_file() and "/" in p:
                alt = repo_root / p.split("/", 1)[1]
                if alt.is_file():
                    full_path = alt
            if full_path.is_file():
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = [f.readline() for _ in range(75)]
                        snippet = "".join(lines).strip()
                        if snippet:
                            snippets.append({"path": p, "snippet": snippet})
                except Exception:
                    pass
        result.key_file_snippets = snippets

    return result


def _coerce_int(value, default: int, floor: int) -> int:
    """Coerce a planner-supplied numeric-ish value to an int, falling back to
    ``default``; clamp to ``>= floor``. The same leniency the search tools show
    a weak model's ``"true"`` for ``regex``: a stray ``"10"`` survives as 10, a
    garbage value falls back, a negative is clamped (``top_k=0`` would strip the
    centrality list to nothing, so it's floored at 1)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(floor, v)


__all__ = [
    "ArchitectureResult",
    "DEFAULT_FOCUS_RADIUS",
    "DEFAULT_TOP_K",
    "EdgeView",
    "KeyFile",
    "ModuleView",
    "build_architecture_map",
    "query_architecture",
]
