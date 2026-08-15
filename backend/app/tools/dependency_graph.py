"""Dependency Graph Query tool for the RepoLens agent layer (SDD §10).

The first of the two graph-leaning context tools the §9.3 Context Agent
dispatches. Given a file (or a module name resolvable to one) and a depth,
return its **dependency neighborhood** over the file-level import graph the
import-index stage built (:mod:`app.indexing.import_graph`): the files it
imports (``neighbors_out``) and the files that import it
(``neighbors_in``), up to ``depth`` hops via BFS.

This is SDD §10 "Dependency Graph Builder" — "NetworkX over ``edges(imports)``,
file or module, ``{node, neighbors_in, neighbors_out, depth}``". The graph is
already materialized in Postgres (one ``edges`` row, ``edge_type="imports"``,
per file→file dependency — see :mod:`app.indexing.import_graph`); this tool is
a pure read that loads that slice and hands it to NetworkX for the
neighborhood walk. Building the graph in-network at query time (rather than
persisting a NetworkX object) is the SDD §11 "caching" trade-off's inverse:
the import graph is small (flask = 187 internal edges) and rebuilt per query in
milliseconds, and persisting a pickled graph object would fight the "re-index =
replace rows in a transaction" contract (SDD §11).

Design split (the project's pure-core / live-wrapper convention, mirrored from
:mod:`app.tools.text_search` and the indexing stages' pure / persist):

* :func:`build_dependency_subgraph` — **pure**: takes edges as plain ``(src,
  dst)`` int pairs + a ``path_by_id`` map + a root node id + depth, runs an
  undirected-but-direction-aware BFS over NetworkX, and returns a
  :class:`DependencyGraphResult`. No DB, no session. This is the unit-testable
  core (the tests pin it against an in-memory synthetic graph with no Postgres).
* :func:`query_dependency_graph` — **live**: reads the repo's internal import
  edges (``edge_type="imports"``, ``source_type="file"``, ``target_type=
  "file"`` — internal only; external edges have ``target_id IS NULL`` by the
  import stage's contract, so they're excluded by the same ``target_id``-join
  posture the reference index uses), resolves ``target`` to a ``file_id`` via
  :func:`app.tools._path_resolve.resolve_file_id`, and delegates the BFS to the
  pure core. Injected ``Session``, owns no transaction.

Why directed forward + backward closures: the import edge is directed
(``A imports B``), so a directionally-honest query splits into two questions —
what does ``A`` *pull in* (forward / out) and what *pulls in* ``A`` (backward /
in). Loading the edges as a directed NetworkX :class:`~networkx.DiGraph` and
running a shortest-path-length BFS outbound from the root once over the graph
and once over its reverse answers both with one loaded graph. ``depth`` bounds
*each* closure (out up to ``depth`` hops, in up to ``depth`` hops) rather than an
undirected neighborhood: at ``depth=1`` ``neighbors_out`` is the files ``A``
imports and ``neighbors_in`` the files that import ``A`` (the direct read); at
``depth=2`` each list grows to its 2-hop transitivity ("what does ``A``'s
dependency tree look like two levels down" / "what transitively depends on
``A``"). The two closures are independent — a node ``A`` that imports nothing
but is widely depended on reports ``neighbors_out=[]`` with a populated
``neighbors_in``, and vice versa — which is the honest split (a "depends on"
question and a "depended on by" question are different questions, bounded by the
same ``depth`` knob).

Edges with ``target_type="external"`` (``target_id`` NULL — third-party /
stdlib / unresolvable) are *deliberately* dropped at load: they name no node in
the file graph, so including them would dead-end the BFS and pollute the
neighborhood with ``None`` placeholders. The architecture tool
(:mod:`app.tools.architecture`) makes the same drop for the same reason; the
choice is documented here rather than shared because each tool's load query is
small and the read is the live boundary (mirroring the duplicated
:func:`_ilike_contains` posture — a copy per self-contained module beats a
shared load that couples two independent tools' SQL to one owner).
"""
import os
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Edge, File, Repository
from app.tools._path_resolve import resolve_file_id

# Default neighborhood depth. SDD §10 Dependency Graph Builder names the shape
# (``{node, neighbors_in, neighbors_out, depth}``) and lists "depth" as a field,
# not a fixed value; 1 is the natural MVP default ("what does this file directly
# depend on, and what directly depends on it") and matches the typical
# "what's related to this file" intent. A higher depth is the caller's call
# (the Planner may emit it once it knows the context tools), bounded by the
# graph size; nothing here enforces a ceiling, but the result is monotone in
# depth and the live graph is small, so a misjudged large depth is a slow answer,
# not a crash.
DEFAULT_DEPTH: int = 1


# ─── result types ────────────────────────────────────────────────────────────


@dataclass
class GraphNode:
    """One node in the dependency neighborhood, as the Synthesizer will cite it.

    ``file_id`` is the node's key (ints are JSON-serializable); ``path`` is the
    POSIX-rel ``files.path`` for citation (agents cite paths, not surrogate ids
    — the same field-naming posture as :class:`app.tools.symbol_search.SymbolResult`
    whose ``file`` is ``files.path``). ``path`` is ``None`` only if a node id
    surfaced in the edges slice but no row remains in ``files`` (a deleted file
    whose edges weren't swept — the reference-index stage notes the same stale
    edge possibility); the caller treats ``None``-path nodes as best-effort.
    """

    file_id: int
    path: str | None


@dataclass
class DependencyGraphResult:
    """The SDD §10 Dependency Graph output shape, populated by BFS."""

    node_id: int | None
    node_path: str | None
    depth: int
    neighbors_in: list[GraphNode] = field(default_factory=list)
    neighbors_out: list[GraphNode] = field(default_factory=list)
    target_file_snippet: str | None = None


# ─── pure core (no DB) ───────────────────────────────────────────────────────


def build_dependency_subgraph(
    edges: Iterable[tuple[int, int]],
    path_by_id: dict[int, str],
    node_id: int,
    depth: int,
) -> DependencyGraphResult:
    """Directed forward/backward closures up to ``depth`` hops around ``node_id``
    over the file→file import graph given by ``edges`` (``(source_file_id,
    target_file_id)`` pairs; forward = "imports").

    *Pure*: no DB, no session. Builds a NetworkX :class:`~networkx.DiGraph` from
    ``edges``, then runs two shortest-path-length BFS walks each bounded by
    ``depth`` -- one outbound from ``node_id`` over the graph
    (``neighbors_out`` = files reachable by following import edges up to ``depth``
    hops, i.e. the forward transitive dependency tree) and one over the *reversed*
    graph (``neighbors_in`` = files that reach ``node_id`` by following import
    edges up to ``depth`` hops, i.e. the reverse transitive dependent set).

    ``depth <= 0`` yields only the node itself (both neighbor lists empty -- depth
    0 is "the node, no neighborhood"). ``depth == 1`` is the direct read
    (immediate imports / immediate dependents); ``depth >= 2`` grows each list to
    its transitive closure at that hop budget. ``neighbors_out`` and
    ``neighbors_in`` are independent -- a node that imports nothing but is widely
    depended on reports an empty ``neighbors_out`` and a populated ``neighbors_in``
    (and vice versa) -- which is the honest split. Nothing here blows up on a
    large ``depth``; a deeper walk returns a bigger (still graph-bounded) set.

    ``node_id`` not present in ``edges`` (an isolated file with no import
    relationships) returns a result with ``node_path`` populated from
    ``path_by_id`` if known and empty neighbor lists -- the file exists, it just
    has no graph neighborhood; the caller's "the file is a leaf, nothing depends
    on it and it depends on nothing" reading is the honest, clearly-empty
    answer. ``node_id`` absent from ``path_by_id`` too yields ``node_path=None``
    (a dangling edge from a swept file). A node present in ``edges`` but absent
    from ``path_by_id`` surfaces on the neighbor lists with ``path=None`` -- a
    stale edge, mirrored from the reference-index stage's documented possibility.
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)

    out_ids: set[int] = set()
    in_ids: set[int] = set()
    if depth > 0 and node_id in g:
        # Forward closure: files reachable from node_id by following import
        # edges, up to `depth` hops. single_source_shortest_path_length returns
        # {node: distance}; keep nodes at distance 1..depth (exclude node_id @ 0).
        fwd = nx.single_source_shortest_path_length(g, node_id, cutoff=depth)
        out_ids = {n for n, dist in fwd.items() if 0 < dist <= depth}
        # Backward closure: same walk over the reversed graph = who reaches
        # node_id by following import edges forward (i.e. node_id's dependents).
        rev = nx.single_source_shortest_path_length(g.reverse(), node_id, cutoff=depth)
        in_ids = {n for n, dist in rev.items() if 0 < dist <= depth}
    # node_id not in g (no edges touched it): both sets stay empty, node still
    # reported via node_path from path_by_id (the isolated-leaf case above).

    def _node(fid: int) -> GraphNode:
        return GraphNode(file_id=fid, path=path_by_id.get(fid))

    return DependencyGraphResult(
        node_id=node_id,
        node_path=path_by_id.get(node_id),
        depth=depth,
        # Deterministic order (sorted by file_id) for stable tests + display.
        neighbors_in=[_node(n) for n in sorted(in_ids)],
        neighbors_out=[_node(n) for n in sorted(out_ids)],
    )


# ─── live wrapper (injected session) ─────────────────────────────────────────


def _load_internal_import_edges(
    repository_id: int,
    session: Session,
) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """Load the repo's internal file→file import edges + a ``file_id -> path``
    map, for the pure-core BFS.

    *Internal only*: ``edges`` filtered to ``edge_type="imports"``,
    ``source_type="file"``, ``target_type="file"`` (``target_id`` is set). The
    import stage writes external deps as ``target_type="external"`` with
    ``target_id`` NULL (:mod:`app.indexing.import_graph`); those name no node in
    the file graph and are dropped — see the module docstring.

    The ``file_id -> path`` map is built from *all* the repo's ``files`` rows
    (not just edge endpoints) so a resolved root node whose edges were swept, or
    an isolated leaf, still gets its ``path`` populated. A row missing from the
    join surfaces as ``path=None`` on the corresponding :class:`GraphNode`
    rather than dropping the node (the reference-index stage's documented stale
    edge possibility, mirrored).
    """
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


def query_dependency_graph(
    repository_id: int,
    target: str,
    session: Session,
    *,
    depth: int = DEFAULT_DEPTH,
) -> DependencyGraphResult:
    """Return the dependency neighborhood around ``target`` (a file path or a
    resolvable module name) for ``repository_id``, up to ``depth`` import hops
    (SDD §10 Dependency Graph Builder).

    *Live read*: loads the repo's internal import edges + the file-id→path map
    via :func:`_load_internal_import_edges`, resolves ``target`` to a ``file_id``
    via :func:`app.tools._path_resolve.resolve_file_id`, and hands the slice to
    :func:`build_dependency_subgraph`. Injected ``Session``, owns no transaction.

    An unresolvable ``target`` (the resolver returns ``None``) yields an empty
    :class:`DependencyGraphResult` (``node_id=None``, ``node_path=None``, empty
    neighbor lists) — the SDD §10 contract that an empty result is a valid
    answer, and the Context Agent's "don't guess" posture for a Planner-supplied
    identifier that maps to no indexed file. ``depth`` is coerced to ``int`` and
    floored at 0 (a stray ``"2"`` from a weak model survives; a negative is
    clamped to 0 — the root-only case — the same leniency the search tools show
    a weak model's ``"true"`` for ``regex``).
    """
    try:
        depth_int = int(depth)
    except (TypeError, ValueError):
        depth_int = DEFAULT_DEPTH
    if depth_int < 0:
        depth_int = 0

    edges, path_by_id = _load_internal_import_edges(repository_id, session)
    node_id = resolve_file_id(repository_id, target, session)
    if node_id is None:
        return DependencyGraphResult(
            node_id=None, node_path=None, depth=depth_int
        )
    res = build_dependency_subgraph(edges, path_by_id, node_id, depth_int)
    if res.node_path:
        from app.tools._path_resolve import resolve_repo_root
        repo_root = resolve_repo_root(repository_id, session)
        if repo_root:
            full_p = os.path.join(repo_root, res.node_path)
            if not os.path.exists(full_p) and "/" in res.node_path:
                alt_p = os.path.join(repo_root, res.node_path.split("/", 1)[1])
                if os.path.exists(alt_p):
                    full_p = alt_p
            if os.path.exists(full_p) and os.path.isfile(full_p):
                try:
                    with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline() for _ in range(75)]
                    res.target_file_snippet = "".join(lines).strip()
                except Exception:
                    pass
    return res


__all__ = [
    "DEFAULT_DEPTH",
    "DependencyGraphResult",
    "GraphNode",
    "build_dependency_subgraph",
    "query_dependency_graph",
]
