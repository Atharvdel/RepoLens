"""Tests for the four Context Agent tools (SDD §10).

Three layers, mirroring the project's pure-core / live-wrapper convention used
by the sibling tool + agent tests:

* ``test_dotted_to_path_*`` / ``test_build_dependency_subgraph_*`` /
  ``test_build_architecture_map_*`` / ``test_build_file_history_*`` /
  ``test_build_github_metadata_*`` -- *pure, no DB, no networkx-on-Postgres*.
  Exercise the pure cores (:func:`app.tools._path_resolve._dotted_to_path`,
  :func:`app.tools.dependency_graph.build_dependency_subgraph`,
  :func:`app.tools.architecture.build_architecture_map`,
  :func:`app.tools.file_history.build_file_history`,
  :func:`app.tools.github_metadata.build_github_metadata`) against crafted
  in-memory inputs (synthetic graphs / commit tuples / issue rows), pinning the
  forward+backward closure, the in-degree centrality ranking, the package-dir
  module grouping, the contributor / recent-commit aggregation, and the
  issue/PR URL split + linked-file filter. These run anywhere -- no Postgres.

* ``test_resolve_file_id_*`` / ``test_query_file_history_*`` /
  ``test_query_github_metadata_*`` -- *synthetic-Postgres round-trip*. Insert a
  throwaway repo + files (+ commits/file_commits / issues for the two metadata
  tools), exercise the live wrappers' resolution + read + aggregation, clean up.
  Skip cleanly when Postgres is not reachable (per the per-file readiness-probe
  precedent in :mod:`tests.test_search_agent`).

* ``test_query_*_flask_*`` -- *live against the indexed flask repo*. The graph
  tools (architecture / dependency_graph) assert real structure from flask's
  import-graph slice; the two metadata tools (file_history / github_metadata)
  assert their honest "clear empty until §7 step 7 / step 8 lands" posture
  (``last_modified`` populated by the walker, contributor/commit lists empty;
  issues/PRs empty). Skip cleanly when flask is not indexed.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_context_tools.py -v
"""
from __future__ import annotations

from datetime import datetime

import pytest
import sqlalchemy as sa

from app.db import SessionLocal
from app.models import Commit, Edge, File, Issue, Repository, file_commits
from app.tools._path_resolve import _dotted_to_path, resolve_file_id
from app.tools.architecture import (
    ArchitectureResult,
    EdgeView,
    KeyFile,
    ModuleView,
    build_architecture_map,
    query_architecture,
)
from app.tools.dependency_graph import (
    DependencyGraphResult,
    GraphNode,
    build_dependency_subgraph,
    query_dependency_graph,
)
from app.tools.file_history import (
    ContributorView,
    FileHistoryResult,
    build_file_history,
    query_file_history,
)
from app.tools.github_metadata import (
    GitHubMetadataResult,
    IssueView,
    build_github_metadata,
    query_github_metadata,
)

# ─── readiness probes (per-file, defensive — mirrors test_search_agent) ─────
# conftest couples no DB state, so each test file owns its probes. Every probe
# defensively swallows any DB/filesystem error to ``(False, None)`` so a fresh
# environment skips cleanly rather than erroring at collection time.


def _db_ready() -> bool:
    """True iff the Postgres the app is configured for is reachable. The
    synthetic-DB tests (resolve_file_id / query_file_history /
    query_query_github_metadata) need a live connection to insert throwaway
    rows; without it they skip rather than surface a connection error."""
    try:
        with SessionLocal() as session:
            session.execute(sa.select(sa.literal(1)))
        return True
    except Exception:
        return False


_DB_OK = _db_ready()
db_required = pytest.mark.skipif(not _DB_OK, reason="Postgres not reachable at the configured DATABASE_URL")


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)`` for the DB-backed flask tests where a target just
    needs to resolve to a real flask ``files`` row (file_history /
    github_metadata live-empty checks). Ready iff a ``flask`` repo exists with
    ``files`` rows."""
    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return False, None
            n_files = session.scalar(
                sa.select(sa.func.count())
                .select_from(File)
                .where(File.repository_id == repo.id)
            ) or 0
            return (n_files > 0, repo.id if n_files else None)
    except Exception:
        return False, None


def _flask_imports_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)`` for the graph tools' live tests. Ready iff flask
    has ``files`` rows AND at least one internal import edge
    (``edge_type="imports"``, ``source_type="file"``, ``target_type="file"``)
    -- the architecture + dependency_graph tools read that slice, so an empty
    slice is the not-ready signal rather than a vacuous-pass."""
    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return False, None
            n_files = session.scalar(
                sa.select(sa.func.count())
                .select_from(File)
                .where(File.repository_id == repo.id)
            ) or 0
            n_edges = session.scalar(
                sa.select(sa.func.count())
                .select_from(Edge)
                .where(
                    Edge.repository_id == repo.id,
                    Edge.edge_type == "imports",
                    Edge.source_type == "file",
                    Edge.target_type == "file",
                    Edge.target_id.is_not(None),
                )
            ) or 0
            ready = n_files > 0 and n_edges > 0
            return (ready, repo.id if ready else None)
    except Exception:
        return False, None


_FLASK_READY, FLASK_REPO_ID = _flask_indexed()
flask_required = pytest.mark.skipif(
    not _FLASK_READY,
    reason="flask repo not indexed in DB (run scripts/run_walker_once.py first)",
)

_FLASK_IMPORTS_READY, FLASK_IMPORTS_REPO_ID = _flask_imports_indexed()
flask_imports_required = pytest.mark.skipif(
    not _FLASK_IMPORTS_READY,
    reason="flask import graph not indexed (run scripts/index_all_flask_imports.py)",
)


# ═══════════════════════════════════════════════════════════════════════════
# pure: _dotted_to_path (the path↔module identifier adapter)
# ═══════════════════════════════════════════════════════════════════════════
# The heuristic that keeps a real path faithful (``.py``/separators = literal
# dots) while translating a bare dotted module name (``flask.app``) to path form.
# Pinned directly the way the sibling tests pin the private ``_ilike_contains``.


def test_dotted_to_path_translates_bare_module_name():
    """A bare dotted module name (no ``/``, no ``.py``) → path form, the common
    case where a plan step carries ``{"target": "flask.app"}`` and must resolve
    to ``flask/app`` (then ``flask/app.py`` by the resolver)."""
    assert _dotted_to_path("flask.app") == "flask/app"
    assert _dotted_to_path("flask.sansio.app") == "flask/sansio/app"


def test_dotted_to_path_leaves_real_path_unchanged():
    """A path already containing a separator or ending in ``.py`` is returned
    unchanged — its dots are LITERAL (``.py`` extension, dots in segment
    names), not module separators. This is the regression guard for the bug
    where ``src/flask/app.py`` was mangled to ``src/flask/app/py``."""
    assert _dotted_to_path("src/flask/app.py") == "src/flask/app.py"
    assert _dotted_to_path("flask/app.py") == "flask/app.py"
    # A path with a separator but no .py: dots in segment names stay literal.
    assert _dotted_to_path("pkg/a.b/run.py") == "pkg/a.b/run.py"
    # A bare ``__init__`` module name (no sep, no .py) → path form so the
    # resolver's package-tail stage catches it.
    assert _dotted_to_path("flask") == "flask"


def test_dotted_to_path_handles_empty_and_whitespace_and_backslashes():
    """Empty / whitespace-only → ``""`` (the resolver's "unresolvable"
    sentinel); Windows backslashes collapse to forward slashes so the same
    identifier resolves regardless of the host the plan was built on."""
    assert _dotted_to_path("") == ""
    assert _dotted_to_path("   ") == ""
    assert _dotted_to_path(None) == ""  # type: ignore[arg-type]
    assert _dotted_to_path("src\\flask\\app.py") == "src/flask/app.py"
    # backslashes in a bare module name -> forward slashes BEFORE the
    # is-it-a-path check, so ``flask\sansio`` is treated as already-path-like.
    assert _dotted_to_path("flask\\sansio.app") == "flask/sansio.app"


# ═══════════════════════════════════════════════════════════════════════════
# pure: build_dependency_subgraph (directed forward+backward closures)
# ═══════════════════════════════════════════════════════════════════════════
# Synthetic directed graph (``src imports dst``):
#     1 (app/__init__)  ->  2 (helpers) , 3 (models)
#     2 (helpers)       ->  4 (utils)
#     3 (models)        ->  4 (utils)
#     5 (tools/run)      ->  1 (app/__init__)
# i.e. edges = [(1,2),(1,3),(2,4),(3,4),(5,1)]
# 4 imports nothing but is widely depended on; 5 imports but is depended on
# by nothing. ids are JSON-safe ints; paths only matter for path-bearers.

_DEP_EDGES = [(1, 2), (1, 3), (2, 4), (3, 4), (5, 1)]
_DEP_PATHS = {
    1: "pkg/__init__.py",
    2: "pkg/helpers.py",
    3: "pkg/models.py",
    4: "pkg/utils.py",
    5: "tools/run.py",
}


def test_dependency_subgraph_depth1_direct_in_and_out():
    """At depth 1 a node reports its DIRECT imports (out) and direct dependents
    (in) independently — node 2 imports utils (out={4}), and is imported by
    app (in={1}). The two lists are separate (a "depends on" vs "depended on
    by" split), not an undirected neighborhood."""
    r = build_dependency_subgraph(_DEP_EDGES, _DEP_PATHS, node_id=2, depth=1)
    assert isinstance(r, DependencyGraphResult)
    assert r.node_id == 2
    assert r.node_path == "pkg/helpers.py"
    assert r.depth == 1
    assert [n.file_id for n in r.neighbors_out] == [4], r.neighbors_out
    assert [n.file_id for n in r.neighbors_in] == [1], r.neighbors_in


def test_dependency_subgraph_utils_imports_nothing_depended_on_by_many():
    """Node 4 (utils) imports nothing — ``neighbors_out=[]`` — but is imported
    by 2 and 3, so ``neighbors_in={2,3}``. This is the directional-independence
    case the old undirected-BFS design collapsed: a leaf-by-imports-but-root-
    by-dependency file reports an empty out and a populated in, honestly."""
    r = build_dependency_subgraph(_DEP_EDGES, _DEP_PATHS, node_id=4, depth=1)
    assert [n.file_id for n in r.neighbors_out] == [], r.neighbors_out
    assert [n.file_id for n in r.neighbors_in] == [2, 3], r.neighbors_in  # sorted by file_id


def test_dependency_subgraph_depth2_transitive_out_and_in():
    """At depth 2 the closures become transitive. Node 1's forward closure
    reaches {2,3} at hop 1 and {4} at hop 2 (via 2→4 and 3→4) → out={2,3,4}.
    Node 1's backward closure: who reaches 1? Only 5 (5→1) at hop 1; nothing
    reaches 5 → in={5}. This is the behavior that makes ``depth`` meaningful
    beyond 1 (the regression target of this rewrite)."""
    r = build_dependency_subgraph(_DEP_EDGES, _DEP_PATHS, node_id=1, depth=2)
    assert [n.file_id for n in r.neighbors_out] == [2, 3, 4], r.neighbors_out
    assert [n.file_id for n in r.neighbors_in] == [5], r.neighbors_in


def test_dependency_subgraph_depth2_backward_reaches_transitive_dependents():
    """Node 4: backward closure at depth 2 reaches {2,3} at hop 1 and {1} at
    hop 2 (1→2→4, 1→3→4) → in={1,2,3}. Forward stays empty (4 imports
    nothing)."""
    r = build_dependency_subgraph(_DEP_EDGES, _DEP_PATHS, node_id=4, depth=2)
    assert [n.file_id for n in r.neighbors_out] == [], r.neighbors_out
    assert [n.file_id for n in r.neighbors_in] == [1, 2, 3], r.neighbors_in


def test_dependency_subgraph_depth0_is_node_only_no_neighborhood():
    """depth=0 → just the node, no neighborhood (both lists empty); the node is
    still reported via node_path. The caller's "show me the file, nothing
    around it" reading."""
    r = build_dependency_subgraph(_DEP_EDGES, _DEP_PATHS, node_id=1, depth=0)
    assert r.node_path == "pkg/__init__.py"
    assert r.neighbors_in == [] and r.neighbors_out == []


def test_dependency_subgraph_isolated_node_reports_path_empty_neighbors():
    """A node present in ``path_by_id`` but with no edges (an isolated leaf)
    returns its path with empty neighbor lists — the "this file exists but has
    no import relationships" honest answer, NOT a crash."""
    paths = dict(_DEP_PATHS)
    paths[6] = "orphan.py"
    r = build_dependency_subgraph(_DEP_EDGES, paths, node_id=6, depth=1)
    assert r.node_path == "orphan.py"
    assert r.neighbors_in == [] and r.neighbors_out == []


def test_dependency_subgraph_node_unknown_to_paths_surfaces_none_path():
    """A node id that appears in edges but whose path is missing from
    ``path_by_id`` (a stale edge from a swept file) surfaces on neighbor lists
    with ``path=None`` rather than dropping the node — the reference-index
    stage's documented stale-edge possibility, mirrored."""
    paths = {1: "pkg/__init__.py"}  # 2/3/4/5 missing
    r = build_dependency_subgraph(_DEP_EDGES, paths, node_id=1, depth=1)
    assert r.node_path == "pkg/__init__.py"
    assert all(n.path is None for n in r.neighbors_out), r.neighbors_out
    assert [n.file_id for n in r.neighbors_out] == [2, 3]


def test_dependency_subgraph_neighbor_ordering_is_deterministic_by_file_id():
    """Neighbor lists are sorted by ``file_id`` ascending — stable for tests and
    for agent display, the same posture as the search tools' ``order_by``."""
    r = build_dependency_subgraph(_DEP_EDGES, _DEP_PATHS, node_id=4, depth=1)
    assert [n.file_id for n in r.neighbors_in] == sorted([2, 3])
    # GraphNode carries both file_id and path.
    assert all(isinstance(n, GraphNode) for n in r.neighbors_in + r.neighbors_out)


def test_dependency_subgraph_empty_graph_isolated_node():
    """An empty edge set with a node from the path map → isolated, empty lists,
    path set. No NetworkX error on a graph with no nodes."""
    r = build_dependency_subgraph([], {9: "solo.py"}, node_id=9, depth=2)
    assert r.node_path == "solo.py"
    assert r.neighbors_in == [] and r.neighbors_out == []
    assert r.depth == 2


# ═══════════════════════════════════════════════════════════════════════════
# pure: build_architecture_map (whole-repo + focused centrality/module/edges)
# ═══════════════════════════════════════════════════════════════════════════
# Same graph as the dependency tests (ids 1..5), with package-aware paths so
# the module map is non-trivial: ``pkg/__init__.py`` makes ``pkg`` a package;
# ``tools/run.py`` has no ``__init__`` so it's a bare-script under ``""``.


def test_architecture_whole_repo_scope_modules_keyfiles_edges():
    """Whole-repo (focus None) → scope "whole", focus_path None, the full module
    map + repo-wide in-degree centrality key files + all internal edges. On the
    5-node graph: ``pkg`` holds 4 files (1,2,3,4), ``""`` holds 1 (5); node 4
    (utils) has in-degree 2 → centrality 0.5 = the top key file; nodes 1,2,3
    tie at in-degree 1 (0.25), broken by path asc; node 5 at 0.0."""
    r = build_architecture_map(_DEP_EDGES, _DEP_PATHS)
    assert isinstance(r, ArchitectureResult)
    assert r.scope == "whole"
    assert r.focus_path is None

    # Module map: pkg(4) > ""(1).
    assert r.modules == [ModuleView(name="pkg", file_count=4), ModuleView(name="", file_count=1)], r.modules

    # Key files: 4 (0.5), then 1,2,3 (0.25, path-asc tie-break), then 5 (0.0).
    kf = [(k.file_id, k.path, k.centrality) for k in r.key_files]
    assert kf == [
        (4, "pkg/utils.py", 0.5),
        (1, "pkg/__init__.py", 0.25),
        (2, "pkg/helpers.py", 0.25),
        (3, "pkg/models.py", 0.25),
        (5, "tools/run.py", 0.0),
    ], kf

    # Edges: all 5 internal edges, rendered citation-form and sorted by
    # (source path, target path).
    assert r.edges == [
        EdgeView(source="pkg/__init__.py", target="pkg/helpers.py"),
        EdgeView(source="pkg/__init__.py", target="pkg/models.py"),
        EdgeView(source="pkg/helpers.py", target="pkg/utils.py"),
        EdgeView(source="pkg/models.py", target="pkg/utils.py"),
        EdgeView(source="tools/run.py", target="pkg/__init__.py"),
    ], r.edges


def test_architecture_whole_repo_includes_isolated_leaf_from_path_map():
    """An isolated file present only in ``path_by_id`` (no edges) still appears
    in the whole-repo module map at centrality 0.0 — the architecture view
    doesn't drop files with no import relationships."""
    paths = dict(_DEP_PATHS)
    paths[6] = "orphan.py"
    r = build_architecture_map(_DEP_EDGES, paths)
    # "" module now holds ids 5 and 6 → count 2; pkg still 4.
    assert r.modules == [
        ModuleView(name="pkg", file_count=4),
        ModuleView(name="", file_count=2),
    ], r.modules
    # orphan is at centrality 0.0, sorted last among the 0.0/0.25 tier by path.
    orphan = [k for k in r.key_files if k.file_id == 6]
    assert orphan and orphan[0].centrality == 0.0 and orphan[0].path == "orphan.py"


def test_architecture_focused_scope_radius_bounded_subgraph():
    """Focused on node 4 (pkg/utils) at radius 1 → scope "file",
    focus_path "pkg/utils.py", the induced 1-hop subgraph. Undirected BFS from
    4 reaches {4,2,3} (2→4, 3→4); edges restricted to that induced subgraph =
    {(2,4),(3,4)}; modules restricted to just ``pkg`` (the 3 nodes); key files
    ranked over the subgraph with whole-graph centrality (4=0.5, 2=3=0.25)."""
    r = build_architecture_map(_DEP_EDGES, _DEP_PATHS, focus_node_id=4, radius=1)
    assert r.scope == "file"
    assert r.focus_path == "pkg/utils.py"
    assert r.modules == [ModuleView(name="pkg", file_count=3)], r.modules
    kf = [(k.file_id, k.centrality) for k in r.key_files]
    assert kf == [(4, 0.5), (2, 0.25), (3, 0.25)], kf
    assert r.edges == [
        EdgeView(source="pkg/helpers.py", target="pkg/utils.py"),
        EdgeView(source="pkg/models.py", target="pkg/utils.py"),
    ], r.edges


def test_architecture_focused_isolated_node_is_one_node_subgraph():
    """A focus node with no edges (isolated) → scope "file" with the focus's
    path, a one-node subgraph: modules group it under ``""`` (no packages),
    key_files reports it at centrality 0.0, edges empty. The honest "this file
    is an island in the import graph" reading, not an empty crash."""
    r = build_architecture_map([], {6: "lone.py"}, focus_node_id=6, radius=2)
    assert r.scope == "file"
    assert r.focus_path == "lone.py"
    assert r.modules == [ModuleView(name="", file_count=1)], r.modules
    assert [(k.file_id, k.centrality) for k in r.key_files] == [(6, 0.0)]
    assert r.edges == []


def test_architecture_focused_node_unknown_to_paths_focus_path_none():
    """A focus node absent from ``path_by_id`` → focus_path None but the node is
    still reported (node_id known, path missing). Modules empty (no known path
    to group), key_files with path=None, edges empty (no paths to render)."""
    r = build_architecture_map([], {7: "x.py"}, focus_node_id=8, radius=1)
    assert r.scope == "file"
    assert r.focus_path is None
    assert r.modules == []
    assert r.key_files == [KeyFile(file_id=8, path=None, centrality=0.0)]
    assert r.edges == []


def test_architecture_stale_edge_renders_file_id_fallback():
    """An edge whose endpoint path is missing renders the ``#<id>`` fallback
    rather than silently dropping the edge — the stale-edge contract the
    reference-index stage documents, mirrored here."""
    paths = {1: "pkg/__init__.py"}  # 2 missing
    r = build_architecture_map([(1, 2)], paths)
    assert EdgeView(source="pkg/__init__.py", target="#2") in r.edges


def test_architecture_top_k_truncates_key_files():
    """``top_k`` bounds the key-file list; the highest-centrality files survive."""
    r = build_architecture_map(_DEP_EDGES, _DEP_PATHS, top_k=2)
    assert [k.file_id for k in r.key_files] == [4, 1], r.key_files  # 0.5, then first 0.25


# ═══════════════════════════════════════════════════════════════════════════
# pure: build_file_history (contributor + recent-commit aggregation)
# ═══════════════════════════════════════════════════════════════════════════


def _commit(hash_: str, msg: str, date: datetime, author: str) -> tuple[str, str, datetime, str]:
    return (hash_, msg, date, author)


def test_file_history_aggregates_contributors_and_recent_commits():
    """The pure core aggregates contributors (count per author, ranked count
    desc then author asc) and recent commits (newest-first, capped) from
    ``(hash,message,date,author)`` tuples, ISO-encoding the dates at this
    boundary. Alice authored 2 of these, Bob 1 → Alice first."""
    commits = [
        _commit("aaa1", "init", datetime(2024, 3, 1), "Alice"),
        _commit("bbb2", "fix", datetime(2024, 3, 5), "Bob"),
        _commit("ccc3", "refactor", datetime(2024, 3, 3), "Alice"),
    ]
    r = build_file_history("pkg/app.py", datetime(2024, 3, 10), commits)
    assert isinstance(r, FileHistoryResult)
    assert r.file_path == "pkg/app.py"
    assert r.last_modified == "2024-03-10T00:00:00"
    # Contributors: Alice(2) then Bob(1).
    assert r.top_contributors == [
        ContributorView(author="Alice", commits=2),
        ContributorView(author="Bob", commits=1),
    ], r.top_contributors
    # Recent commits newest-first (5,3,1), all 3 within cap of 10.
    assert [(c.hash, c.date, c.author) for c in r.recent_commits] == [
        ("bbb2", "2024-03-05T00:00:00", "Bob"),
        ("ccc3", "2024-03-03T00:00:00", "Alice"),
        ("aaa1", "2024-03-01T00:00:00", "Alice"),
    ], r.recent_commits


def test_file_history_recent_cap_truncates_to_newest():
    """``recent_cap`` caps the recent-commits list (newest-first); contributors
    are NOT capped (they aggregate over ALL commits regardless of cap)."""
    commits = [
        _commit("h1", "m1", datetime(2024, 1, 1), "Alice"),
        _commit("h2", "m2", datetime(2024, 1, 5), "Bob"),
        _commit("h3", "m3", datetime(2024, 1, 3), "Alice"),
        _commit("h4", "m4", datetime(2024, 1, 7), "Alice"),
    ]
    r = build_file_history("f.py", None, commits, recent_cap=2)
    # Only the 2 newest commits survive the cap.
    assert [c.hash for c in r.recent_commits] == ["h4", "h2"], r.recent_commits
    # But contributors aggregate over all 4 (Alice=3, Bob=1).
    assert r.top_contributors == [
        ContributorView(author="Alice", commits=3),
        ContributorView(author="Bob", commits=1),
    ], r.top_contributors
    assert r.last_modified is None


def test_file_history_contributor_tie_broken_by_author_name():
    """A tie in commit count is broken by author name ascending (stable,
    deterministic — same posture as the search tools' ``order_by``)."""
    commits = [
        _commit("h1", "m1", datetime(2024, 1, 1), "Zoe"),
        _commit("h2", "m2", datetime(2024, 1, 2), "Amy"),
    ]
    r = build_file_history("f.py", None, commits)
    assert [c.author for c in r.top_contributors] == ["Amy", "Zoe"]


def test_file_history_empty_commits_clear_empty_with_path_and_last_modified():
    """No commits → empty contributor / recent lists, but file_path +
    last_modified are still populated (the walker set last_modified even before
    §7 step 7 lands — the honest "no history indexed yet" shape)."""
    r = build_file_history("pkg/app.py", datetime(2024, 3, 10), [])
    assert r.file_path == "pkg/app.py"
    assert r.last_modified == "2024-03-10T00:00:00"
    assert r.top_contributors == [] and r.recent_commits == []


def test_file_history_none_path_is_unresolvable_target_clear_empty():
    """A ``None`` file_path with no commits is the unresolvable-target case the
    live wrapper hands the pure core → clear empty (file_path None,
    last_modified None, empty lists)."""
    r = build_file_history(None, None, [])
    assert r.file_path is None
    assert r.last_modified is None
    assert r.top_contributors == [] and r.recent_commits == []


# ═══════════════════════════════════════════════════════════════════════════
# pure: build_github_metadata (issue/PR split + linked-file filter + sort)
# ═══════════════════════════════════════════════════════════════════════════

_ISSUE_ROWS = [
    (101, "Bug X", "open", ["bug"], "https://github.com/o/r/issues/101", ["pkg/app.py"]),
    (102, "Feature", "closed", ["feature"], "https://github.com/o/r/pull/102", ["pkg/app.py", "pkg/util.py"]),
    (100, "Old", "closed", [], "https://github.com/o/r/issues/100", None),
    (103, "Weird", "open", [], None, ["pkg/app.py"]),  # url None -> issue bucket
]


def test_github_metadata_partitions_by_url_and_sorts_desc():
    """Whole-repo (no target_path): split rows into issues/prs by the URL
    heuristic (``/pull/`` → PR, else issue; ``None`` url → issue), each sorted
    by number desc. PR #102 → prs; #101/#100/#103 (None url) → issues."""
    r = build_github_metadata(_ISSUE_ROWS)
    assert isinstance(r, GitHubMetadataResult)
    assert r.file_path is None
    assert [i.number for i in r.issues] == [103, 101, 100], r.issues  # desc; 103 first
    assert [p.number for p in r.prs] == [102], r.prs
    # labels coerced from nullable array; null → [].
    assert r.issues[2].labels == []  # #100 had [] labels
    assert r.issues[2].linked_files == []  # None linked_files -> []


def test_github_metadata_filters_by_target_path_membership():
    """With target_path set, keep a row iff target is a member of its
    ``linked_files`` (PyGithub's explicit link metadata — no text guessing).
    #100 (None linked_files) is dropped under a target filter; a NULL array
    can't match. file_path carries the target."""
    r = build_github_metadata(_ISSUE_ROWS, target_path="pkg/app.py")
    assert r.file_path == "pkg/app.py"
    assert [i.number for i in r.issues] == [103, 101], r.issues  # #100 dropped (None linked)
    assert [p.number for p in r.prs] == [102], r.prs


def test_github_metadata_target_path_matching_no_rows_is_clear_empty():
    """A target_path that no row links to → empty both lists, file_path set:
    the honest "no issues/PRs linked to this file" answer (not a crash)."""
    r = build_github_metadata(_ISSUE_ROWS, target_path="pkg/ghost.py")
    assert r.file_path == "pkg/ghost.py"
    assert r.issues == [] and r.prs == []


def test_github_metadata_empty_rows_clear_empty():
    """No rows at all → clear empty (the backing-table-empty case until §7
    step 8 lands)."""
    r = build_github_metadata([])
    assert isinstance(r, GitHubMetadataResult)
    assert r.issues == [] and r.prs == []
    assert r.file_path is None
    # IssueView is the JSON-serializable shape both lists use.
    assert all(isinstance(v, IssueView) for v in (r.issues + r.prs))


def test_github_metadata_pr_split_by_url_not_by_state():
    """The split is by URL convention (``/pull/``), NOT by state — a closed PR
    is still a PR (#102 is closed). Pins the documented heuristic (the §11
    table has no type discriminator column)."""
    r = build_github_metadata([
        (1, "x", "closed", [], "https://github.com/o/r/pull/1", None),
        (2, "y", "open", [], "https://github.com/o/r/issues/2", None),
    ])
    assert [p.number for p in r.prs] == [1]
    assert [i.number for i in r.issues] == [2]


# ═══════════════════════════════════════════════════════════════════════════
# synthetic-DB: resolve_file_id (the plan-step identifier ↦ file_id adapter)
# ═══════════════════════════════════════════════════════════════════════════


def _setup_files_repo(session) -> tuple[int, dict[str, int]]:
    """Persist a throwaway repo + a small file tree and return
    ``(repo_id, {posix_path: file_id})``. Layout exercised by the resolver
    tests: an exact-path file, two module-tail files, and a shared-basename
    set for the ambiguity test."""
    session.add(Repository(url_or_path="file:///test/resolve", name="test/resolve", status="indexing"))
    session.commit()
    repo_id = session.execute(
        sa.select(Repository.id).where(Repository.name == "test/resolve")
    ).scalar_one()

    paths = [
        "src/flask/app.py",
        "src/flask/helpers.py",
        "src/flask/__init__.py",
        "src/flask/sansio/app.py",   # shares basename with src/flask/app.py
        "src/flask/blueprints.py",
    ]
    rows = [File(repository_id=repo_id, path=p, language="python", loc=10) for p in paths]
    session.add_all(rows)
    session.flush()
    session.commit()
    return repo_id, {f.path: f.id for f in rows}


def _cleanup_files_repo(session, repo_id: int | None) -> None:
    """FK-safe teardown: files -> repo (no edges/commits/issues for this setup)."""
    if repo_id is not None:
        session.execute(sa.delete(File).where(File.repository_id == repo_id))
        session.execute(sa.delete(Repository).where(Repository.id == repo_id))
        session.commit()


@pytest.fixture
def resolve_repo():
    """Yields ``(session, repo_id, paths)`` for a synthetic resolve repo, cleaning
    up after (even on failure). Skip-decorated individually below so the body
    only runs when Postgres is up."""
    repo_id = None
    paths = {}
    session_cm = SessionLocal()
    session = session_cm.__enter__()
    try:
        repo_id, paths = _setup_files_repo(session)
        yield session, repo_id, paths
    finally:
        _cleanup_files_repo(session, repo_id)
        session_cm.__exit__(None, None, None)


@db_required
def test_resolve_file_id_exact_path_wins(resolve_repo):
    """An exact ``files.path == target`` hit is the decisive first stage — the
    Search Agent returns exact paths, so this is the common case."""
    session, repo_id, paths = resolve_repo
    assert resolve_file_id(repo_id, "src/flask/app.py", session) == paths["src/flask/app.py"]


@db_required
def test_resolve_file_id_dotted_module_name_resolves(resolve_repo):
    """A dotted module name ``flask.app`` resolves to ``src/flask/app.py`` via
    the ``<path-form>.py`` tail match (the dot is a module separator, not a
    path one) — even though the stored path has the ``src/`` prefix."""
    session, repo_id, paths = resolve_repo
    assert resolve_file_id(repo_id, "flask.app", session) == paths["src/flask/app.py"]


@db_required
def test_resolve_file_id_dotted_package_name_resolves_to_init(resolve_repo):
    """``flask`` (a bare package name) resolves to ``src/flask/__init__.py`` via
    the ``<path-form>/__init__.py`` tail match — the package-root identifier."""
    session, repo_id, paths = resolve_repo
    assert resolve_file_id(repo_id, "flask", session) == paths["src/flask/__init__.py"]


@db_required
def test_resolve_file_id_ambiguous_basename_returns_none(resolve_repo):
    """A bare ``app`` matches BOTH ``src/flask/app.py`` and
    ``src/flask/sansio/app.py`` (shared basename) — ambiguous, so the resolver
    returns ``None`` rather than guessing. The "don't guess" posture."""
    session, repo_id, paths = resolve_repo
    # Substring "app.py" hits two files -> None (not the first one).
    assert resolve_file_id(repo_id, "app.py", session) is None


@db_required
def test_resolve_file_id_unique_substring_resolves(resolve_repo):
    """A substring that matches exactly one file is decisive: ``blueprints``
    only appears in ``src/flask/blueprints.py`` → resolves to that file."""
    session, repo_id, paths = resolve_repo
    assert resolve_file_id(repo_id, "blueprints", session) == paths["src/flask/blueprints.py"]


@db_required
def test_resolve_file_id_unresolvable_returns_none(resolve_repo):
    """An identifier matching nothing → ``None`` (the graph tools turn this
    into a clear empty result, the caller distinguishes from a missing key)."""
    session, repo_id, paths = resolve_repo
    assert resolve_file_id(repo_id, "no/such/file.py", session) is None
    assert resolve_file_id(repo_id, "totally.unknown.module", session) is None


@db_required
def test_resolve_file_id_empty_returns_none(resolve_repo):
    """Empty / whitespace target → ``None`` (handled before any query)."""
    session, repo_id, paths = resolve_repo
    assert resolve_file_id(repo_id, "", session) is None
    assert resolve_file_id(repo_id, "   ", session) is None


# ═══════════════════════════════════════════════════════════════════════════
# synthetic-DB: query_file_history (live wrapper: resolve + read + aggregate)
# ═══════════════════════════════════════════════════════════════════════════


def _setup_history_repo(session) -> tuple[int, dict[str, int], list[int]]:
    """Persist a repo + one file + three commits, linking two of them to the
    file via the ``file_commits`` join. Returns ``(repo_id, {path: file_id},
    [commit_ids])``. Layout: Alice(2 commits, one unlinked), Bob(1 linked)."""
    session.add(Repository(url_or_path="file:///test/history", name="test/history", status="indexing"))
    session.commit()
    repo_id = session.execute(
        sa.select(Repository.id).where(Repository.name == "test/history")
    ).scalar_one()

    f = File(repository_id=repo_id, path="pkg/app.py", language="python", loc=10,
             last_modified=datetime(2024, 3, 10))
    session.add(f)
    session.flush()  # assign f.id
    session.commit()

    commits = [
        Commit(repository_id=repo_id, hash="aaa1", author="Alice",
               date=datetime(2024, 3, 1), message="init"),
        Commit(repository_id=repo_id, hash="bbb2", author="Bob",
               date=datetime(2024, 3, 5), message="fix"),
        Commit(repository_id=repo_id, hash="ccc3", author="Alice",
               date=datetime(2024, 3, 3), message="refactor (unlinked)"),
    ]
    session.add_all(commits)
    session.flush()
    session.commit()
    cids = [c.id for c in commits]

    # Link the first (init, Alice) and second (fix, Bob); leave the refactor
    # unlinkled to prove the aggregation only sees linked commits.
    session.execute(file_commits.insert().values(file_id=f.id, commit_id=cids[0]))
    session.execute(file_commits.insert().values(file_id=f.id, commit_id=cids[1]))
    session.commit()

    return repo_id, {"pkg/app.py": f.id}, cids


def _cleanup_history_repo(session, repo_id: int | None) -> None:
    """FK-safe teardown: file_commits -> commits -> files -> repo. file_commits
    has no repository_id column, so delete its rows via the file-id subquery
    before files/commits get swept (otherwise the FK dangles)."""
    if repo_id is None:
        return
    file_ids = sa.select(File.id).where(File.repository_id == repo_id)
    session.execute(sa.delete(file_commits).where(file_commits.c.file_id.in_(file_ids)))
    session.execute(sa.delete(Commit).where(Commit.repository_id == repo_id))
    session.execute(sa.delete(File).where(File.repository_id == repo_id))
    session.execute(sa.delete(Repository).where(Repository.id == repo_id))
    session.commit()


@pytest.fixture
def history_repo():
    repo_id = None
    session_cm = SessionLocal()
    session = session_cm.__enter__()
    try:
        repo_id, _paths, _cids = _setup_history_repo(session)
        yield session, repo_id
    finally:
        _cleanup_history_repo(session, repo_id)
        session_cm.__exit__(None, None, None)


@db_required
def test_query_file_history_aggregates_linked_commits(history_repo):
    """The live wrapper resolves the target, reads ``files.last_modified`` +
    commits joined through ``file_commits``, and aggregates. The unlinked
    refactor commit does NOT appear in the file's history (only linked ones
    do). Alice=1, Bob=1 (only the linked commits counted); both authors tie →
    broken by name; recent commits newest-first (Bob's 3/5, Alice's 3/1)."""
    session, repo_id = history_repo
    r = query_file_history(repo_id, "pkg/app.py", session)
    assert isinstance(r, FileHistoryResult)
    assert r.file_path == "pkg/app.py"
    assert r.last_modified == "2024-03-10T00:00:00"
    # Two linked commits (one Alice, one Bob); tie at 1 each -> author asc.
    assert r.top_contributors == [
        ContributorView(author="Alice", commits=1),
        ContributorView(author="Bob", commits=1),
    ], r.top_contributors
    assert [c.hash for c in r.recent_commits] == ["bbb2", "aaa1"], r.recent_commits
    # The unlinked refactor commit must not appear.
    assert all(c.hash != "ccc3" for c in r.recent_commits)


@db_required
def test_query_file_history_unresolvable_target_clear_empty(history_repo):
    """An unresolvable target → clear empty (file_path None, last_modified
    None, empty lists) — the live wrapper catches the resolver's None and
    returns the honest empty, never crashing."""
    session, repo_id = history_repo
    r = query_file_history(repo_id, "no/such/file.py", session)
    assert r.file_path is None
    assert r.last_modified is None
    assert r.top_contributors == [] and r.recent_commits == []


@db_required
def test_query_file_history_dotted_module_target_resolves(history_repo):
    """A dotted module name resolves to the file (path-tail match), so
    ``pkg.app`` reaches the same row as ``pkg/app.py`` — the identifier
    adapter make a plan arg in either form dispatch correctly."""
    session, repo_id = history_repo
    r = query_file_history(repo_id, "pkg.app", session)
    assert r.file_path == "pkg/app.py"


@db_required
def test_query_file_history_recent_cap_is_coerced_and_floored(history_repo):
    """``recent_cap`` is int-coerced (a stray ``"1"`` survives) and floored at
    0; a cap of 1 yields just the newest commit but leaves contributors
    aggregated over ALL linked commits (the cap bounds only the recent list)."""
    session, repo_id = history_repo
    r = query_file_history(repo_id, "pkg/app.py", session, recent_cap="1")
    assert [c.hash for c in r.recent_commits] == ["bbb2"]
    assert len(r.top_contributors) == 2  # both authors still aggregated


# ═══════════════════════════════════════════════════════════════════════════
# synthetic-DB: query_github_metadata (live wrapper: resolve + filter + split)
# ═══════════════════════════════════════════════════════════════════════════


def _setup_issues_repo(session) -> tuple[int, dict[str, int]]:
    """Persist a repo + one file + three issues (one issue, one PR, one issue
    with no linked files). Returns ``(repo_id, {path: file_id})``."""
    session.add(Repository(url_or_path="file:///test/issues", name="test/issues", status="indexing"))
    session.commit()
    repo_id = session.execute(
        sa.select(Repository.id).where(Repository.name == "test/issues")
    ).scalar_one()

    f = File(repository_id=repo_id, path="pkg/app.py", language="python", loc=1)
    session.add(f)
    session.flush()
    session.commit()

    issues = [
        Issue(repository_id=repo_id, number=101, title="Bug X", state="open",
              labels=["bug"], url="https://github.com/o/r/issues/101",
              linked_files=["pkg/app.py"]),
        Issue(repository_id=repo_id, number=102, title="Feature", state="closed",
              labels=["feature"], url="https://github.com/o/r/pull/102",
              linked_files=["pkg/app.py", "pkg/util.py"]),
        Issue(repository_id=repo_id, number=100, title="Old", state="closed",
              labels=[], url="https://github.com/o/r/issues/100",
              linked_files=[]),  # no linked files -> excluded by ANY filter
    ]
    session.add_all(issues)
    session.commit()
    return repo_id, {"pkg/app.py": f.id}


def _cleanup_issues_repo(session, repo_id: int | None) -> None:
    if repo_id is None:
        return
    session.execute(sa.delete(Issue).where(Issue.repository_id == repo_id))
    session.execute(sa.delete(File).where(File.repository_id == repo_id))
    session.execute(sa.delete(Repository).where(Repository.id == repo_id))
    session.commit()


@pytest.fixture
def issues_repo():
    repo_id = None
    session_cm = SessionLocal()
    session = session_cm.__enter__()
    try:
        repo_id, _paths = _setup_issues_repo(session)
        yield session, repo_id
    finally:
        _cleanup_issues_repo(session, repo_id)
        session_cm.__exit__(None, None, None)


@db_required
def test_query_github_metadata_whole_repo_splits_issues_and_prs(issues_repo):
    """Whole-repo (no target): partitions all rows into issues (#101, #100) /
    PRs (#102) by URL, sorted by number desc. file_path is None."""
    session, repo_id = issues_repo
    r = query_github_metadata(repo_id, None, session)
    assert isinstance(r, GitHubMetadataResult)
    assert r.file_path is None
    assert [i.number for i in r.issues] == [101, 100], r.issues
    assert [p.number for p in r.prs] == [102], r.prs


@db_required
def test_query_github_metadata_target_filter_excludes_null_linked(issues_repo):
    """With target ``pkg/app.py``: keeps rows whose ``linked_files`` contains
    it (#101, #102); #100 (empty linked_files) is excluded by the ARRAY ANY
    predicate (NULL/empty never matches). file_path is the resolved path."""
    session, repo_id = issues_repo
    r = query_github_metadata(repo_id, "pkg/app.py", session)
    assert r.file_path == "pkg/app.py"
    assert [i.number for i in r.issues] == [101], r.issues  # #100 dropped
    assert [p.number for p in r.prs] == [102], r.prs


@db_required
def test_query_github_metadata_dotted_target_re_resolved_to_path(issues_repo):
    """``pkg.app`` (dotted module name) re-resolves to the canonical
    ``pkg/app.py`` and THEN filters ``linked_files`` by that path — so a
    module-name plan arg still matches path-form ``linked_files`` entries."""
    session, repo_id = issues_repo
    r = query_github_metadata(repo_id, "pkg.app", session)
    assert r.file_path == "pkg/app.py"
    assert [i.number for i in r.issues] == [101]


@db_required
def test_query_github_metadata_unresolvable_target_clear_empty(issues_repo):
    """An unresolvable target → clear empty (file_path None, both lists empty),
    not a crash — the same "don't guess" posture the sibling tools hold."""
    session, repo_id = issues_repo
    r = query_github_metadata(repo_id, "ghost/module.py", session)
    assert r.file_path is None
    assert r.issues == [] and r.prs == []


# ═══════════════════════════════════════════════════════════════════════════
# live vs flask: graph tools (real structure) + metadata tools (clear-empty)
# ═══════════════════════════════════════════════════════════════════════════


@flask_imports_required
def test_query_architecture_flask_whole_repo_has_modules_and_edges():
    """Whole-repo architecture over the indexed flask import graph: scope
    "whole", a non-empty module map (flask is a package), a ranked key-files
    list, and non-empty edges (187 internal import edges were indexed)."""
    with SessionLocal() as session:
        r = query_architecture(FLASK_IMPORTS_REPO_ID, None, session)
    assert r.scope == "whole"
    assert r.focus_path is None
    assert r.modules, "expected a non-empty module map for flask"
    assert any(m.name and "flask" in m.name for m in r.modules), r.modules
    assert r.key_files, "expected ranked key files for flask"
    assert r.edges, "expected internal import edges for flask"
    # JSON-serializable: the §10 "structured JSON, never free text" contract.
    import dataclasses
    import json

    json.dumps(dataclasses.asdict(r))


@flask_imports_required
def test_query_architecture_flask_focused_on_app_py():
    """Focused architecture on ``src/flask/app.py``: scope "file",
    focus_path set, an induced radius-bounded subgraph with at least one
    edge (app.py imports and is imported by many)."""
    with SessionLocal() as session:
        r = query_architecture(FLASK_IMPORTS_REPO_ID, "src/flask/app.py", session)
    assert r.scope == "file"
    assert r.focus_path == "src/flask/app.py"
    assert r.edges, "expected app.py to have an import neighborhood"


@flask_imports_required
def test_query_dependency_graph_flask_app_py_has_neighborhood():
    """app.py's dependency neighborhood over the indexed flask import graph:
    it both imports (neighbors_out non-empty) and is depended on
    (neighbors_in non-empty) — flask's central file. node_path is the
    resolved path; depth defaults to 1."""
    with SessionLocal() as session:
        r = query_dependency_graph(FLASK_IMPORTS_REPO_ID, "src/flask/app.py", session)
    assert isinstance(r, DependencyGraphResult)
    assert r.node_path == "src/flask/app.py"
    assert r.depth == 1
    assert r.neighbors_out, "expected app.py to import other flask files"
    assert r.neighbors_in, "expected app.py to be imported by other flask files"
    # JSON-serializable.
    import dataclasses
    import json

    json.dumps(dataclasses.asdict(r))


@flask_imports_required
def test_query_dependency_graph_flask_dotted_module_resolves():
    """A dotted module name ``flask.app`` dispatch-resolves to app.py (the
    path-tail resolver), giving the same neighborhood as the path form."""
    with SessionLocal() as session:
        r_path = query_dependency_graph(FLASK_IMPORTS_REPO_ID, "src/flask/app.py", session)
        r_mod = query_dependency_graph(FLASK_IMPORTS_REPO_ID, "flask.app", session)
    assert r_mod.node_path == r_path.node_path == "src/flask/app.py"
    assert r_mod.neighbors_out == r_path.neighbors_out
    assert r_mod.neighbors_in == r_path.neighbors_in


@flask_imports_required
def test_query_dependency_graph_flask_unresolvable_target_clear_empty():
    """An unresolvable flask target → clear empty (node_path None, empty
    neighbor lists), never a crash."""
    with SessionLocal() as session:
        r = query_dependency_graph(FLASK_IMPORTS_REPO_ID, "no/such/flask/file.py", session)
    assert r.node_path is None
    assert r.neighbors_in == [] and r.neighbors_out == []


@flask_imports_required
def test_query_dependency_graph_flask_depth_param_is_honored():
    """``depth=2`` widens the closure vs the default ``depth=1`` — at least one
    of the two neighbor lists grows (flask's import graph is connected enough
    that 2 hops reach strictly more than 1 hop for its central file)."""
    with SessionLocal() as session:
        r1 = query_dependency_graph(FLASK_IMPORTS_REPO_ID, "src/flask/app.py", session, depth=1)
        r2 = query_dependency_graph(FLASK_IMPORTS_REPO_ID, "src/flask/app.py", session, depth=2)
    assert r1.depth == 1 and r2.depth == 2
    # The closure at depth 2 is a superset of depth 1 (monotone in depth).
    ids_out_1 = {n.file_id for n in r1.neighbors_out}
    ids_out_2 = {n.file_id for n in r2.neighbors_out}
    assert ids_out_1 <= ids_out_2
    assert ids_out_2 - ids_out_1, "depth=2 should reach strictly more out-neighbors"


@flask_required
def test_query_file_history_flask_clear_empty_until_section7():
    """file_history over flask today: the walker populated ``last_modified``
    (§7 step 1, done) but the §7 step 7 git-history indexer is not yet built,
    so contributor / commit lists are empty — the documented
    build-the-tool-flag-the-pending-backing posture. This pins that honest
    interim shape so the moment §7 step 7 lands the test is updated, not
    silently passing."""
    with SessionLocal() as session:
        # Find a flask file path to target (resolves via the real resolver).
        sample = session.scalar(
            sa.select(File.path)
            .where(File.repository_id == FLASK_REPO_ID)
            .order_by(File.path)
            .limit(1)
        )
    assert sample, "flask should have indexed files"
    with SessionLocal() as session:
        r = query_file_history(FLASK_REPO_ID, sample, session)
    assert isinstance(r, FileHistoryResult)
    assert r.file_path == sample
    # last_modified is populated (walker §7 step 1), lists empty until §7 step 7.
    assert r.last_modified is not None
    assert r.top_contributors == [] and r.recent_commits == [], (
        "§7 step 7 git-history indexer must have landed — update this test "
        "(the clear-empty-until-§7 posture no longer holds) and the tool docstring."
    )


@flask_required
def test_query_github_metadata_flask_clear_empty_until_section8():
    """github_metadata over flask today: the ``issues`` table exists in the
    schema but §7 step 8 (PyGithub) is not yet built, so both lists are
    empty — the documented pending-backing posture (pinned so the moment §7
    step 8 lands the test is updated, not silently passing)."""
    with SessionLocal() as session:
        r = query_github_metadata(FLASK_REPO_ID, None, session)
    assert isinstance(r, GitHubMetadataResult)
    assert r.file_path is None
    assert r.issues == [] and r.prs == [], (
        "§7 step 8 PyGithub indexer must have landed — update this test "
        "(the clear-empty-until-§7 posture no longer holds) and the tool docstring."
    )
