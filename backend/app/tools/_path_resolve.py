"""Shared path → ``file_id`` resolution for the graph-leaning context tools
(SDD §10 Architecture Query / Dependency Graph Query).

The §9.3 Context Agent's structural inputs (SDD §9.3: "typically a file or
symbol identifier from the Search Agent's output") arrive as *identifiers*, not
``file_id`` ints: a plan step carries ``{"target": "src/flask/app.py"}`` or
sometimes a dotted module name ``{"target": "flask.app"}``. The import graph's
edges table, by contrast, keys nodes by ``file_id`` (``edges.source_id`` /
``edges.target_id``). This module is the adapter that turns an identifier the
Planner/Search might emit into the ``file_id`` the graph math operates on.

Kept as a shared internal module (underscore-prefixed — *not* a tool an agent
calls by name) rather than the duplicated one-liner posture of
:func:`app.tools.symbol_search._ilike_contains` / :func:`app.tools.file_search._ilike_contains`
because the resolution is non-trivial multi-stage logic that two tools
(Architecture Query + Dependency Graph Query) genuinely share. A one-line helper
earns a copy per module for self-containment; ~60 lines of ranking earned a
single home.

Resolution is **best-effort, not authoritative**, consistent with the project's
"flag rather than guess" posture for the import graph (SDD §8 / the PEP 420
namespace note in :mod:`app.indexing.import_graph`):

* An exact ``files.path == target`` hit wins (the Search Agent returns exact
  paths, the common case).
* Failing that, a dotted module name is tried as ``<path-form>.py`` and
  ``<path-form>/__init__.py`` (the whole tail, anywhere in the tree) — so
  ``"flask.app"`` resolves to ``src/flask/app.py`` even though the stored path is
  ``src/flask/app.py`` and the dot is a module separator, not a path one.
* Failing that, a *unique* substring match — but only when exactly one file
  matches, so a bare ``"app"`` that hits many files returns ``None`` (ambiguous)
  rather than guessing.
* A return of ``None`` is the caller's signal to produce a clear empty result, not
  a crash — the same contract the search tools hold ("an empty result is a valid
  answer", SDD §10). The graph tools treat an unresolvable node as an empty
  neighborhood, not a dispatch error: resolving is query-time fuzziness, while a
  *malformed plan step* (no ``target`` at all) is a dispatch failure surfaced one
  layer up in :mod:`app.agents.context_agent`.

Dotted-module-name resolution is the honest-weakness version of the import
graph's full :class:`app.indexing.import_graph.ImportResolver` (src-layout package
roots, contiguous-__init__ chains): here it's a simple tail match, which covers
the common path-form and module-tail cases without re-deriving package roots at
query time. Honoring PEP 420 namespace dirs (``sansio/`` has no ``__init__.py``)
the way the import resolver does is out of scope at the tool layer — flagging
rather than guessing, as documented there.
"""
import os
import re
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import File, Repository

# Base directory where remote repositories are cloned locally
MANAGED_REPOS_DIR = Path(os.getenv("MANAGED_REPOS_DIR", "data/repos"))


def resolve_repo_root(repository: Repository | int, session: Session) -> Path | None:
    """Resolve the local on-disk directory root for a repository (local path or cloned GitHub repo)."""
    if isinstance(repository, int):
        repo = session.get(Repository, repository)
    else:
        repo = repository
    if not repo:
        return None

    # 1. If url_or_path is an existing local directory, return it
    if repo.url_or_path:
        local_p = Path(repo.url_or_path)
        if local_p.is_dir():
            return local_p

    # 2. If it's a managed/cloned repository in data/repos
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", repo.name or "")
    managed_dir = MANAGED_REPOS_DIR / f"{repo.id}_{safe_name}"
    if managed_dir.is_dir():
        return managed_dir

    # 3. Search MANAGED_REPOS_DIR for prefix "{repo.id}_"
    if MANAGED_REPOS_DIR.is_dir():
        for p in MANAGED_REPOS_DIR.iterdir():
            if p.is_dir() and p.name.startswith(f"{repo.id}_"):
                return p

    return None

# Escape char for the LIKE/ILIKE patterns emitted below — matches the twin in
# :mod:`app.tools.symbol_search` / :mod:`app.tools.file_search`. Escaping LIKE
# metacharacters keeps a literal ``__init__`` matching its underscores rather
# than the single-char wildcard ``_``.
LIKE_ESCAPE: str = "\\"


def _escape_like_fragment(fragment: str) -> str:
    """Escape LIKE metacharacters (``%``, ``_``, ``\\``) in ``fragment`` for use
    inside a ``%…%`` substring pattern. Pair with
    ``File.path.ilike(..., escape=LIKE_ESCAPE)``. The twin of
    :func:`app.tools.file_search._ilike_contains`'s escape half, kept here so
    this module is self-contained."""
    e = LIKE_ESCAPE
    return fragment.replace(e, e + e).replace("%", e + "%").replace("_", e + "_")


def _dotted_to_path(target: str) -> str:
    """The path form of a module-or-path identifier: dotted module names become
    path-form (``flask.app`` -> ``flask/app``); already-path-like inputs are
    returned unchanged (the dots in ``.py`` and in path segments are *literal*,
    not module separators). ``target`` is stripped first; an empty result
    collapses to ``""`` (the caller treats ``""`` as unresolvable).

    The "is it a path?" heuristic: if the input already contains a separator
    (``/`` or, after Windows normalization, ``/``) or ends in ``.py``, it's a
    *path* — its dots are literal, so they're kept. Only when it looks like a
    bare dotted module name (no separator, no ``.py``) do we translate dots to
    path separators. This keeps ``"src/flask/app.py"`` faithful rather than
    mangling it to ``src/flask/app/py`` (the mangle would corrupt the module-
    tail + substring match stages, though the exact-match stage would still
    catch it)."""
    target = (target or "").strip()
    if not target:
        return ""
    # Windows path backslashes -> forward slashes so the same identifier resolves
    # regardless of the host the plan was built on.
    target = target.replace("\\", "/")
    if "/" in target or target.endswith(".py"):
        return target  # a path: dots are literal (``.py`` or in segment names)
    return target.replace(".", "/")


def resolve_file_id(
    repository_id: int,
    target: str,
    session: Session,
) -> int | None:
    """Resolve ``target`` to a single ``files.id`` for ``repository_id``, or
    ``None`` if it does not map to exactly one file.

    Resolution order (first decisive hit wins; an ambiguous stage falls through
    to the next):

    1. **Exact path** — ``files.path == target`` (the Search Agent's documented
       output form; the common case).
    2. **Dotted-name-as-module** — ``target`` read as a dotted module name,
       matched against ``<path-form>.py`` anywhere in the tree (tail match),
       so ``"flask.app"`` -> ``src/flask/app.py``.
    3. **Dotted-name-as-package** — the same tail matched against
       ``<path-form>/__init__.py`` (``"flask"`` -> ``src/flask/__init__.py``).
    4. **Unique substring** — ``files.path ILIKE '%<escaped>%'``; used only when
       exactly one file matches, so a partial name that hits many files returns
       ``None`` rather than guessing. Among ties at a stage, the **shortest**
       path (fewest ``/``) wins as the most-canonical candidate; a remaining tie
       is still ambiguous -> ``None``.

    Returns ``None`` (not raises) when ``target`` is empty or unresolvable — the
    graph tools turn this into a clear empty result, and a caller that needs to
    distinguish "bad plan" (no ``target`` key) from "unresolvable target" does so
    in :mod:`app.agents.context_agent`, where a missing key is a dispatch error.
    """
    if not target or not str(target).strip():
        return None
    target = str(target).strip()
    path_form = _dotted_to_path(target)

    # Pull every file whose path could be reading this identifier in any of the
    # match forms, ranking later only among this candidate set (cheaper than
    # four round-trips and keeps the ranking in one place). The OR spans exact +
    # both module tails + substring; we then score.
    e = LIKE_ESCAPE
    exact = target
    mod_py = path_form + ".py"
    mod_pkg = path_form + "/__init__.py"
    sub = _escape_like_fragment(path_form)

    clauses = [
        File.path == exact,
        # ends-with module-as-file (e.g. ``.../flask/app.py``)
        File.path.like(f"%{e}/{_escape_like_fragment(mod_py)}", escape=e)
        if path_form
        else sa.false(),
        # exactly the module-as-file at top level (no leading dir): ``flask/app.py`` is rare but covered
        File.path == mod_py,
        # ends-with package __init__ (e.g. ``.../flask/__init__.py``)
        File.path.like(f"%{e}/{_escape_like_fragment(mod_pkg)}", escape=e)
        if path_form
        else sa.false(),
        File.path == mod_pkg,
        # substring fallback (only decisive when unique, scored last)
        File.path.ilike(f"%{sub}%", escape=e),
    ]
    stmt = (
        sa.select(File.id, File.path)
        .where(File.repository_id == repository_id)
        .where(sa.or_(*clauses))
    )
    rows = list(session.execute(stmt).all())

    if not rows:
        return None

    # Rank: exact == 0, ends-with module .py == 1, ends-with package __init__ == 2,
    # substring == 3. Among same-rank candidates prefer the shortest path.
    def _rank(path: str) -> tuple[int, int]:
        if path == exact:
            return (0, len(path))
        if path == mod_py or (path_form and path.endswith("/" + mod_py)):
            return (1, len(path))
        if path == mod_pkg or (path_form and path.endswith("/" + mod_pkg)):
            return (2, len(path))
        return (3, len(path))

    ranked = sorted(rows, key=lambda r: _rank(r.path))
    best_rank = _rank(ranked[0].path)
    best_group = [r for r in ranked if _rank(r.path)[0] == best_rank[0]]
    if len(best_group) == 1:
        return best_group[0].id
    # Tie within the best rank (ambiguous): only accept at the exact tier (rank 0
    # can't tie — `path == exact` is a unique row). Substring/module-tier ties
    # mean "several files plausibly match this identifier" -> don't guess.
    return None


def _package_dirs(paths: list[str]) -> set[str]:
    """The set of POSIX directory paths that are *packages*: the dirname of each
    ``__init__.py`` path (and ``""`` for a repo-root ``__init__.py``). A file is
    grouped under a module by its nearest enclosing package dir
    (:func:`_enclosing_module`); this derivation mirrors the import graph's
    package detection (:class:`app.indexing.import_graph.ImportResolver`)."""
    dirs: set[str] = set()
    for p in paths:
        if p.endswith("/__init__.py"):
            dirs.add(p[: -len("/__init__.py")])
        elif p == "__init__.py":
            dirs.add("")
    return dirs


def _enclosing_module(file_path: str, package_dirs: set[str]) -> str:
    """The nearest enclosing package dir of ``file_path`` (the longest
    ``package_dirs`` member that is a directory ancestor of the file), or ``""``
    if the file lives under no package (a bare script). ``__init__.py`` files map
    to their own dir's package label (the package they declare), not their
    parent — so ``src/flask/__init__.py`` -> ``"src/flask"``.

    A near-ancestor check (is ``pkg`` a prefix-dir of ``file_path``) keeps
    ``flask`` from matching ``flask_admin`` — the dir boundary (``/``) must
    align, the same care the import resolver takes."""
    if not package_dirs:
        return ""
    # Resolve __init__.py to its own directory's package label.
    if file_path.endswith("/__init__.py"):
        own = file_path[: -len("/__init__.py")]
        if own in package_dirs:
            return own
        # an __init__.py not in our package_dirs (not seen as one) -> fall through
    elif file_path == "__init__.py__":
        return "" if "" in package_dirs else ""

    # Longest prefix-dir among package_dirs that is an ancestor of the file.
    best = ""
    file_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
    for pkg in package_dirs:
        if not pkg:
            continue
        # pkg is a dir-ancestor iff file_path == pkg/... (pkg is a path prefix
        # followed by a separator) OR file_dir == pkg.
        if file_path == pkg + "/__init__.py" or file_path.startswith(pkg + "/"):
            if len(pkg) > len(best):
                best = pkg
    return best
