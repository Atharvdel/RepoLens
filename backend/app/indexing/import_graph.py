"""Import-graph stage of the RepoLens indexing pipeline (SDD §7 step 4).

Builds the file-level dependency graph: one ``edges`` row of ``edge_type =
"imports"`` per Python **file → module** dependency, resolved to an internal
``files`` row where we can confidently map it and flagged ``external`` (with the
module dotted name in ``target_label``) where we cannot (SDD §7 step 4 +
§8/§18 — "explicitly flag unresolved imports as external/unresolved rather than
guessing").

What counts as one edge:

* ``import a.b``                            → one edge to the module ``a.b``
* ``import a.b, c.d``  (``import`` form is list-shaped) → one edge **per module**
  (``a.b`` and ``c.d`` are different targets)
* ``from flask import Flask, request``      → **one** edge to the module ``flask``
  (the imported *names* are a reference-graph concern — SDD §7 step 5 — not the
  file-level import graph; two names from one module are still one module
  dependency)
* ``from .sub import bar``                  → one edge to the resolved ``.sub`` file
* ``from . import bar, baz``               → **two** edges (``bar`` and ``baz``
  are *each* a submodule candidate of the current package, not the same module)

Duplicate ``(level, target)`` pairs within one file collapse to a single edge
(``from .helpers import x`` and ``from .helpers import y`` are one dependency on
``.helpers``).

Scope contrast with :mod:`app.indexing.parser`: the symbol parser walks *only*
top-level definitions, deliberately dropping nested ``def``s (implementation
detail, not addressable). Import extraction does the **opposite** — it walks the
*entire* tree, because an import hidden inside ``if TYPE_CHECKING:`` or a
``try/except ImportError`` block is still a real dependency the file relies on.
A nested def is noise for navigation; a conditional import is signal for the
dependency graph. The line recorded is the import statement's own line.

Three cooperating pieces:

* :func:`parse_imports` — *pure*: reads bytes from disk, returns a list of
  :class:`ParsedImport` (one per ``(level, target)`` dependency, deduped). No
  DB, no session. This is the unit-testable core.
* :class:`ImportResolver` — *pure data* (no DB session): built once from a
  repository's file rows ``[(id, posix_path)]``, it maps both absolute dotted
  module names and relative import specs to ``file_id`` values. Holding only
  plain dicts keeps it cheap to build once and reuse across every file in a
  repo (the symbol indexer needs no such context; the import graph does,
  because resolution is cross-file).
* :func:`index_file_imports` — *persists*: takes a parsed list, resolves each
  via the resolver, and writes ``Edge`` rows. Returns an :class:`IndexResult`
  (added / resolved-to-internal / flagged-external counts). Does **not commit**;
  the caller owns the transaction, exactly like
  :func:`app.indexing.walker.walk_repository` and
  :func:`app.indexing.parser.index_file_symbols` (SDD §7 step 9). No ``flush`` is
  needed — both ``target_id`` (a previously-persisted file) and
  ``repository_id`` (a committed repo) reference rows that already have PKs, so
  nothing inside the batch needs a deferred FK.

:func:`parse_and_index_imports` glues the three together for the future
``pipeline.py`` orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tree_sitter_python as tsp
from sqlalchemy.orm import Session
from tree_sitter import Language, Node, Parser

from app.models import Edge

# The two top-level import-statement node kinds tree-sitter-python exposes. We
# walk the whole tree (not just root children) so conditional / nested imports
# are captured too — see the module docstring's scope note.
IMPORT_STATEMENT_TYPES: frozenset[str] = frozenset(
    {"import_statement", "import_from_statement"}
)

# Node types that carry a *single* dotted module name as raw text. ``dotted_name``
# is the bare form (``os.path``); ``aliased_import`` wraps a ``dotted_name`` plus
# ``as <alias>`` (``os.path as oss``). ``wildcard_import`` (``*``) names no module.
ALIASED_IMPORT = "aliased_import"
DOTTED_NAME = "dotted_name"
RELATIVE_IMPORT = "relative_import"
WILDCARD_IMPORT = "wildcard_import"
IMPORT_PREFIX = "import_prefix"


@dataclass
class ParsedImport:
    """One file → module dependency, in the minimal shape the resolver needs.

    ``target`` is the module operand with relative dots **stripped**:

    * ``import a.b``            → ``target="a.b"``, ``level=0``
    * ``from flask import Flask`` → ``target="flask"``, ``level=0``
    * ``from .sub import bar``  → ``target="sub"``,  ``level=1``
    * ``from ..pkg.mod import x`` → ``target="pkg.mod"``, ``level=2``
    * ``from . import bar``     → ``target="bar"``,  ``level=1`` — note that for
      the empty-tail relative form each imported *name* is its own target (a
      submodule candidate of the current package), so the caller emits one
      ``ParsedImport`` per name.

    ``level`` is the count of leading dots (0 for absolute). ``line`` is the
    1-indexed line of the import statement (first occurrence wins on dedup).
    """

    target: str
    level: int
    line: int


@dataclass
class IndexResult:
    """Tally returned by :func:`index_file_imports`.

    ``added`` is total edges written this call; it equals ``resolved + external``
    (every import produces exactly one edge — resolved or flagged — per SDD §8).
    """

    added: int
    resolved: int  # edges resolved to an internal file_id
    external: int  # edges flagged target_type="external"

    def __add__(self, other: "IndexResult") -> "IndexResult":
        return IndexResult(
            added=self.added + other.added,
            resolved=self.resolved + other.resolved,
            external=self.external + other.external,
        )


# ─── pure extraction ──────────────────────────────────────────────────────────


def _iterate_import_nodes(root: Node) -> Iterable[Node]:
    """Yield every import-statement node anywhere in ``root``'s subtree, in
    document (source) order.

    A flat pre-order walk (rather than only ``root.children``) so conditional
    imports — ``if TYPE_CHECKING:`` blocks, ``try/except ImportError`` guards,
    function-local ``import`` for lazy deps — are collected. Imports are never
    wrapped in ``decorated_definition`` (unlike def/class), so no unwrapping is
    needed; we simply descend into every named and anonymous child.

    Document order is an intentional, tested invariant (:func:`parse_imports`
    dedups by ``(level, target)`` keeping the *first-seen* line, and
    ``test_parse_imports_is_pure`` pins the walk order), so the stack is pushed
    with children **reversed** — see the push site for why.
    """
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.type in IMPORT_STATEMENT_TYPES:
            yield node
            # Imports have no nested imports of their own (their children are
            # dotted names / aliases), so no need to descend further.
            continue
        # Intentionally push *all* children (named and punctuation) — import
        # nodes live at arbitrary nesting depths inside if/try/with blocks.
        # Reversed before pushing so the LIFO stack pops them in *document
        # order*: pop() returns the last-pushed first, so pushing source-ordered
        # [A, B, C] reversed -> [C, B, A] yields A, then B, then C. Without the
        # reversal the last child surfaces first, reversing the import order —
        # the bug that made `test_parse_imports_is_pure` see `sibling` before
        # `collections` and (worse) made dedup keep the wrong *last* line.
        stack.extend(reversed(node.children))


def _dotted_text(node: Node) -> str:
    """The dotted module path of a ``dotted_name`` node as a string (``os.path``).

    A ``dotted_name``'s text is exactly the source between its delimiters, so
    decoding the node's bytes is faithful (no need to rejoin ``identifier``
    children). Whitespace is stripped defensively against odd source spacing.
    """
    return node.text.decode("utf-8", errors="replace").strip()


def _relative_decomposition(rel_node: Node) -> tuple[int, str]:
    """Split a ``relative_import`` node into ``(level, dotted_tail)``.

    ``level`` = number of leading dots (the ``import_prefix``); ``dotted_tail``
    is the dotted module after the dots, or ``""`` for the bare ``from . import``
    form. The ``import_prefix`` is composed solely of ``.`` tokens, so counting
    dots in its bytes is robust to multi-dot prefixes (``..``, ``...``).
    """
    level = 0
    tail = ""
    for child in rel_node.children:
        if child.type == IMPORT_PREFIX:
            level = child.text.count(b".")
        elif child.type == DOTTED_NAME:
            tail = _dotted_text(child)
    return level, tail


def _imported_module_name(node: Node) -> str | None:
    """The bare dotted module/name an import operand names, alias stripped.

    * ``aliased_import`` (``os.path as oss``) → ``"os.path"`` (the part before
      ``as``); the alias is irrelevant to file-level resolution.
    * ``dotted_name`` (``os.path``)            → ``"os.path"``
    * ``wildcard_import`` (``*``)              → ``None`` — names no module to
      resolve; the caller skips it.

    For ``import_statement`` each named child is a module operand of this shape.
    For the empty-tail relative form (``from . import X``) each imported name is
    likewise one of these (a submodule candidate of the current package).
    """
    if node.type == ALIASED_IMPORT:
        for child in node.children:
            if child.type == DOTTED_NAME:
                return _dotted_text(child)
        return None  # an aliased import with no dotted_name — malformed tree
    if node.type == DOTTED_NAME:
        return _dotted_text(node)
    return None  # wildcard_import, punctuation, etc.


def parse_imports(path: Path | str) -> list[ParsedImport]:
    """Parse one Python file and return its import dependencies, deduped.

    *Pure*: reads bytes from ``path``, runs the Python grammar, walks the whole
    tree for import statements, and returns a list of :class:`ParsedImport`
    (one per ``(level, target)`` dependency — duplicates across statements in
    the same file collapse to one, first line kept). No DB, no session, no side
    effects. Raises ``OSError`` if ``path`` cannot be read.

    See the module docstring for exactly which import forms produce which edges.

    Like :func:`app.indexing.parser.parse_file`, tree-sitter is error-recovering:
    a syntactically broken file still yields a best-effort tree, extracted
    as-is. Dedicated parse-error reporting is a later pipeline-hardening step.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        from app.indexing.ts_js_parser import parse_ts_js_file_imports
        raw_imports = parse_ts_js_file_imports(p)
        ts_seen: dict[tuple[int, str], ParsedImport] = {}
        for raw in raw_imports:
            lvl = getattr(raw, "level", (1 if raw.is_relative else 0))
            key = (lvl, raw.module)
            if key not in ts_seen:
                ts_seen[key] = ParsedImport(target=raw.module, level=lvl, line=raw.line_number)
        return list(ts_seen.values())

    src = p.read_bytes()
    language = Language(tsp.language())
    parser = Parser(language)
    tree = parser.parse(src)
    root = tree.root_node

    # Dedup by (level, target); first occurrence wins the reported line. A dict
    # preserves insertion order, so the returned list is stable and follows the
    # order imports first appear in the file.
    seen: dict[tuple[int, str], ParsedImport] = {}

    def _emit(level: int, target: str, line: int) -> None:
        if not target:
            return  # defensive: no module operand to resolve (e.g. bare `import`)
        key = (level, target)
        if key not in seen:
            seen[key] = ParsedImport(target=target, level=level, line=line)

    for node in _iterate_import_nodes(root):
        line = node.start_point[0] + 1
        if node.type == "import_statement":
            # `import X` / `import Y as z` / `import a.b, c.d` — every named
            # child is a separate absolute module operand (level 0).
            for child in node.named_children:
                name = _imported_module_name(child)
                if name is not None:
                    _emit(0, name, line)
        else:  # import_from_statement
            children = node.named_children
            # The first named child is always the module operand (the part after
            # `from`): a `dotted_name` (absolute) or `relative_import` (relative,
            # which the grammar emits even when the tail is empty: `from . import`).
            operand = children[0]
            if operand.type == RELATIVE_IMPORT:
                level, tail = _relative_decomposition(operand)
            elif operand.type == DOTTED_NAME:
                level, tail = 0, _dotted_text(operand)
            else:
                continue  # malformed; nothing to resolve

            if level == 0:
                # `from flask import Flask, request` → one edge to `flask`.
                _emit(0, tail, line)
            elif tail != "":
                # `from .sub import bar` → one edge to `.sub` (the dotted tail),
                # regardless of how many names are imported from it.
                _emit(level, tail, line)
            else:
                # `from . import bar, baz` (empty relative tail) → each imported
                # name is its own submodule candidate of the current package, so
                # emit one edge per name. Wildcards (`from . import *`) name no
                # file and are skipped.
                for child in children[1:]:
                    name = _imported_module_name(child)
                    if name is not None:
                        _emit(level, name, line)

    return list(seen.values())


# ─── resolver (pure data, no DB session) ───────────────────────────────────────


class ImportResolver:
    """Maps import specs to internal ``file_id`` values for one repository.

    Built once from the file rows a repository already has in the ``files`` table
    ``[(id, posix_path)]`` and reused for every file in the repo. Holds only plain
    dicts — no :class:`~sqlalchemy.orm.Session` — so it is cheap to construct and
    trivially testable against synthetic ``(id, path)`` lists.

    Resolution rules:

    * **Absolute** imports (``import a.b`` / ``from a.b import x``) resolve via an
      importable-name map. A file's *importable* dotted name is computed by
      walking up its directory chain until the package root — the shallowest
      ancestor directory that is a package (has ``__init__.py``) whose own parent
      is *not* a package — and joining from there. This transparently handles
      ``src/``-layout repos (``src/flask/helpers.py`` → ``flask.helpers``, not
      ``src.flask.helpers``). A directory is a package iff its ``__init__.py`` is
      present in the file set. Files with no contiguous package chain from root
      to parent get **no** importable name and are omitted from the map — so an
      absolute import of a bare script (``import my_tool``) is not falsely
      resolved; it is flagged external. **Known limitation (SDD §18):** PEP 420
      implicit namespace packages — directories with no ``__init__.py`` that are
      nonetheless importable — are *not* honored by the absolute resolver (e.g.
      flask's ``src/flask/sansio/`` dir has no ``__init__.py``, so
      ``import flask.sansio.app`` resolves to ``None`` and is flagged external).
      The common case still works: flask imports ``sansio`` modules via
      *relative* imports (``from .sansio.scaffold import ...``), which resolve by
      path and do not require ``__init__.py``. Flagging rather than guessing is
      the SDD-mandated posture for unresolvable imports.
    * **Relative** imports (``from .sub import x``, ``from ..pkg.mod import x``)
      resolve by path: the source file's directory, walked up ``level - 1``
      times, joined with the dotted tail, then checked as ``<dir>/<tail>.py`` or
      ``<dir>/<tail>/__init__.py``. Going above the repo root returns ``None``.

    Unresolved imports are not an error here — the caller flags them ``external``
    (SDD §7 step 4): third-party packages, stdlib, and anything we cannot
    confidently map to a repo file all simply return ``None``.
    """

    def __init__(self, files: Iterable[tuple[int, str]]) -> None:
        # POSIX path → file_id (every file row, including every __init__.py).
        # Used for relative resolution (candidate-file lookups) and to detect
        # package directories via their __init__.py membership.
        self._path_to_id: dict[str, int] = {}

        # Dotted importable name → file_id. Only files with a valid contiguous
        # package chain from a package root down to (but excluding) the file
        # appear here; bare scripts and broken chains are intentionally absent.
        self._importable_to_id: dict[str, int] = {}

        # POSIX directory paths that are packages (their __init__.py is present).
        self._package_dirs: set[str] = set()

        rows = [(fid, pth) for fid, pth in files]
        for fid, pth in rows:
            self._path_to_id[pth] = fid
            if pth.endswith("/__init__.py"):
                self._package_dirs.add(pth[: -len("/__init__.py")])
            elif pth == "__init__.py":
                # A repo-root __init__.py would make the root itself a package;
                # treat the root dir as "" for package-root bookkeeping.
                self._package_dirs.add("")

        for fid, pth in rows:
            name = self._importable_name(pth)
            if name is not None and name not in self._importable_to_id:
                # First-wins on name collision (a pathological `pkg.py` next to a
                # `pkg/__init__.py`); real repos don't have both, so ordering
                # only matters for the broken-chain edge case.
                self._importable_to_id[name] = fid

    # -- internals -----------------------------------------------------------

    def _importable_name(self, posix_path: str) -> str | None:
        """The dotted name under which ``posix_path`` could be ``import``-ed, or
        ``None`` if it is not importable via a dotted path (no package chain).

        ``__init__.py`` files map to their package name (``src/flask/__init__.py``
        → ``"flask"``); regular modules map to ``package.module``. Returns
        ``None`` when there is no package root OR the chain from root to the
        file's parent is broken by a non-package directory — those files cannot
        be the target of a confident absolute import and so must not appear in
        the importable map.
        """
        if posix_path.endswith("/__init__.py"):
            dirs = posix_path[: -len("/__init__.py")].split("/")
            is_init = True
        elif posix_path == "__init__.py":
            return None  # repo-root __init__: degenerate, skip
        else:
            dirs = posix_path[: -len(".py")].split("/")  # last element is the leaf module
            is_init = False

        # The package root is the shallowest directory that IS a package while its
        # parent is NOT (treating the repo root as never-a-package). dir_parts[:j]
        # for j in 1..len: dir prefix of length j.
        start: int | None = None
        for j in range(1, len(dirs) + 1):
            here = "/".join(dirs[:j])
            below = "/".join(dirs[: j - 1]) if j > 1 else ""
            here_is_pkg = here in self._package_dirs
            below_is_pkg = (below in self._package_dirs) if j > 1 else False
            if here_is_pkg and not below_is_pkg:
                start = j - 1  # zero-based start index into `dirs`
                break
        if start is None:
            return None  # no package anywhere in the chain → not importable

        # Require a *contiguous* package chain from the root down to the file's
        # parent dir; a non-package directory mid-chain breaks importability.
        # For a regular module the dirs to verify are indices start+1 .. len-1
        # (parent dir is dirs[:len-1]); for __init__, start+1 .. len (all dirs).
        last_dir_index = len(dirs) if is_init else len(dirs) - 1
        for j in range(start + 1, last_dir_index):
            if "/".join(dirs[: j + 1]) not in self._package_dirs:
                return None

        return ".".join(dirs[start:])

    # -- public --------------------------------------------------------------

    def resolve(self, source_path: str, imp: ParsedImport) -> int | None:
        """Resolve ``imp`` (seen from the file at ``source_path``) to a
        ``file_id``, or ``None`` if it cannot be confidently mapped.
        """
        if imp.level == 0:
            return self._importable_to_id.get(imp.target)

        # Root-relative alias (@/ or ~/) in JS/TS/Next.js projects
        if imp.level == -1:
            root_prefix = source_path.split("/", 1)[0] if "/" in source_path else ""
            target_path = imp.target
            if target_path.startswith("@/"):
                target_path = target_path[2:]
            elif target_path.startswith("~/"):
                target_path = target_path[2:]
            cand = f"{root_prefix}/{target_path}" if root_prefix else target_path
            candidates = [
                f"{cand}.ts",
                f"{cand}.tsx",
                f"{cand}.js",
                f"{cand}.jsx",
                f"{cand}/index.ts",
                f"{cand}/index.tsx",
                f"{cand}/index.js",
                f"{cand}/index.jsx",
                f"{cand}.py",
                cand,
            ]
            for c in candidates:
                fid = self._path_to_id.get(c)
                if fid is not None:
                    return fid
            return None

        # Relative: path-space resolution against the source file's own directory.
        src_dir = source_path.rsplit("/", 1)[0] if "/" in source_path else ""
        base = src_dir
        # Each dot beyond the first walks one directory up; `.` (level 1) stays
        # in the source file's own dir, `..` (level 2) its parent, etc.
        for _ in range(imp.level - 1):
            if "/" in base:
                base = base.rsplit("/", 1)[0]
            elif base == "":
                return None  # already at repo root — `..` would leave the tree
            else:
                base = ""  # the final step from a top-level dir up to root ("")

        target_path = imp.target
        if target_path.startswith("./") or target_path.startswith("../"):
            clean_parts = [p for p in target_path.split("/") if p not in {".", ".."}]
            target_path = "/".join(clean_parts)
        elif "." in target_path and not target_path.startswith("/") and not any(target_path.endswith(ext) for ext in [".js", ".ts", ".jsx", ".tsx", ".py"]):
            target_path = target_path.replace(".", "/")

        cand = f"{base}/{target_path}" if base else target_path
        candidates = [
            f"{cand}.py",
            f"{cand}/__init__.py",
            f"{cand}.ts",
            f"{cand}.tsx",
            f"{cand}.js",
            f"{cand}.jsx",
            f"{cand}/index.ts",
            f"{cand}/index.tsx",
            f"{cand}/index.js",
            f"{cand}/index.jsx",
            cand,
        ]
        for c in candidates:
            fid = self._path_to_id.get(c)
            if fid is not None:
                return fid
        return None

    def importable_name(self, posix_path: str) -> str | None:
        """Expose :meth:`_importable_name` for diagnostics / tests."""
        return self._importable_name(posix_path)


# ─── persistence wrapper ──────────────────────────────────────────────────────


def _external_label(imp: ParsedImport) -> str:
    """The ``target_label`` for an unresolved import: the dotted operand with
    leading dots restored for relative imports, so an unresolved ``.sub`` reads
    as ``".sub"`` (relative intent preserved) and a third-party ``werkzeug.routing``
    as ``"werkzeug.routing"``. Absolute imports have no dots to restore."""
    return ("." * imp.level) + imp.target


def index_file_imports(
    file_id: int,
    source_path: str,
    imports: list[ParsedImport],
    resolver: ImportResolver,
    repository_id: int,
    session: Session,
) -> IndexResult:
    """Write one ``Edge`` (``edge_type="imports"``) per import in ``imports``.

    Resolves each import via ``resolver`` (using ``source_path`` for relative
    ones) and returns an :class:`IndexResult` tally. Does **not commit** — the
    caller owns the transaction (mirrors :func:`app.indexing.walker.walk_repository`
    and :func:`app.indexing.parser.index_file_symbols`, SDD §7 step 9).

    Every import produces exactly one edge (SDD §8): resolved→``target_type=
    "file"`` with ``target_id`` set and ``target_label`` NULL, unresolved→
    ``target_type="external"`` with ``target_id`` NULL and ``target_label`` set to
    the dotted module operand. No ``flush`` is needed — ``target_id`` and
    ``repository_id`` always reference already-persisted rows, so there is no
    deferred-FK ordering to satisfy inside the batch.
    """
    added = 0
    resolved = 0
    external = 0
    for imp in imports:
        target_id = resolver.resolve(source_path, imp)
        if target_id is not None:
            session.add(
                Edge(
                    repository_id=repository_id,
                    source_type="file",
                    source_id=file_id,
                    target_type="file",
                    target_id=target_id,
                    target_label=None,
                    edge_type="imports",
                )
            )
            resolved += 1
        else:
            session.add(
                Edge(
                    repository_id=repository_id,
                    source_type="file",
                    source_id=file_id,
                    target_type="external",
                    target_id=None,
                    target_label=_external_label(imp),
                    edge_type="imports",
                )
            )
            external += 1
        added += 1
    return IndexResult(added=added, resolved=resolved, external=external)


def parse_and_index_imports(
    path: Path | str,
    file_id: int,
    source_path: str,
    resolver: ImportResolver,
    repository_id: int,
    session: Session,
) -> IndexResult:
    """Convenience: parse ``path`` then persist its import edges to ``session``.

    Composes :func:`parse_imports` + :func:`index_file_imports` for the future
    ``pipeline.py`` orchestrator. Same transaction-ownership rule applies — this
    adds rows and returns; the caller commits.
    """
    return index_file_imports(
        file_id,
        source_path,
        parse_imports(path),
        resolver,
        repository_id,
        session,
    )
