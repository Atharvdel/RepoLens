"""Tests for the import-graph stage (SDD §7 step 4).

Three layers, mirroring :mod:`tests.test_parser`'s pure-parse / persist split:

* ``test_parse_imports_*`` and ``test_import_resolver_*`` — *pure, no DB*.
  Exercise :func:`parse_file`-style pure extraction (:func:`parse_imports`) and
  the :class:`ImportResolver` against crafted synthetic inputs, asserting the
  extraction + resolution rules. These run anywhere — no Postgres.
* ``test_index_file_imports_*`` — *live-Postgres round-trip* (same container the
  smoke test uses). Two tests, one per the build instructions: a known **local**
  import resolving to the correct ``file_id``, and an **external** import
  correctly flagged (with a genuinely-local import alongside it as a guard so
  the test cannot pass vacuously by flagging *everything* external).

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_import_graph.py -v
"""
from collections import Counter

import sqlalchemy as sa

from app.db import SessionLocal
from app.indexing import (
    ImportResolver,
    ParsedImport,
    index_file_imports,
    parse_imports,
)
from app.models import Edge, File, Repository

# One source file exercising every extraction decision at once. *Which* edges
# surface (and how many) is what matters — line numbers are not asserted.
SRC = '''"""Module docstring — not an import."""

import os
import os.path as osspath
import numpy as np, a.b.c

from flask import Flask, request
from flask.helpers import url_for
from a.b import *

from . import sibling, other_sib
from .subpkg import deep
from ..parent_pkg.mod import thing
from .helpers import x
from .helpers import y  # same dependency as the line above -> deduped

import sys
try:
    import c_extension
except ImportError:
    c_extension = None

if TYPE_CHECKING:
    from .models import TypedModel


def f():
    """A function with a nested import — still a real dependency."""
    import lazy_dep
    return lazy_dep
'''


def _pairs(imports: list[ParsedImport]) -> list[tuple[int, str]]:
    """The ``(level, target)`` of each parsed import, in dedup order."""
    return [(imp.level, imp.target) for imp in imports]


def test_parse_imports_extracts_every_form(tmp_path):
    """Every import shape produces the right ``(level, target)`` dependencies,
    deduped: two names from one ``from``-module are ONE edge; two names in a
    bare ``from . import`` (empty relative tail) are TWO edges (each name is its
    own submodule candidate); a ``*`` names the module, not a name; aliases are
    stripped; and conditional / function-nested imports are captured (the whole
    tree is walked, unlike the symbol parser's top-level-only scope)."""
    path = tmp_path / "imports.py"
    path.write_text(SRC, encoding="utf-8")

    imports = parse_imports(path)

    # ── the full deduped dependency set, as (level, target) ──────────────────
    expected = {
        # absolute (import … / from <abs> import …)
        (0, "os"),
        (0, "os.path"),
        (0, "numpy"),
        (0, "a.b.c"),
        (0, "flask"),
        (0, "flask.helpers"),
        (0, "a.b"),
        (0, "sys"),
        (0, "c_extension"),  # inside try/except ImportError
        (0, "lazy_dep"),     # nested inside a function body
        # relative, level 1
        (1, "sibling"),
        (1, "other_sib"),
        (1, "subpkg"),
        (1, "helpers"),     # two statements -> one deduped dependency
        (1, "models"),       # inside `if TYPE_CHECKING:`
        # relative, level 2
        (2, "parent_pkg.mod"),
    }
    assert set(_pairs(imports)) == expected, set(_pairs(imports)) ^ expected

    # ── structural contrasts the set alone can't prove ───────────────────────
    counts = Counter(_pairs(imports))

    # `from flask import Flask, request` -> ONE edge to the module `flask`, not
    # one per imported name. (Imported names are a reference-graph concern.)
    assert counts[(0, "flask")] == 1
    # `from a.b import *` -> the module `a.b`, never a "*" target.
    assert (0, "*") not in counts
    assert counts[(0, "a.b")] == 1

    # `from . import sibling, other_sib` (empty relative tail) -> TWO edges, one
    # per name, because each name is a submodule candidate of the current package.
    assert counts[(1, "sibling")] == 1
    assert counts[(1, "other_sib")] == 1

    # `from .helpers import x` and `from .helpers import y` -> ONE deduped edge.
    assert counts[(1, "helpers")] == 1

    # The import depth the symbol parser deliberately skips is captured here:
    # a try/except-guarded import and a function-local import both surface.
    assert (0, "c_extension") in set(_pairs(imports))
    assert (0, "lazy_dep") in set(_pairs(imports))
    assert (1, "models") in set(_pairs(imports))  # if TYPE_CHECKING:


def test_parse_imports_is_pure(tmp_path):
    """``parse_imports`` touches no DB — it reads bytes and returns data, so it
    runs without a live Postgres (the pure-parse layer the build asked for)."""
    path = tmp_path / "pure.py"
    path.write_text("import collections\nfrom . import sibling\n", encoding="utf-8")
    # No session, no engine import side effects — just a list back.
    imports = parse_imports(path)
    assert _pairs(imports) == [(0, "collections"), (1, "sibling")]


def _make_resolver(
    files: list[tuple[int, str]],
) -> ImportResolver:
    """Build an :class:`ImportResolver` from synthetic ``(id, posix_path)`` rows
    — the no-DB construction path, mirroring how the batch driver builds it from
    a queried file list."""
    return ImportResolver(files)


def test_import_resolver_importable_name_handles_src_layout():
    """A ``src/``-layout repo maps to the *package-internal* dotted name, not
    ``src.flask.…``. The package root is the shallowest package dir whose parent
    isn't a package, so ``src/flask/helpers.py`` -> ``flask.helpers``. Files with
    no contiguous package chain (bare scripts, ``tests/``) get ``None`` and so
    cannot be falsely resolved by an absolute import."""
    files = [
        (1, "src/flask/__init__.py"),
        (2, "src/flask/app.py"),
        (3, "src/flask/helpers.py"),
        (4, "src/flask/json/__init__.py"),
        (5, "src/flask/json/provider.py"),
        (6, "src/other/__init__.py"),  # a second package root -> "other"
        (7, "src/other/mod.py"),
        (8, "tests/test_app.py"),     # no package above -> not importable
        (9, "setup.py"),              # repo-root bare script -> not importable
    ]
    r = _make_resolver(files)

    assert r.importable_name("src/flask/__init__.py") == "flask"
    assert r.importable_name("src/flask/app.py") == "flask.app"
    assert r.importable_name("src/flask/helpers.py") == "flask.helpers"
    assert r.importable_name("src/flask/json/__init__.py") == "flask.json"
    assert r.importable_name("src/flask/json/provider.py") == "flask.json.provider"
    assert r.importable_name("src/other/__init__.py") == "other"
    assert r.importable_name("src/other/mod.py") == "other.mod"
    assert r.importable_name("tests/test_app.py") is None
    assert r.importable_name("setup.py") is None

    # Absolute resolution rides on those importable names.
    assert r.resolve("src/flask/app.py", ParsedImport("flask", 0, 1)) == 1
    assert r.resolve("src/flask/app.py", ParsedImport("flask.helpers", 0, 1)) == 3
    assert r.resolve("src/flask/app.py", ParsedImport("flask.json.provider", 0, 1)) == 5
    # A name with no repo file -> None (third-party / stdlib / unknown).
    assert r.resolve("src/flask/app.py", ParsedImport("werkzeug", 0, 1)) is None
    # A bare-script name is NOT in the importable map -> not falsely resolved.
    assert r.resolve("src/flask/app.py", ParsedImport("setup", 0, 1)) is None


def test_import_resolver_relative_levels_and_boundary():
    """Relative imports resolve by path against the importer's own directory:
    ``.`` (level 1) stays in the dir, ``..`` (level 2) the parent, ``...``
    (level 3) the grandparent; going above the repo root returns ``None``, and a
    tail naming a package resolves to its ``__init__.py``."""
    files = [
        (1, "pkg/__init__.py"),
        (2, "pkg/a.py"),
        (3, "pkg/sub/__init__.py"),
        (4, "pkg/sub/b.py"),
        (5, "pkg/sub/deep/__init__.py"),  # a package dir (not resolved to below)
        (6, "pkg/sub/deep/c.py"),
        (7, "pkg/sub/deep/d.py"),
    ]
    r = _make_resolver(files)

    # Importer lives at pkg/sub/deep/c.py; its directory is pkg/sub/deep.
    src = "pkg/sub/deep/c.py"

    # level 1 (`.`) -> sibling in the same dir.
    assert r.resolve(src, ParsedImport("d", 1, 1)) == 7
    # level 2 (`..`) -> up one dir (pkg/sub), tail `b` -> pkg/sub/b.py.
    assert r.resolve(src, ParsedImport("b", 2, 1)) == 4
    # level 3 (`...`) -> up two dirs (pkg), tail `a` -> pkg/a.py.
    assert r.resolve(src, ParsedImport("a", 3, 1)) == 2

    # A tail naming a *package* resolves to its __init__.py, not pkg/sub/deep.py.
    # From pkg/a.py, `from . import sub` -> pkg/sub/__init__.py (id 3).
    assert r.resolve("pkg/a.py", ParsedImport("sub", 1, 1)) == 3

    # Going above the repo root is None, not a crash.
    # From pkg/a.py (dir `pkg`), `..` would step above root -> None.
    assert r.resolve("pkg/a.py", ParsedImport("anything", 2, 1)) is None
    # A target that doesn't exist is None (not an error).
    assert r.resolve(src, ParsedImport("missing", 1, 1)) is None


def _setup_package_repo(session) -> tuple[int, dict[str, int]]:
    """Persist a small ``proj`` package the importer under test can depend on.

    Returns ``(repo_id, {posix_path: file_id})``. Layout::

        proj/__init__.py        (root package)
        proj/util.py            (resolvable as proj.util)
        proj/sub/__init__.py    (resolvable as proj.sub)
        proj/sub/thing.py       (resolvable as proj.sub.thing)
        proj/importer.py        (the importer-under-test row, for source_id)

    All files get a placeholder ``loc``; only ``path`` + ``id`` matter to the
    import graph. The importer's *contents* live on disk (written by each test),
    decoupled from this row — :func:`parse_imports` reads bytes, resolution uses
    the DB path.
    """
    repo = Repository(url_or_path="file:///test/imports", name="test/imports",
                      status="indexing")
    session.add(repo)
    session.commit()
    session.refresh(repo)

    paths = [
        "proj/__init__.py",
        "proj/util.py",
        "proj/sub/__init__.py",
        "proj/sub/thing.py",
        "proj/importer.py",
    ]
    rows = [File(repository_id=repo.id, path=p, language="python", loc=10)
            for p in paths]
    session.add_all(rows)
    # One commit + flush assigns all PKs in one shot (mirrors the parser test's
    # single-commit for its lone file; here generalized to a batch).
    session.flush()
    session.commit()
    ids: dict[str, int] = {f.path: f.id for f in rows}
    return repo.id, ids


def _cleanup(session, repo_id: int | None) -> None:
    """Defensive cleanup: edges -> files -> repo (mirrors test_parser's
    symbols -> file -> repo teardown so a failure never leaves stray rows)."""
    if repo_id is not None:
        session.execute(sa.delete(Edge).where(Edge.repository_id == repo_id))
        session.execute(sa.delete(File).where(File.repository_id == repo_id))
        session.execute(sa.delete(Repository).where(Repository.id == repo_id))
        session.commit()


def test_index_file_imports_resolves_local_import_to_file_id(tmp_path):
    """Live-DB round-trip: a known local import — absolute and relative —
    resolves to the correct ``file_id`` and persists as an edge with
    ``target_type="file"``, ``target_id`` set, and ``target_label`` NULL."""
    src = '''import proj.util
from proj.sub import thing
from .util import helper
from .sub.thing import Data
'''
    path = tmp_path / "importer.py"
    path.write_text(src, encoding="utf-8")

    repo_id = None
    try:
        with SessionLocal() as session:
            repo_id, ids = _setup_package_repo(session)
            importer_id = ids["proj/importer.py"]
            resolver = ImportResolver(
                [(fid, pth) for pth, fid in ids.items()]
            )

            parsed = parse_imports(path)
            result = index_file_imports(
                importer_id, "proj/importer.py", parsed,
                resolver, repo_id, session,
            )
            session.commit()

        # Every import here is local, so all edges resolve.
        assert result.added == 4, result
        assert result.resolved == 4, result
        assert result.external == 0, result

        # Read back through a fresh session (real DB hit, no identity map).
        with SessionLocal() as session:
            rows = (
                session.execute(
                    sa.select(Edge)
                    .where(Edge.source_id == importer_id,
                           Edge.source_type == "file",
                           Edge.edge_type == "imports")
                )
                .scalars()
                .all()
            )

            # All four are internal edges: file -> file, no label.
            assert len(rows) == 4, [r.target_id for r in rows]
            assert all(r.target_type == "file" for r in rows), rows
            assert all(r.target_label is None for r in rows), rows
            assert all(r.target_id is not None for r in rows), rows

            # ── the correctness core: each resolves to exactly the right file.
            #
            # `import proj.util`            -> proj/util.py        (ids['proj/util.py'])
            # `from proj.sub import thing`  -> proj/sub/__init__.py (the *package*
            #                                   named by the `from` operand —
            #                                   ids['proj/sub/__init__.py'])
            # `from .util import helper`    -> proj/util.py        (relative sibling)
            # `from .sub.thing import Data` -> proj/sub/thing.py
            #
            # Note `import proj.util` (absolute) and `from .util import helper`
            # (relative) are two distinct import statements expressing the same
            # dependency, so they produce TWO edges to proj/util.py — proving
            # dedup is by (level, target) within a statement, not by file_id.
            target_counts = Counter(r.target_id for r in rows)
            util_id = ids["proj/util.py"]
            subpkg_id = ids["proj/sub/__init__.py"]
            thing_id = ids["proj/sub/thing.py"]
            assert target_counts == {
                util_id: 2,     # import proj.util  +  from .util import helper
                subpkg_id: 1,   # from proj.sub import thing  -> the package __init__
                thing_id: 1,    # from .sub.thing import Data -> the module file
            }, target_counts
    finally:
        with SessionLocal() as session:
            _cleanup(session, repo_id)


def test_index_file_imports_flags_external_import(tmp_path):
    """Live-DB round-trip: third-party, stdlib, and unresolved-relative imports
    are flagged ``target_type="external"`` with ``target_label`` set to the
    dotted module operand (relative dots preserved). A genuinely-local import is
    included alongside as a guard so the test cannot pass by flagging everything
    external — it must resolve to a real ``file_id`` while the rest stay external."""
    src = '''import werkzeug.routing
from os import path
from .ghost import missing
from . import phantom
import proj.util  # local — must resolve, proving the resolver isn't just
                  # flagging everything external.
'''
    path = tmp_path / "importer.py"
    path.write_text(src, encoding="utf-8")

    repo_id = None
    try:
        with SessionLocal() as session:
            repo_id, ids = _setup_package_repo(session)
            importer_id = ids["proj/importer.py"]
            resolver = ImportResolver(
                [(fid, pth) for pth, fid in ids.items()]
            )

            parsed = parse_imports(path)
            result = index_file_imports(
                importer_id, "proj/importer.py", parsed,
                resolver, repo_id, session,
            )
            session.commit()

        # 5 imports: 4 external + the 1 local guard.
        assert result.added == 5, result
        assert result.external == 4, result
        assert result.resolved == 1, result

        with SessionLocal() as session:
            rows = (
                session.execute(
                    sa.select(Edge)
                    .where(Edge.source_id == importer_id,
                           Edge.source_type == "file",
                           Edge.edge_type == "imports")
                )
                .scalars()
                .all()
            )

            externals = [r for r in rows if r.target_type == "external"]
            internals = [r for r in rows if r.target_type == "file"]

            # ── the four external flags carry the dotted operand as the label,
            # with relative dots preserved for the unresolved relative imports.
            assert {r.target_label for r in externals} == {
                "werkzeug.routing",  # third-party, absolute
                "os",                # stdlib, absolute
                ".ghost",            # relative, names a module (from .ghost import …)
                ".phantom",          # relative, empty tail (from . import phantom)
            }, {r.target_label for r in externals}
            assert all(r.target_id is None for r in externals), externals

            # ── the guard: exactly one edge resolved to the local util file.
            assert len(internals) == 1, internals
            assert internals[0].target_id == ids["proj/util.py"]
            assert internals[0].target_label is None
    finally:
        with SessionLocal() as session:
            _cleanup(session, repo_id)
