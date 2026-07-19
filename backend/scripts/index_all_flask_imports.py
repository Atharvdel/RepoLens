"""THROWAWAY one-shot batch driver: build the import graph for the flask repo
(SDD §7 step 4).

Not part of the application — a quick harness that takes the rows the
file-walker (``run_walker_once.py``) persisted into the ``files`` table for the
``flask`` repository and runs each through ``parse_imports`` (tree-sitter) +
``index_file_imports`` (resolve → ``edges``, ``edge_type="imports"``). Safe to
delete once the real indexing pipeline (``pipeline.py`` + an async job / API
endpoint) is wired up; that layer will drive the import graph the same way this
script does.

Same pattern as ``scripts/index_all_flask_symbols.py``: targets the existing
``flask`` repo row, is idempotent (clears that repo's existing ``imports`` edges
first so a re-run re-inserts cleanly rather than accumulating duplicates),
commits once at the end, then reports counts + per-file failures.

Per-file failures are isolated with SQLAlchemy nested transactions (SAVEPOINTs)
so one unreadable / unparseable file rolls back only its own edges and the
remaining files still get indexed in the same outer commit.

The :class:`ImportResolver` is built once (from all of the repo's ``files``
rows) before the per-file loop — resolution is cross-file, so the whole-repo
importable-name map and package-dir set are needed up front; building it once
keeps that work O(files) total rather than once per file. Each query loop
iteration re-roots the POSIX-rel ``files.path`` under ``REPO_PATH`` so
``parse_imports`` gets an absolute path it can read from disk.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/index_all_flask_imports.py
"""
import sys
from pathlib import Path

import sqlalchemy as sa

# Allow running as `python scripts/index_all_flask_imports.py` from backend/
# without a pre-set PYTHONPATH (mirrors tests/conftest.py and alembic/env.py so
# `import app.*` always resolves).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import SessionLocal  # noqa: E402
from app.indexing.import_graph import (  # noqa: E402
    ImportResolver,
    IndexResult,
    index_file_imports,
    parse_imports,
)
from app.models import Edge, File, Repository  # noqa: E402

REPO_NAME = "flask"
REPO_PATH = r"C:\Users\Atharv Sharma\Desktop\Work\flask"


def main() -> None:
    # Windows consoles are often cp1252/cp437; force UTF-8 so non-ASCII in paths
    # or labels survives the report. Best-effort: ignore if the stream can't be
    # reconfigured (redirected output / older runtime).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    root = Path(REPO_PATH)
    if not root.is_dir():
        print(f"ERROR: repository path is not a directory: {REPO_PATH}", file=sys.stderr)
        raise SystemExit(1)

    with SessionLocal() as session:
        repo = session.execute(
            sa.select(Repository).where(Repository.name == REPO_NAME)
        ).scalar_one_or_none()
        if repo is None:
            print(
                f"ERROR: no repository row named {REPO_NAME!r} in DB — run "
                "scripts/run_walker_once.py first to populate the files table.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        files = (
            session.execute(
                sa.select(File)
                .where(File.repository_id == repo.id)
                .order_by(File.path)
            )
            .scalars()
            .all()
        )
        if not files:
            print(
                f"ERROR: repo {REPO_NAME!r} (id={repo.id}) has no file rows — run "
                "scripts/run_walker_once.py first.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Idempotency: drop imports edges already attached to this repo's files
        # so a re-run re-inserts cleanly rather than doubling the import graph
        # (same idea as index_all_flask_symbols.py clearing symbol rows, one
        # relation down). Scoping by repository_id is exact and independent of
        # any particular file set.
        cleared = session.execute(
            sa.delete(Edge).where(
                Edge.repository_id == repo.id,
                Edge.edge_type == "imports",
            )
        ).rowcount
        session.commit()
        if cleared:
            print(
                f"cleared {cleared} previously-indexed import edges for repo id={repo.id}"
            )

        # Build the resolver ONCE from all the repo's file rows — resolution is
        # cross-file (a file's importable name depends on the whole repo's
        # package layout), so the importable-name map and package-dir set must
        # exist before any file is indexed. One build keeps it O(files) total.
        resolver = ImportResolver(
            (f.id, f.path) for f in files
        )

        ok = 0
        total = IndexResult(added=0, resolved=0, external=0)
        # No per-file flush needed: target_id (a previously-persisted file) and
        # repository_id (a committed repo) always reference rows with PKs, so a
        # savepoint per file only guards rollback, never FK assignment order.
        failures: list[tuple[str, str]] = []  # (posix-rel path, repr of the error)

        for f in files:
            # f.path is POSIX-relative to the repo root; re-root it so
            # parse_imports gets an absolute path it can read_bytes() from disk.
            abs_path = Path(REPO_PATH) / f.path
            try:
                # Savepoint isolates this file end-to-end: a parse error (file
                # vanished/unreadable since the walk) OR a flush error rolls back
                # only this file's work, leaving the outer transaction usable for
                # the next file. Mirrors index_all_flask_symbols.py's posture.
                with session.begin_nested():
                    imports = parse_imports(abs_path)
                    result = index_file_imports(
                        f.id, f.path, imports, resolver, repo.id, session
                    )
                total = total + result
                ok += 1
            except Exception as exc:  # noqa: BLE001 — surface every failure type
                failures.append((f.path, repr(exc)))

        session.commit()

        # Ground-truth: count edges re-read through a fresh query (a real DB
        # round-trip, not the in-session accumulation) — mirrors
        # index_all_flask_symbols.py's persisted-vs-inserted check.
        persisted = session.scalar(
            sa.select(sa.func.count())
            .select_from(Edge)
            .where(
                Edge.repository_id == repo.id,
                Edge.edge_type == "imports",
            )
        )

    # --- Report ----------------------------------------------------------
    n_files = len(files)
    print()
    print(f"repository: name={REPO_NAME!r} id={repo.id} files_in_table={n_files}")
    print(f"parsed successfully          : {ok}")
    print(f"failed                      : {len(failures)}")
    print(f"import edges created        : {total.added}")
    print(f"  resolved to internal files: {total.resolved}")
    print(f"  flagged external          : {total.external}")
    if persisted is not None:
        mismatch = "" if persisted == total.added else "  (MISMATCH vs created!)"
        print(f"import edges in DB          : {persisted}{mismatch}")
    if failures:
        print()
        print("failures:")
        for path, err in failures:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
