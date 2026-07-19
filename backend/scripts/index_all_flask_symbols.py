"""THROWAWAY one-shot batch driver: parse + symbol-index the flask repo (SDD §7 step 3).

Not part of the application — a quick harness that takes the rows the
file-walker (``run_walker_once.py``) already persisted into the ``files`` table for
the ``flask`` repository and runs each through ``parse_and_index_file``
(tree-sitter parse → ``symbols`` rows). Safe to delete once the real indexing
pipeline (``pipeline.py`` + an async job / API endpoint) is wired up; that layer
will drive the parser the same way this script does.

Same pattern as ``scripts/run_walker_once.py``: targets the existing ``flask``
repo row, is idempotent (clears that repo's existing ``symbols`` rows first so a
re-run re-inserts cleanly rather than accumulating duplicates), commits once at
the end, then reports counts + per-file failures.

Per-file failures are isolated with SQLAlchemy nested transactions (SAVEPOINTs)
so one unreadable / parse-broken file rolls back only its own symbols and the
remaining files still get indexed in the same outer commit — a plain per-file
``try/except`` without savepoints would leave the session wedged after any DB
error and risk rolling back the whole batch.

The ``files`` table stores paths POSIX-relative to the repo root (see
``walker.walk_repository``); this script re-roots each one under ``REPO_PATH`` to
give ``parse_file`` an absolute path it can read from disk.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/index_all_flask_symbols.py
"""
import sys
from pathlib import Path

import sqlalchemy as sa

# Allow running as `python scripts/index_all_flask_symbols.py` from backend/
# without a pre-set PYTHONPATH (mirrors tests/conftest.py and alembic/env.py so
# `import app.*` always resolves).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import SessionLocal  # noqa: E402
from app.indexing.parser import parse_and_index_file  # noqa: E402
from app.models import File, Repository, Symbol  # noqa: E402

REPO_NAME = "flask"
REPO_PATH = r"C:\Users\Atharv Sharma\Desktop\Work\flask"


def main() -> None:
    # Windows consoles are often cp1252/cp437; force UTF-8 so non-ASCII in
    # docstrings or file paths survives the report. Best-effort: ignore if the
    # stream can't be reconfigured (redirected output / older runtime).
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

        # Idempotency: drop symbols already attached to this repo's files so a
        # re-run re-inserts cleanly rather than doubling the symbol index (same
        # idea as run_walker_once.py clearing File rows, one level down).
        cleared = session.execute(
            sa.delete(Symbol).where(
                Symbol.file_id.in_([f.id for f in files])
            )
        ).rowcount
        session.commit()
        if cleared:
            print(f"cleared {cleared} previously-indexed symbol rows for repo id={repo.id}")

        ok = 0
        inserted_total = 0
        failures: list[tuple[str, str]] = []  # (posix-rel path, repr of the error)

        for f in files:
            # f.path is POSIX-relative to the repo root; re-root it so parse_file
            # gets an absolute path it can read_bytes() from disk.
            abs_path = Path(REPO_PATH) / f.path
            try:
                # Savepoint isolates this file: a flush error rolls back only
                # this file's rows, leaving the outer transaction usable for the
                # next file. released on a clean exit, rolled back on exception
                # (the context manager re-raises, caught below).
                with session.begin_nested():
                    inserted_total += parse_and_index_file(abs_path, f.id, session)
                ok += 1
            except Exception as exc:  # noqa: BLE001 — surface every failure type
                failures.append((f.path, repr(exc)))

        session.commit()

        # Ground-truth: count symbols re-read through a fresh query (a real DB
        # round-trip, not the in-session accumulation) to confirm what actually
        # persisted — mirrors run_walker_once.py's persisted-vs-found check.
        persisted = session.scalar(
            sa.select(sa.func.count())
            .select_from(Symbol)
            .join(File, Symbol.file_id == File.id)
            .where(File.repository_id == repo.id)
        )

    # --- Report ----------------------------------------------------------
    total = len(files)
    print()
    print(f"repository: name={REPO_NAME!r} id={repo.id} files_in_table={total}")
    print(f"parsed successfully : {ok}")
    print(f"failed              : {len(failures)}")
    print(f"symbols inserted    : {inserted_total}")
    if persisted is not None:
        mismatch = "" if persisted == inserted_total else "  (MISMATCH vs inserted!)"
        print(f"symbols in DB       : {persisted}{mismatch}")
    if failures:
        print()
        print("failures:")
        for path, err in failures:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
