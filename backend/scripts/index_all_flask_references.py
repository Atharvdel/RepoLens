"""THROWAWAY one-shot batch driver: build the reference graph for the flask repo
(SDD §7 step 5).

Not part of the application — a quick harness that takes the symbols the parser
(``index_all_flask_symbols.py``) persisted into the ``symbols`` table for the
``flask`` repository and runs each through ``find_references`` (ripgrep, whole
word) + ``index_symbol_references`` (resolve hits → ``edges``,
``edge_type="references"``). Safe to delete once the real indexing pipeline
(``pipeline.py`` + an async job / API endpoint) is wired up; that layer will drive
the reference index the same way this script does.

Same pattern as ``scripts/index_all_flask_imports.py``: targets the existing
``flask`` repo row, is idempotent (clears that repo's existing ``references``
edges first so a re-run re-inserts cleanly rather than accumulating
duplicates), commits once at the end, then reports counts + per-symbol failures.

Per-symbol failures are isolated with SQLAlchemy nested transactions
(SAVEPOINTs) so one ripgrep error / rogue name rolls back only its own edges and
the remaining symbols still get indexed in the same outer commit.

The ``path -> file_id`` map (the trivial analog of the import graph's
:class:`ImportResolver`) is built once from all of the repo's ``files`` rows
before the per-symbol loop — resolution is a dict lookup, but it needs every
indexed file's id up front so a hit ripgrep surfaces in *any* repo file resolves
to a target (or is correctly counted ``dropped`` if that file wasn't indexed).

"Bounded noise": ``DEFAULT_CAP`` (50) is passed to ``find_references`` so a very
common name (``run`` / ``get`` / ``__init__``) returns at most ``cap`` hits via
ripgrep's ``--max-count``. A symbol is reported as **hit-the-cap** when
``len(hits) == cap`` — a strict lower bound (a symbol whose own binding line
falls within the first ``cap`` ripgrep matches is excluded *before* the cap, so
it surfaces ``cap - 1`` hits and reads as not-saturated; documented here rather
than over-fitted). The report also counts symbols that produced *zero* hits.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/index_all_flask_references.py
"""
import sys
from pathlib import Path

import sqlalchemy as sa

# Allow running as `python scripts/index_all_flask_references.py` from backend/
# without a pre-set PYTHONPATH (mirrors tests/conftest.py and alembic/env.py so
# `import app.*` always resolves).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import SessionLocal  # noqa: E402
from app.indexing.reference_index import (  # noqa: E402
    DEFAULT_CAP,
    ReferenceIndexResult,
    find_references,
    index_symbol_references,
    ripgrep_available,
)
from app.models import Edge, File, Repository, Symbol  # noqa: E402

REPO_NAME = "flask"
REPO_PATH = r"C:\Users\Atharv Sharma\Desktop\Work\flask"


def main() -> None:
    # Windows consoles are often cp1252/cp437; force UTF-8 so non-ASCII in paths
    # or line text survives the report. Best-effort: ignore if the stream can't
    # be reconfigured (redirected output / older runtime).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

    if not ripgrep_available():
        print(
            "ERROR: ripgrep binary `rg` not found on PATH — the reference index "
            "shells out to it (SDD §0 stack). Install ripgrep or add it to PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)

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
        path_to_file_id = {f.path: f.id for f in files}

        # All symbols the parser persisted for this repo, joined to their file so
        # each carries (name, definition_path, definition_line=line_start) for the
        # ripgrep call. Ordered by path then line_start for a stable, file-grouped
        # run (ripgrep re-reads the same files repeatedly across symbols; locality
        # in the OS file cache is a minor incidental win, not a design goal).
        symbol_rows = (
            session.execute(
                sa.select(Symbol, File)
                .join(File, File.id == Symbol.file_id)
                .where(File.repository_id == repo.id)
                .order_by(File.path, Symbol.line_start)
            )
            .all()
        )
        if not symbol_rows:
            print(
                f"ERROR: repo {REPO_NAME!r} (id={repo.id}) has no symbol rows — run "
                "scripts/index_all_flask_symbols.py first.",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Idempotency: drop references edges already attached to this repo so a
        # re-run re-inserts cleanly rather than doubling the reference graph
        # (same idea as the imports driver clearing `imports` edges; one relation
        # over). Scoping by repository_id + edge_type is exact and independent of
        # the particular symbol set being re-indexed.
        cleared = session.execute(
            sa.delete(Edge).where(
                Edge.repository_id == repo.id,
                Edge.edge_type == "references",
            )
        ).rowcount
        session.commit()
        if cleared:
            print(
                f"cleared {cleared} previously-indexed reference edges for repo id={repo.id}"
            )

        ok = 0
        total = ReferenceIndexResult(added=0, dropped=0)
        saturated = 0   # symbols whose hit count reached `cap` (capped-and-bounded)
        zero = 0        # symbols that produced no references at all
        failures: list[tuple[str, str, str]] = []  # (name, posix-rel path, repr(err))

        # No per-symbol flush needed: target_id (a previously-persisted file),
        # repository_id (a committed repo), and source_id (a committed symbol) all
        # reference rows with PKs, so a savepoint per symbol only guards rollback,
        # never FK assignment order.
        for sym, file in symbol_rows:
            try:
                # Savepoint isolates this symbol end-to-end: a ripgrep error
                # (RuntimeError on exit code >= 2) OR a flush error rolls back
                # only this symbol's edges, leaving the outer transaction usable
                # for the next symbol. Mirrors the imports driver's posture.
                with session.begin_nested():
                    hits = find_references(
                        sym.name, REPO_PATH,
                        definition_path=file.path,
                        definition_line=sym.line_start,
                        cap=DEFAULT_CAP,
                    )
                    result = index_symbol_references(
                        sym.id, hits, path_to_file_id, repo.id, session
                    )
                total = total + result
                ok += 1
                if len(hits) == 0:
                    zero += 1
                if len(hits) == DEFAULT_CAP:
                    saturated += 1
            except Exception as exc:  # noqa: BLE001 — surface every failure type
                failures.append((sym.name, file.path, repr(exc)))

        session.commit()

        # Ground-truth: count edges re-read through a fresh query (a real DB
        # round-trip, not the in-session accumulation) — mirrors the imports
        # driver's persisted-vs-created check.
        persisted = session.scalar(
            sa.select(sa.func.count())
            .select_from(Edge)
            .where(
                Edge.repository_id == repo.id,
                Edge.edge_type == "references",
            )
        )

    # --- Report ----------------------------------------------------------
    n_symbols = len(symbol_rows)
    print()
    print(f"repository: name={REPO_NAME!r} id={repo.id} symbols_in_table={n_symbols}")
    print(f"processed successfully       : {ok}")
    print(f"failed                       : {len(failures)}")
    print(f"reference edges created       : {total.added}")
    print(f"  hits dropped (unindexed file): {total.dropped}")
    print(f"  symbols with zero references : {zero}")
    print(f"  symbols that hit the cap    : {saturated}  (cap={DEFAULT_CAP})")
    if persisted is not None:
        mismatch = "" if persisted == total.added else "  (MISMATCH vs created!)"
        print(f"reference edges in DB         : {persisted}{mismatch}")
    if failures:
        print()
        print("failures:")
        for name, path, err in failures:
            print(f"  {name!r} @ {path}: {err}")


if __name__ == "__main__":
    main()
