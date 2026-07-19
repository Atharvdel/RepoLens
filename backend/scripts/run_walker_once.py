"""THROWAWAY one-shot driver for the RepoLens file-walker (SDD §7 steps 1–2).

Not part of the application — a quick harness to exercise the walker against a
real local repo and report how many files were found and inserted into the
``files`` table. Safe to delete once the real indexing pipeline
(``pipeline.py`` + an async job / API endpoint) is wired up; that layer will
call ``walk_repository`` the same way this script does.

Idempotent: reuses an existing ``repositories`` row named ``flask`` and clears
its files first, so re-runs re-insert cleanly instead of tripping the
UNIQUE(repository_id, path) constraint.

Run from the ``backend/`` directory::

    PYTHONPATH=. python scripts/run_walker_once.py
"""
import sys
from pathlib import Path

import sqlalchemy as sa

# Allow running as `python scripts/run_walker_once.py` from backend/ without a
# pre-set PYTHONPATH (mirrors the sys.path bootstrap in tests/conftest.py and
# alembic/env.py so `import app.*` always resolves).
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db import SessionLocal  # noqa: E402
from app.indexing.walker import walk_repository  # noqa: E402
from app.models import File, Repository  # noqa: E402

REPO_NAME = "flask"
REPO_PATH = r"C:\Users\Atharv Sharma\Desktop\Work\flask"


def get_or_create_repo(session) -> Repository:
    """Reuse an existing test repo row (so re-runs don't pile up) else create
    one in the ``indexing`` state. Its files are cleared separately by the
    caller right before the walk."""
    repo = session.execute(
        sa.select(Repository).where(Repository.name == REPO_NAME)
    ).scalar_one_or_none()
    if repo is None:
        repo = Repository(url_or_path=REPO_PATH, name=REPO_NAME, status="indexing")
        session.add(repo)
        session.commit()
        session.refresh(repo)
    return repo


def main() -> None:
    root = Path(REPO_PATH)
    if not root.is_dir():
        print(f"ERROR: repository path is not a directory: {REPO_PATH}", file=sys.stderr)
        raise SystemExit(1)

    with SessionLocal() as session:
        repo = get_or_create_repo(session)

        # Idempotency: drop any files already attached to this repo so a re-run
        # re-inserts cleanly rather than violating UNIQUE(repository_id, path).
        deleted = session.execute(
            sa.delete(File).where(File.repository_id == repo.id)
        ).rowcount
        session.commit()
        if deleted:
            print(f"cleared {deleted} previously-indexed file rows for repo id={repo.id}")

        found = walk_repository(root, repo.id, session)
        session.commit()

        # Re-read through a fresh query to confirm the rows actually persisted
        # (a genuine DB round-trip, not an identity-map shortcut).
        persisted = session.scalar(
            sa.select(sa.func.count())
            .select_from(File)
            .where(File.repository_id == repo.id)
        )
        sample = (
            session.execute(
                sa.select(File.path)
                .where(File.repository_id == repo.id)
                .order_by(File.path)
                .limit(5)
            )
            .scalars()
            .all()
        )

    print(f"repository: name={REPO_NAME!r} id={repo.id} path={REPO_PATH}")
    print(f"walker found & added (in-session): {found}")
    print(f"files persisted in DB            : {persisted}")
    print("first 5 paths:")
    for path in sample:
        print(f"  {path}")


if __name__ == "__main__":
    main()
