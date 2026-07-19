"""Fast smoke test for the RepoLens schema (SDD §11).

Two integration checks against the live Postgres instance (DATABASE_URL /
the ``repolens-postgres`` container):

1. ``test_all_ten_model_tables_exist_in_db`` — confirms the ORM model
   metadata declares exactly the 10 SDD §11 tables and that every one of
   them is present in the database (i.e. ``alembic upgrade head`` ran and
   the models and DB agree).
2. ``test_repositories_round_trip`` — a create → read → delete round-trip
   on the ``repositories`` table, committing at each step and reading back
   through fresh sessions so it's a genuine DB round-trip, not an
   identity-map shortcut.

Scope: environment + schema only. This deliberately does NOT touch the
indexing pipeline, agents, tools, or the frontend.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_smoke_db.py -v
"""
import sqlalchemy as sa
from sqlalchemy import inspect

from app.db import Base, SessionLocal, engine
from app.models import Repository

# The 10 RepoLens tables per SDD §11 (see app/models/).
EXPECTED_TABLES = frozenset(
    {
        "repositories",
        "files",
        "symbols",
        "edges",
        "documents",
        "commits",
        "file_commits",
        "issues",
        "chat_sessions",
        "chat_messages",
    }
)


def test_all_ten_model_tables_exist_in_db():
    """The ORM declares exactly the 10 expected tables, and all of them exist
    in the live Postgres database."""
    db_tables = set(inspect(engine).get_table_names())
    model_tables = set(Base.metadata.tables.keys())

    # Sanity: the ORM itself defines exactly the 10 SDD tables.
    assert model_tables == EXPECTED_TABLES, (
        f"ORM defines {sorted(model_tables)}, "
        f"expected {sorted(EXPECTED_TABLES)}"
    )

    # Every model table must be present in Postgres. ``alembic_version`` is
    # not a model table and is intentionally excluded from the comparison.
    missing = model_tables - db_tables
    assert not missing, f"Tables missing from Postgres: {sorted(missing)}"


def test_repositories_round_trip():
    """Insert one repository row, read it back in a fresh session, delete it,
    and confirm it's gone — commits at each step so it's a real (persisted)
    round-trip, not an in-session illusion."""
    url = "https://github.com/example/repolens-smoke.git"
    name = "example/repolens-smoke"
    repo_id = None

    try:
        # ── create (persist) ──────────────────────────────────────────────
        with SessionLocal() as session:
            repo = Repository(url_or_path=url, name=name, status="indexing")
            session.add(repo)
            session.commit()
            session.refresh(repo)
            repo_id = repo.id
        assert repo_id is not None, "insert did not assign a primary key"

        # ── read back in a fresh session (forces a real DB hit) ───────────
        with SessionLocal() as session:
            fetched = session.get(Repository, repo_id)
            assert fetched is not None, "could not read the row back by id"
            assert fetched.name == name
            assert fetched.status == "indexing"

            # ── delete (persist) ──────────────────────────────────────────
            session.delete(fetched)
            session.commit()

        # ── confirm it's gone (third session) ────────────────────────────
        with SessionLocal() as session:
            assert session.get(Repository, repo_id) is None, (
                "row survived the delete — round-trip did not persist"
            )
    finally:
        # Defensive cleanup so a mid-test failure (assertion, flake, interrupt)
        # never leaves a stray row behind. No-op if the delete above succeeded.
        if repo_id is not None:
            with SessionLocal() as session:
                session.execute(sa.delete(Repository).where(Repository.id == repo_id))
                session.commit()
