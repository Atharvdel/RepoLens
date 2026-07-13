"""Alembic migrations environment for RepoLens.

Reads the database URL from ``app.db`` (which honours ``DATABASE_URL`` / a local
``.env``) and uses the SQLAlchemy model metadata declared under ``app.models``
as the migration target — so the migrations and the ORM always share one source
of truth for the schema (SDD §11).
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the backend/ package importable regardless of the current working
# directory the Alembic CLI is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import Base, DATABASE_URL  # noqa: E402
import app.models  # noqa: E402,F401  # registers every model on Base.metadata

config = context.config

# Drive the connection from the same source the application uses.
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection (`alembic upgrade --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
