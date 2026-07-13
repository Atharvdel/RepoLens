"""Database engine, session factory, and declarative base for RepoLens.

Configuration is read from environment variables; a local ``.env`` file in the
working directory is loaded automatically via python-dotenv when present. The
defaults match ``docker-compose.yml`` so the scaffold runs out of the box.
(SDD §16 lists ``db.py`` as the DB wiring entrypoint; settings live here too.)
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# No-op if no .env is present; otherwise populates os.environ from it.
load_dotenv()

# PostgreSQL connection string. Default matches docker-compose.yml.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://repolens:repolens@localhost:5432/repolens",
)

# Ollama host for the local LLM used by the agent orchestrator (SDD §9).
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models (SDD §11)."""

    pass
