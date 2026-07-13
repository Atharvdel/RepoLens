"""SQLAlchemy model for the `repositories` table (SDD §11)."""
from sqlalchemy import Column, DateTime, Integer, String

from app.db import Base


class Repository(Base):
    """A git repository the user has added to RepoLens for indexing."""

    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url_or_path = Column(String, nullable=False)
    name = Column(String, nullable=False)
    default_branch = Column(String, nullable=True)
    # One of: indexing | ready | failed. The allowed values are enforced by the
    # application layer; stored as a plain string for migration simplicity.
    status = Column(String, nullable=False)
    indexed_at = Column(DateTime, nullable=True)
    github_owner = Column(String, nullable=True)
    github_repo = Column(String, nullable=True)
