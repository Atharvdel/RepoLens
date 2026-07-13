"""SQLAlchemy model for the `files` table (SDD §11)."""
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from app.db import Base


class File(Base):
    """A source or doc file discovered in a repository during indexing."""

    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("repository_id", "path", name="uq_files_repository_id_path"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    # The UNIQUE(repository_id, path) constraint below already provides a
    # B-tree index with repository_id as its leading column, satisfying the
    # §11 indexing strategy's "index on FK" for this table without a redundant
    # standalone index.
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    path = Column(String, nullable=False)
    language = Column(String, nullable=False)
    loc = Column(Integer, nullable=False)
    last_modified = Column(DateTime, nullable=True)
