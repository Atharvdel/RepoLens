"""SQLAlchemy model for the `issues` table (SDD §11)."""
from sqlalchemy import ARRAY, Column, ForeignKey, Integer, String

from app.db import Base


class Issue(Base):
    """A cached GitHub issue/PR for a repository, pulled at index time by
    PyGithub (SDD §7 step 8). Metadata only — title, labels, linked files,
    state. No PR-diff reasoning (explicitly out of scope; SDD §0)."""

    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    number = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    # open | closed (GitHub state).
    state = Column(String, nullable=False)
    labels = Column(ARRAY(String), nullable=False)
    url = Column(String, nullable=False)
    # File paths parsed from issue/PR body where derivable; nullable.
    linked_files = Column(ARRAY(String), nullable=True)
