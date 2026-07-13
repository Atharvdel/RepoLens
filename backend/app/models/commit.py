"""SQLAlchemy models for the `commits` and `file_commits` tables (SDD §11).

`commits` is a lightweight fact table (hash/author/date/message) populated once
at index time by the git-history scan (SDD §7 step 7). `file_commits` is the
join table tying files to the commits that touched them — used to answer
"who/when for this file" (History Search, SDD §10) without re-walking git log
on every query.
"""
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)

from app.db import Base


class Commit(Base):
    """A single git commit record for a repository, indexed from the log."""

    __tablename__ = "commits"
    __table_args__ = (
        # History queries filter by repository and order/filter by date.
        Index("ix_commits_repository_id_date", "repository_id", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    hash = Column(String, nullable=False)
    author = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    message = Column(Text, nullable=False)


# Join table: a file can be touched by many commits, and a commit touches many
# files. Composite primary key on (file_id, commit_id); index on file_id for
# the common "commits for this file" lookup (SDD §11 specifies INDEX(file_id)).
file_commits = Table(
    "file_commits",
    Base.metadata,
    Column("file_id", Integer, ForeignKey("files.id"), primary_key=True),
    Column("commit_id", Integer, ForeignKey("commits.id"), primary_key=True),
    Index("ix_file_commits_file_id", "file_id"),
)
