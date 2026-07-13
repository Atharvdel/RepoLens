"""SQLAlchemy model for the `documents` table (SDD §11)."""
from sqlalchemy import Column, ForeignKey, Integer, String, Text

from app.db import Base


class Document(Base):
    """A documentation file (README, CONTRIBUTING, docs/*) extracted at index time
    (SDD §7 step 6) for retrieval and citation by the agents."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    path = Column(String, nullable=False)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
