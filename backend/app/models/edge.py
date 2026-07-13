"""SQLAlchemy model for the `edges` table (SDD §11)."""
from sqlalchemy import Column, ForeignKey, Index, Integer, String

from app.db import Base


class Edge(Base):
    """A relationship between files/symbols (imports | references | contains).

    Edges are polymorphic: ``source_type``/``target_type`` denote whether the
    ``source_id``/``target_id`` refer to a ``file`` or a ``symbol`` (or an
    ``external`` package, in which case ``target_id`` is NULL and
    ``target_label`` carries the package name).
    """

    __tablename__ = "edges"
    __table_args__ = (
        # Nearly every graph query filters by repository_id AND edge_type.
        Index("ix_edges_repository_id_edge_type", "repository_id", "edge_type"),
        # Supports lookups of the form "what does this source node point at".
        Index("ix_edges_source_type_source_id", "source_type", "source_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False)
    # file | symbol
    source_type = Column(String, nullable=False)
    source_id = Column(Integer, nullable=False)
    # file | symbol | external
    target_type = Column(String, nullable=False)
    target_id = Column(Integer, nullable=True)  # nullable if external
    # Human-readable label for external package targets.
    target_label = Column(String, nullable=True)
    # imports | references | contains
    edge_type = Column(String, nullable=False)
