"""SQLAlchemy model for the `symbols` table (SDD §11)."""
from sqlalchemy import Column, ForeignKey, Integer, String, Text

from app.db import Base


class Symbol(Base):
    """A class/function/method/variable symbol extracted from a file.

    ``parent_symbol_id`` links methods to their enclosing class symbol
    (methods-in-class, SDD §11).
    """

    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    # One of: class | function | method | variable.
    kind = Column(String, nullable=False)
    line_start = Column(Integer, nullable=False)
    line_end = Column(Integer, nullable=False)
    docstring = Column(Text, nullable=True)
    # Self-referential FK for methods nested inside a class symbol.
    parent_symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=True, index=True)
