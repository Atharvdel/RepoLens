"""SQLAlchemy ORM models for RepoLens (SDD §11).

Importing this package registers every model — and the ``file_commits`` join
table — on the shared declarative metadata (``app.db.Base.metadata``), so both
Alembic (autogenerate/upgrade) and the application see the full schema.

Tables (SDD §11):
    repositories, files, symbols, edges, documents, commits, file_commits,
    issues, chat_sessions, chat_messages
"""
from app.models.chat import ChatMessage, ChatSession
from app.models.commit import Commit, file_commits
from app.models.document import Document
from app.models.edge import Edge
from app.models.file import File
from app.models.issue import Issue
from app.models.repository import Repository
from app.models.symbol import Symbol

__all__ = [
    "ChatMessage",
    "ChatSession",
    "Commit",
    "Document",
    "Edge",
    "File",
    "Issue",
    "Repository",
    "Symbol",
    "file_commits",
]
