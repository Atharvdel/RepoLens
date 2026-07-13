"""SQLAlchemy models for the `chat_sessions` and `chat_messages` tables (SDD §11).

Persisted chat history ties each conversation to a repository and records the
agent tool trace alongside each assistant message, for debugging/transparency
(SDD §13.2 "collapsible tool trace").
"""
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from app.db import Base


class ChatSession(Base):
    """A chat conversation scoped to a single repository."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)


class ChatMessage(Base):
    """A single user or assistant message within a chat session.

    ``tool_trace`` stores the agent plan / tool-call record (SDD §9, §15) as
    JSONB so the UI can render the "what the agents actually did" panel.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    # user | assistant
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    tool_trace = Column(JSONB, nullable=True)
