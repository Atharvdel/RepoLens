"""Chat and agent orchestration endpoints (SDD §12, §15).

Routes:
- POST /repositories/{id}/chat                   Run LangGraph agent flow & return cited answer + tool trace
- GET  /repositories/{id}/chat/sessions          List conversation sessions for repository
- GET  /repositories/{id}/chat/sessions/{sid}    Get messages and tool traces for a conversation
"""
from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.graph import GraphResult, run_query
from app.db import get_session
from app.models import ChatMessage, ChatSession, Repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repositories/{repo_id}/chat", tags=["chat"])


# ─── Pydantic schemas ────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Natural language question about repository structure")
    session_id: Optional[int] = Field(None, description="Existing chat session ID to continue conversation")
    model: Optional[str] = Field(None, description="Override local LLM model name")


class ChatResponse(BaseModel):
    session_id: int
    message_id: int
    answer: str
    citations: list[str]
    node_trace: list[str]
    tool_trace: dict[str, Any]
    replans_used: int
    created_at: datetime


class ChatSessionItem(BaseModel):
    id: int
    repository_id: int
    created_at: datetime
    message_count: int = 0
    last_message: Optional[str] = None


class ChatMessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    tool_trace: Optional[dict[str, Any]] = None


# ─── Routes ──────────────────────────────────────────────────────────────────


@router.post("", response_model=ChatResponse)
def ask_question(
    repo_id: int,
    payload: ChatRequest,
    session: Session = Depends(get_session),
):
    """Ask a question about repository structure and receive a cited response with agent execution trace."""
    repo = session.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Resolve or create ChatSession
    chat_session: ChatSession | None = None
    if payload.session_id:
        chat_session = session.get(ChatSession, payload.session_id)
        if not chat_session or chat_session.repository_id != repo_id:
            raise HTTPException(status_code=404, detail="Chat session not found for this repository")
    else:
        chat_session = ChatSession(
            repository_id=repo_id,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(chat_session)
        session.flush()

    # Record user message
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    user_msg = ChatMessage(
        session_id=chat_session.id,
        role="user",
        content=payload.message.strip(),
        created_at=now,
        tool_trace=None,
    )
    session.add(user_msg)
    session.flush()

    # Execute LangGraph query
    try:
        graph_result: GraphResult = run_query(
            question=payload.message.strip(),
            repository_id=repo_id,
            session=session,
            model=payload.model,
        )
    except Exception as e:
        logger.error("LangGraph agent error on question '%s': %s", payload.message, e)
        # Record error response
        err_text = f"An error occurred while analyzing the repository: {str(e)}"
        err_msg = ChatMessage(
            session_id=chat_session.id,
            role="assistant",
            content=err_text,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            tool_trace={"error": str(e)},
        )
        session.add(err_msg)
        session.commit()
        return ChatResponse(
            session_id=chat_session.id,
            message_id=err_msg.id,
            answer=err_text,
            citations=[],
            node_trace=["error"],
            tool_trace={"error": str(e)},
            replans_used=0,
            created_at=err_msg.created_at,
        )

    # Format tool trace for frontend transparency
    serialized_plan = graph_result.plan if isinstance(graph_result.plan, dict) else (
        dataclasses.asdict(graph_result.plan) if dataclasses.is_dataclass(graph_result.plan) else {}
    )

    serialized_search_results = []
    for s_res in graph_result.search_results:
        try:
            serialized_search_results.append(dataclasses.asdict(s_res))
        except Exception:
            pass

    serialized_context_results = []
    for c_res in graph_result.context_results:
        try:
            serialized_context_results.append(dataclasses.asdict(c_res))
        except Exception:
            pass

    tool_trace_payload = {
        "node_trace": graph_result.node_trace,
        "plan": serialized_plan,
        "replans_used": graph_result.replans_used,
        "search_results": serialized_search_results,
        "context_results": serialized_context_results,
        "citations": graph_result.cited_file_paths,
        "hit_file_paths": graph_result.hit_file_paths,
    }

    final_text = graph_result.answer or "No answer could be synthesized from the retrieved context."

    # Record assistant message
    assistant_msg = ChatMessage(
        session_id=chat_session.id,
        role="assistant",
        content=final_text,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        tool_trace=tool_trace_payload,
    )
    session.add(assistant_msg)
    session.commit()
    session.refresh(assistant_msg)

    return ChatResponse(
        session_id=chat_session.id,
        message_id=assistant_msg.id,
        answer=final_text,
        citations=graph_result.cited_file_paths,
        node_trace=graph_result.node_trace,
        tool_trace=tool_trace_payload,
        replans_used=graph_result.replans_used,
        created_at=assistant_msg.created_at,
    )


@router.get("/sessions", response_model=list[ChatSessionItem])
def list_chat_sessions(repo_id: int, session: Session = Depends(get_session)):
    """List conversation sessions for this repository."""
    sessions = session.scalars(
        sa.select(ChatSession)
        .where(ChatSession.repository_id == repo_id)
        .order_by(ChatSession.id.desc())
    ).all()

    items = []
    for s in sessions:
        msgs = session.scalars(
            sa.select(ChatMessage)
            .where(ChatMessage.session_id == s.id)
            .order_by(ChatMessage.id.asc())
        ).all()
        last_msg = msgs[-1].content if msgs else None
        items.append(
            ChatSessionItem(
                id=s.id,
                repository_id=s.repository_id,
                created_at=s.created_at,
                message_count=len(msgs),
                last_message=last_msg,
            )
        )
    return items


@router.get("/sessions/{session_id}", response_model=list[ChatMessageItem])
def get_chat_session_messages(
    repo_id: int,
    session_id: int,
    session: Session = Depends(get_session),
):
    """Retrieve message history and tool traces for a chat session."""
    chat_session = session.get(ChatSession, session_id)
    if not chat_session or chat_session.repository_id != repo_id:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = session.scalars(
        sa.select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
    ).all()

    return [
        ChatMessageItem(
            id=m.id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
            tool_trace=m.tool_trace,
        )
        for m in messages
    ]
