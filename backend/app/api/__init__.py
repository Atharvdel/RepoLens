"""API routes package for RepoLens (SDD §12, §16)."""
from app.api.chat import router as chat_router
from app.api.files import router as files_router
from app.api.graph import router as graph_router
from app.api.repositories import router as repositories_router
from app.api.search import router as search_router

__all__ = [
    "chat_router",
    "files_router",
    "graph_router",
    "repositories_router",
    "search_router",
]
