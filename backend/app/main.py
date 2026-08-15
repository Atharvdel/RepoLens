"""FastAPI application entrypoint for RepoLens (SDD §12, §16).

Configures middleware (CORS), connects all REST routers (repositories,
chat, graph, search, files), and provides health check probes.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    chat_router,
    files_router,
    graph_router,
    repositories_router,
    search_router,
)

app = FastAPI(
    title="RepoLens API",
    description="Local-first AI repository intelligence platform (SDD §12).",
    version="1.0.0",
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST Routers
app.include_router(repositories_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(graph_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(files_router, prefix="/api")

# Also mount at root for SDD §12 path compatibility
app.include_router(repositories_router)
app.include_router(chat_router)
app.include_router(graph_router)
app.include_router(search_router)
app.include_router(files_router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe -- returns 200 once the API process is running."""
    return {"status": "ok"}


@app.get("/api/health/ollama", tags=["health"])
@app.get("/health/ollama", tags=["health"])
def ollama_health() -> dict[str, Any]:
    """Check Ollama connectivity and return active local models."""
    import json
    import os
    import urllib.request
    from app.agents.planner import OLLAMA_HOST, OLLAMA_MODEL

    try:
        url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return {
                "status": "connected",
                "host": OLLAMA_HOST,
                "active_model": OLLAMA_MODEL,
                "models": models,
                "air_gapped": True,
            }
    except Exception as exc:
        return {
            "status": "disconnected",
            "host": OLLAMA_HOST,
            "active_model": OLLAMA_MODEL,
            "models": [],
            "error": str(exc),
            "air_gapped": True,
        }
