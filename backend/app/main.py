"""FastAPI application entrypoint for RepoLens (SDD §16).

A minimal app at this stage: a single health-check endpoint. The indexing
pipeline, agents, and tools are added in later weeks; nothing here imports them.
"""
from fastapi import FastAPI

app = FastAPI(
    title="RepoLens API",
    description="Local-first AI repository intelligence platform.",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe — returns 200 once the API process is up.

    Intentionally decoupled from the database so it reports process health
    rather than infrastructure state. A DB-aware readiness check can be added
    later if desired.
    """
    return {"status": "ok"}
