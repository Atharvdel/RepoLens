# RepoLens

Local-first AI repository intelligence platform: turns an unfamiliar
Python / TypeScript / JavaScript codebase into a navigable structural map, with
a local LLM (Ollama, orchestrated by LangGraph) that answers questions with
cited, grounded facts rather than guesses. No code generation — comprehension
only. No code ever leaves the machine.

See [`RepoLens_SDD (1).md`](./RepoLens_SDD%20%281%29.md) for the full Software
Design Document (scope, architecture, schema, roadmap).

## Stack

- **Backend** — FastAPI · SQLAlchemy · Alembic · PostgreSQL
- **Indexing** — Tree-sitter (Python / TS / JS) · GitPython · ripgrep · PyGithub · NetworkX
- **Agent layer** — LangGraph · LangChain-Ollama · Ollama (local LLM)
- **Frontend** — Next.js (scaffolded in Week 3)

## Layout

```
backend/            Python API, ORM models (SDD §11), Alembic migrations
frontend/           Next.js app (added in Week 3)
docker-compose.yml  PostgreSQL (backend + Ollama services added later)
```

## Prerequisites

- Python 3.10+ (3.12 recommended; this scaffold was built with 3.12 via [`uv`](https://github.com/astral-sh/uv))
- Docker (for PostgreSQL), or a local PostgreSQL 15+
- Ollama — installed before the agent layer (Week 2)

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows; macOS/Linux: .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # adjust DATABASE_URL / OLLAMA_HOST if needed
```

## Start PostgreSQL

```bash
docker compose up -d            # from the project root
```

## Apply the schema migration

```bash
cd backend && alembic upgrade head
```

## Run the API

```bash
uvicorn app.main:app --reload   # health check: http://localhost:8000/health
```
