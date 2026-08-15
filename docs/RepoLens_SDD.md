# RepoLens — AI Repository Intelligence Platform
### Software Design Document (SDD)
**Version 1.1 — Production-Grade Local Architecture**

---

## 0. Scope & Philosophy

RepoLens is a **100% local, air-gapped AI repository intelligence platform** that turns complex Python, TypeScript, and JavaScript codebases into an interactive, navigable structural map. Using deterministic static analysis (Tree-sitter AST, import resolution, NetworkX topology, Git history, ripgrep) coordinated by a multi-agent LangGraph pipeline powered by local LLMs (via Ollama), RepoLens answers architectural questions with zero hallucination and ground-truth file/line citations.

**Core Philosophy: Comprehension Over Generation**
- No code generation, autocomplete, or unverified guesses.
- All code processing, embeddings, and telemetry stay strictly on the local machine.
- Deterministic static analysis produces structured ground-truth facts; local LLMs are strictly used for planning and natural language synthesis.

---

## 1. System Architecture

RepoLens follows a decoupled, resilient 4-tier architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Next.js 14 Web Application (App Router + Tailwind)          │
│  ┌───────────────────────┐ ┌─────────────────────┐ ┌─────────────────┐  │
│  │ Interactive Cluster   │ │ Architecture Health │ │ Multi-Agent     │  │
│  │ Graph (Minimap + HUD) │ │ Debt Scorecard      │ │ Chat & Tracing  │  │
│  └───────────────────────┘ └─────────────────────┘ └─────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ REST / Streaming JSON (Port 8000)
┌────────────────────────────────────▼────────────────────────────────────┐
│              FastAPI Enterprise Code Intelligence Engine                │
│  ┌─────────────────────────────────┐ ┌────────────────────────────────┐  │
│  │ 9-Stage Indexing Pipeline       │ │ LangGraph Multi-Agent Runtime  │  │
│  │ (Tree-sitter, Git, ripgrep)     │ │ (Planner, Search, Context,     │  │
│  │                                 │ │  Synthesizer)                  │  │
│  └────────────────┬────────────────┘ └───────────────┬────────────────┘  │
└───────────────────┼──────────────────────────────────┼──────────────────┘
                    │                                  │
           ┌────────▼────────┐                ┌────────▼────────┐
           │ PostgreSQL / DB │                │ Ollama Runtime  │
           │ (Files, Symbols,│                │ (Qwen2.5-Coder, │
           │  Edges, Commits)│                │  Llama, Gemma)  │
           └─────────────────┘                └─────────────────┘
```

---

## 2. Multi-Agent Orchestration Layer

RepoLens implements a specialized LangGraph multi-agent pipeline designed to eliminate hallucinations:

```
                  ┌──────────────────────┐
                  │ User Natural Query   │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │    Planner Agent     │
                  │ (Structured Plan)    │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│     Search Agent      │         │     Context Agent     │
│ (Symbol, File, Text,  │         │ (Architecture Graph,  │
│  Multi-Word Aliases)  │         │  Dependency Subgraph, │
│                       │         │  Git Commit History)  │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Re-planning Check   │ ──(0 Hits fallback)──┐
                  └──────────┬───────────┘                      │
                             │ (Hits Present)                   │
                  ┌──────────▼───────────┐                      │
                  │  Synthesizer Agent   │ ◄────────────────────┘
                  │ (Ground-Truth Cit.)  │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │ Verified Markdown    │
                  │ with Line Citations  │
                  └──────────────────────┘
```

### Agent Roles:
1. **Planner Agent**: Parses the question intent and emits a strict JSON execution plan choosing the most targeted tool (`search_symbols`, `search_files`, `search_text`, `architecture`, `dependency_graph`, `file_history`).
2. **Search Agent**: Executes atomic symbol queries, file path matches, or tokenized ripgrep searches with intelligent multi-word fallback. Retrieves verified top 75-line source snippets from disk.
3. **Context Agent**: Traverses NetworkX import and reference subgraphs, calculates in-degree centrality, and aggregates git commit history and contributors across files or repo-wide.
4. **Synthesizer Agent**: Consumes structured tool results and exact code snippets to generate detailed, multi-tiered explanations with verified inline backtick citations.

---

## 3. 9-Stage Indexing Pipeline

Every repository undergoes automated batch indexing:

1. **Clone / Fetch**: Clones remote Git repositories (`data/repos/{id}_{name}`) or validates local filesystem directories.
2. **File Walk & Prune**: Traverses directories respecting `.gitignore`, excluding lockfiles, binary artifacts, and large assets (>1MB).
3. **Multi-Language AST Parsing**:
   - Python via Tree-sitter (`class`, `function`, `method`, `variable`, imports, docstrings).
   - TypeScript & JavaScript via Tree-sitter (`classes`, `functions`, `interfaces`, `enums`, `types`, imports with `@/` path alias resolution).
4. **Import Dependency Graph**: Resolves internal and external module linkages into `edges` table.
5. **Reference Indexing**: Ripgrep-powered symbol usage mapping across project files.
6. **Documentation Ingestion**: Parses `README.md`, `CONTRIBUTING.md`, and markdown docs into searchable records.
7. **Git History Extraction**: Analyzes commit SHAs, timestamps, author commit counts, and file touch frequencies.
8. **GitHub Metadata Sync**: Synchronizes open issues, pull requests, and labels via PyGithub when a token is provided.
9. **Transactional Finalization**: Atomically persists all models and transitions repository status to `ready`.

---

## 4. Frontend & Visual Intelligence Features

### 🕸️ Interactive Subsystem Cluster Graph
- **Clustered Force Layout**: Groups nodes into distinct directory subsystems with soft radial clustering.
- **Visual Edge Weights**: Renders dependency strength using dynamic stroke width (`1px`–`4.5px`) and color intensity.
- **Minimap HUD**: Live radar thumbnail in the bottom-right corner tracking camera viewports and hub nodes.
- **Search Auto-Focus**: Type any file query and press `Enter` to smoothly fly the camera and center the node with an illuminated pulsing beacon ring.
- **1-Click Exports**:
  - **Mermaid Markdown**: Generates formatted subgraph Mermaid.js syntax for GitHub/Notion.
  - **Vector SVG**: Downloads high-resolution vector diagrams for presentations and documentation.

### 🩺 Architecture Health & Tech Debt Scorecard
- **Health Score & Grade**: Computes overall code health (`0–100`, `Grade A+` to `F`).
- **Circular Dependency Detector**: Traverses directed cycles (`nx.simple_cycles(G)`).
- **Coupling Hotspot Radar**: Pinpoints high in-degree files and computes downstream blast radius.
- **Dead / Orphaned File Isolation**: Identifies unreferenced non-entrypoint source files.
- **Subsystem Modularity Ratio**: Measures intra-subsystem cohesion vs cross-module coupling.

### 📂 VS Code-Style Hierarchical Code Inspector
- Collapsible file explorer tree with live instant search filtering.
- Side-by-side tabs for Syntax-Highlighted Source Code, AST Extracted Symbols, and Commit History.

---

## 5. Technology Stack Summary

| Component | Technology |
|---|---|
| **Frontend UI** | Next.js 14 (App Router), React 18, TailwindCSS, Lucide Icons |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 |
| **Database & ORM** | PostgreSQL / SQLite (Local-First), SQLAlchemy 2.0, Alembic |
| **AST Parsers** | Tree-sitter (Python, TypeScript, TSX, JavaScript) |
| **Graph Mathematics** | NetworkX (In-Degree Centrality, Simple Cycles, BFS Subgraphs) |
| **Search Engine** | Ripgrep (`rg` CLI binary wrapper) |
| **Agent Orchestrator**| LangGraph, LangChain-Ollama |
| **Local LLM Engine** | Ollama (`qwen2.5-coder:1.5b`, `llama3.2`, `gemma2`) |

---

## 6. Verification & Quality Assurance

- **163 Automated Unit Tests** (`pytest -v`) covering API endpoints, Tree-sitter parsers, agent dispatchers, and graph algorithms.
- **Real-World Live Verified Repositories**:
  - `GitThatOffer` (Next.js App Router, MongoDB, NextAuth)
  - `Tesseract` (Next.js Pages Router, Supabase, Team Auth)
  - `Flask` (Python WSGI Framework)
