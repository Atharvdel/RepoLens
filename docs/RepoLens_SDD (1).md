# RepoLens — AI Repository Intelligence Platform
### Software Design Document (SDD)
**Version 1.0 — Scoped for Solo Developer, 4-Week MVP**

---

## 0. Scope Note (Read First)

This document deliberately descopes several items from the original product brief. Each cut is justified inline where relevant, and summarized here so reviewers can see at a glance what changed and why:

| Original Ask | Decision | Reason |
|---|---|---|
| Multi-language parsing | **Python + TypeScript/JavaScript only** for MVP | Tree-sitter grammars differ enough per language that "generic" support is a myth; two languages covers most student/OSS repos and is achievable in the timeline |
| Call graph | **Reference graph** (symbol → usage sites), not a true call graph | Resolving dynamic dispatch, indirect calls, and cross-module resolution correctly is a multi-month static-analysis problem, not a 4-week one |
| Vector DB / embeddings | **Omitted from MVP** | ripgrep + symbol index + folder/import graph answers the large majority of "where is X" queries without the complexity, cost, and failure modes of embedding search |
| 6 UI pages (Dashboard, Repo Overview, Architecture View, Graph View, Issue Explorer, Repo Chat, Learning View) | **3 MVP screens**: Repo Overview + Chat, Graph/Architecture View, Indexing Status. Issue Explorer and Learning View → Future Work | Each additional page is real design + state management work; 3 polished screens beat 6 half-built ones |
| 5 agents | **4 agents**, one merged | Every agent hop is a local-LLM round trip (slow, error-prone); fewer, sharper agents reduce both build time and runtime latency |
| Incremental re-indexing | **Full re-index only** for MVP, incremental → Future Work | Correctness of incremental invalidation is its own project; full re-index on a small/medium repo is fast enough locally |
| GitHub Issues/PR semantic analysis | **Metadata only** (title, labels, linked files via PyGithub), no PR-diff reasoning | Diff-level reasoning needs the call graph maturity we're explicitly not building yet |

Everything below is scoped to be buildable by one engineer, part-time-equivalent effort, in four weeks, using only free/local software.

---

## 1. Executive Summary

RepoLens is a locally-run tool that helps developers understand unfamiliar codebases by answering structural questions — "how does auth work," "what touches this file," "where do I start" — using deterministic code analysis (parsing, dependency graphs, git history, symbol search) combined with a local LLM (via Ollama) that orchestrates those tools and explains the results in natural language.

It is explicitly **not** a code-generation assistant. It generates no code and makes no edits. Its only job is comprehension: turning a repository into something a new contributor can navigate in minutes instead of days.

The MVP targets single repositories in Python and/or TypeScript/JavaScript, runs entirely on local infrastructure (no code ever leaves the machine or network), and is built on a stack (FastAPI, PostgreSQL, LangGraph, Tree-sitter, ripgrep, NetworkX, Ollama, Next.js) chosen so that no component requires a paid service, API key, or GPU beyond what a local model needs to run.

---

## 2. Problem Statement

Developers joining a new codebase — new hires, open-source contributors, interns — spend a disproportionate amount of their first days or weeks just building a mental map: which folder does what, where the entry points are, how a request flows through the system, which files are safe to touch. This knowledge usually lives in senior engineers' heads and is transferred inefficiently through Slack messages, tribal knowledge, and trial and error.

Existing tools don't solve this well:

- **Search tools** (`grep`, GitHub code search) find text, not structure. They don't tell you *why* a file matters.
- **AI coding assistants** (Copilot, Cursor) are optimized for writing the next line, not explaining the previous ten thousand.
- **Enterprise code intelligence** (Sourcegraph, Cody) does this well but is priced and deployed for organizations, not something a student or small team can self-host in an afternoon for free.

The gap is a **free, local-first, explanation-oriented** tool for repository comprehension — not code completion, not enterprise search, but a guided map.

---

## 3. Competitive Analysis

| Product | What it does | Why RepoLens is different |
|---|---|---|
| **GitHub Copilot / Cursor** | Autocomplete and inline code generation, conversational edits | RepoLens never writes code; it explains existing structure. Different job entirely — these tools assume you already understand the file you're in. |
| **Sourcegraph / Cody** | Enterprise-grade code search, cross-repo navigation, AI chat over code | Extremely capable but requires infrastructure, licensing, and is overkill for a single repo or a solo user; not meaningfully free at scale. RepoLens trades cross-repo scale for zero-cost, fully local operation. |
| **Plain grep / IDE "Find Usages"** | Fast, precise, zero setup | No narrative layer — doesn't explain *why*, doesn't synthesize across files, doesn't answer "where should I start." RepoLens uses these as primitives, not as the end product. |
| **ChatGPT/Claude pasted code** | General reasoning over pasted snippets | No structural ground-truth; hallucinates file relationships; can't see the whole repo; sends code off-network. RepoLens's deterministic tools are the entire point of avoiding this failure mode. |

**Honest take:** onboarding is the most defensible use case, not because it's the only use case, but because it's the one where "I don't know where anything is yet" makes deterministic structural tools obviously more valuable than raw LLM guessing. Expanding beyond onboarding (e.g., pre-PR impact analysis) is realistic future work, not MVP scope.

---

## 4. Goals

1. Given a git repository (Python and/or TS/JS), produce a navigable structural map: folders, modules, key symbols, import/dependency relationships.
2. Answer natural-language questions about repository structure ("where is X handled," "what imports Y," "what should I read first") with answers grounded in real, cited files — not hallucinated ones.
3. Visualize the module/dependency graph interactively.
4. Surface basic git-history and GitHub-issue context (who touched a file, when, linked issues) where available.
5. Run entirely locally: no code, embeddings, or queries leave the user's machine/network.
6. Be usable by a single developer building it in ~4 weeks of solo effort, using only free/open-source tooling.

---

## 5. Non-Goals (MVP)

- Code generation, autocompletion, or in-editor suggestions.
- True call-graph resolution (dynamic dispatch, reflection, cross-service calls).
- Multi-repo / cross-repo intelligence.
- Semantic (embedding-based) search.
- Incremental re-indexing on every commit.
- Support for languages beyond Python and TypeScript/JavaScript.
- PR-diff-level reasoning or automated code review.
- Multi-user collaboration features, auth, permissions.
- Fine-tuning or training any model.

---

## 6. System Architecture

RepoLens is a single-tenant, self-hosted application with four logical layers:

```
┌─────────────────────────────────────────────────────────┐
│  Next.js Frontend (Dashboard, Chat, Graph View)          │
└───────────────────────────┬───────────────────────────────┘
                             │ REST (JSON)
┌───────────────────────────▼───────────────────────────────┐
│  FastAPI Backend                                           │
│  ┌────────────────┐  ┌─────────────────────────────────┐  │
│  │ Indexing Service │  │ LangGraph Agent Orchestrator    │  │
│  │ (parse, graph,    │  │ (Planner → Search/Arch/GitHub   │  │
│  │  git, GitHub sync)│  │  → Synthesizer)                 │  │
│  └────────┬─────────┘  └────────────┬────────────────────┘  │
└───────────┼─────────────────────────┼───────────────────────┘
            │                         │
   ┌────────▼────────┐       ┌────────▼────────┐
   │  PostgreSQL      │       │  Ollama          │
   │  (files, symbols,│       │  (Qwen3/Gemma/   │
   │   edges, commits,│       │   Llama, local)  │
   │   issues)        │       └──────────────────┘
   └──────────────────┘
            ▲
   ┌────────┴────────┐
   │ Tree-sitter,     │
   │ ripgrep, GitPython,│
   │ PyGithub, NetworkX │
   └─────────────────┘
```

**Why this shape:** the Indexing Service and the Agent Orchestrator are cleanly separated because they run on different triggers (indexing runs once per repo add/refresh; the agent runs per user query) and have different performance characteristics (indexing is I/O and CPU bound; agents are LLM-latency bound). Keeping them decoupled means indexing can be tested and debugged entirely without touching LangGraph or Ollama.

All state lives in PostgreSQL. NetworkX graphs are built in-memory from Postgres rows at query time for small/medium repos rather than persisted as a separate graph database — this avoids running a second database (e.g., Neo4j) for a graph that, at MVP scale (single repo, thousands not millions of nodes), fits comfortably in memory. This is revisited in Future Work if multi-repo/graph-database needs arise.

---

## 7. Repository Indexing Flow

Indexing is a **one-shot batch pipeline**, triggered manually ("Index Repository") or on repo add. It is not incremental in the MVP (see Non-Goals).

**Steps, in order:**

1. **Clone/Pull** — `GitPython` clones the target repo (public URL or local path) into a working directory.
2. **File Walk & Filter** — walk the tree, exclude `node_modules`, `.git`, build artifacts, binary files, anything over a size threshold (configurable, default 1MB) or matching `.gitignore`.
3. **Parse (Tree-sitter)** — for each `.py`/`.ts`/`.tsx`/`.js`/`.jsx` file, parse to an AST; extract: classes, functions/methods, top-level variables, imports, docstrings/leading comments. Store as rows in `symbols` and `imports`.
4. **Build Import Graph** — resolve import statements to file paths where possible (relative imports resolved directly; package imports flagged as external). Produces `edges` of type `imports`.
5. **Build Reference Index** — for each extracted symbol, use `ripgrep` to find other locations in the repo referencing that symbol name (scoped to avoid trivial name collisions where feasible, e.g. excluding matches inside string literals/comments where cheap to detect). Produces `edges` of type `references`. This is the "reference graph," explicitly not a resolved call graph.
6. **Parse Documentation** — extract README, CONTRIBUTING, and any `docs/` markdown into a `documents` table for retrieval and citation.
7. **Git History Scan** — `GitPython` walks commit log per file: last modified date, top contributors by commit count, commit message summaries (stored, not summarized by LLM at index time to save cost).
8. **GitHub Metadata (optional)** — if a GitHub token is configured, `PyGithub` pulls open issues/PRs (title, labels, linked files if parseable from description, state). Stored as-is; no LLM reasoning over diffs.
9. **Persist & Mark Ready** — write everything transactionally; mark repo status `ready` in Postgres so the frontend can move from "indexing" to "chat enabled."

**Realistic scale target for MVP:** repos up to ~2,000 files / ~200k LOC should index in low single-digit minutes on a typical laptop. This is a stated assumption, not a guarantee — very large monorepos are out of scope.

---

## 8. Knowledge Representation

### 8.1 Should RepoLens build a knowledge graph? — **Yes, but a modest one.**

A full "enterprise knowledge graph" (with resolved call graphs, type inference, cross-language linking) is not achievable solo in a month, and isn't necessary for the questions this product targets. A **lightweight structural graph**, stored as plain relational rows and materialized into NetworkX at query time, gives 80% of the value at a fraction of the engineering cost.

**Node types:**
- `File` (path, language, LOC, last modified)
- `Symbol` (class/function/method name, file, line range, docstring)
- `Module/Package` (directory-derived)
- `Document` (README, CONTRIBUTING, docs files)
- `Commit` (hash, author, date, message) — lightweight, not a full node in the graph traversal sense, more a joinable fact table
- `Issue` (GitHub issue/PR metadata)

**Edge types:**
- `imports` (File → File / File → external package)
- `contains` (Module → File, File → Symbol)
- `references` (Symbol → Symbol, the reference-graph edge from step 5 above)
- `modified_by` (File → Commit → Author, via join, not a dense edge)
- `mentions` (Issue → File, where derivable from issue text/labels)

**Storage:** all of the above are Postgres tables (see §11), not a dedicated graph database. **Querying:** for graph-shaped questions (e.g., "what's the dependency neighborhood of file X," "shortest path between two modules"), the backend loads the relevant subset of `edges`/`symbols` rows into a NetworkX `DiGraph` in memory and runs standard graph algorithms (BFS neighborhood, shortest path, degree/centrality for "important file" heuristics). This is fast at MVP scale and requires no new infrastructure.

**Why not a dedicated graph DB (e.g., Neo4j):** it's a real, useful option long-term, but adds a second database to install, operate, and keep in sync with Postgres, for a graph size that fits in memory today. This is the single highest-value simplification in the whole design — it removes an entire piece of infrastructure without meaningfully reducing capability at MVP scale.

---

## 9. Agent Architecture

**Design principle carried over from the brief: as few agents as do the job well, because every agent hop is a local-LLM round trip.** Four agents, not five — Architecture and GitHub Context are merged into one **Context Agent**, since at MVP scope "explain the architecture" and "pull related git/issue context" both reduce to "look up structured facts and hand them to the Synthesizer," and splitting them added a hop without adding a distinct skill.

### 9.1 Planner Agent
- **Responsibility:** interpret the user's question, decide which tool(s)/agent(s) are needed and in what order, and produce a small structured plan (not free text) that downstream agents execute.
- **Inputs:** user query, repo ID, short conversation history (last 2–3 turns).
- **Outputs:** JSON plan, e.g. `{"steps": [{"agent": "search", "tool": "symbol_search", "args": {"name": "authenticate"}}, {"agent": "context", "tool": "file_history", "args": {"file": "..."}}]}`.
- **Why it exists:** local models do noticeably better at picking from a constrained menu of tool calls than at open-ended repo reasoning. Forcing a structured plan up front keeps every later step narrow.

### 9.2 Search Agent
- **Responsibility:** execute symbol/file/text search tools and return structured, cited results (file path + line numbers), never prose summaries of code it hasn't retrieved.
- **Tools:** Symbol Search, File Search, ripgrep-backed Text Search (see §10).
- **Inputs:** plan step (tool name + args).
- **Outputs:** structured JSON hits.
- **Why it exists:** isolates all "find things" logic so the Synthesizer never has to guess file paths — it only explains what Search actually found.

### 9.3 Context Agent (merged Architecture + GitHub Context)
- **Responsibility:** answer structural/relational questions (module boundaries, dependency neighborhood, "what's related to this file") and pull historical/GitHub metadata (recent commits, contributors, linked issues) for a given file or symbol.
- **Tools:** Architecture Query (NetworkX subgraph), Dependency Graph Query, History Search, GitHub Metadata Loader.
- **Inputs:** plan step, typically a file or symbol identifier from the Search Agent's output.
- **Outputs:** structured JSON — subgraph description, commit list, issue list.
- **Why it exists:** these two originally-separate agents share an input shape (a file/symbol → contextual facts about it) and a single local-model call comfortably handles both without a quality drop, saving a full round trip per query.

### 9.4 Synthesizer Agent
- **Responsibility:** the only agent that produces user-facing natural language. Takes all structured outputs collected so far and writes a grounded, cited answer ("Authentication is handled in `auth/session.py` (`authenticate()`, lines 40–78), imported by `api/routes.py`...").
- **Inputs:** original question + all structured tool outputs from prior steps.
- **Outputs:** final markdown answer with inline file/line citations.
- **Why it exists:** keeping synthesis as a single final step — rather than letting each agent write prose — is what prevents hallucinated file references; only this agent talks to the user, and it's explicitly instructed to cite only facts present in its structured inputs.

**Orchestration:** LangGraph wires these as a directed flow: `Planner → (Search | Context, possibly both, possibly looped once if Planner requests a follow-up) → Synthesizer`. The Planner may request at most one re-planning loop (e.g., if Search returns zero hits, replan with a broader query) to bound worst-case latency on weak local models.

---

## 10. Tool Architecture

All tools return **structured JSON**, never free text, so agents (and especially the Synthesizer) work from facts rather than paraphrased summaries.

| Tool | Backing | Input | Output (shape) |
|---|---|---|---|
| **Repository Parser** | Tree-sitter | file path | `{symbols: [{name, kind, line_start, line_end, docstring}], imports: [{target, is_external}]}` |
| **Symbol Search** | Postgres query on `symbols` | name/partial name, optional kind filter | `[{name, kind, file, line_start, line_end, docstring}]` |
| **File Search** | Postgres query on `files` (path/name match) | filename or path fragment | `[{path, language, loc, last_modified}]` |
| **Text Search** | ripgrep subprocess | query string, optional glob | `[{file, line, matched_text}]` (capped result count) |
| **Dependency Graph Builder** | NetworkX over `edges(imports)` | file or module | `{node, neighbors_in, neighbors_out, depth}` |
| **Architecture Query** | NetworkX subgraph + centrality | file/module or none (whole-repo) | `{modules: [...], key_files: [...] (by degree centrality), edges: [...]}` |
| **History Search** | GitPython log parsing (cached at index time) | file path | `{last_modified, top_contributors: [{author, commits}], recent_commits: [{hash, message, date, author}]}` |
| **GitHub Metadata Loader** | PyGithub (cached at index time) | repo, optional file path | `{issues: [{number, title, labels, state, url}], prs: [...]}` |
| **Documentation Parser** | Markdown parse at index time | doc path or none | `{title, sections: [{heading, text}]}` |

**Why deterministic, structured tools rather than LLM summarization of raw files:** this is the core reliability lever for the whole product. A 7–14B local model asked to "summarize what this file does" from raw source will sometimes miss or invent details; a model asked to "explain these five structured facts you were just handed" is dramatically more reliable, and that reliability gap is exactly what determines whether the product is usable on Qwen3/Gemma/Llama-class models versus something GPT-4-class.

---

## 11. Database Design (PostgreSQL)

```
repositories
  id (PK), url_or_path, name, default_branch, status (indexing|ready|failed),
  indexed_at, github_owner, github_repo (nullable)

files
  id (PK), repository_id (FK), path, language, loc, last_modified,
  UNIQUE(repository_id, path)

symbols
  id (PK), file_id (FK), name, kind (class|function|method|variable),
  line_start, line_end, docstring, parent_symbol_id (nullable FK, for methods-in-class)
  INDEX(name), INDEX(file_id)

edges
  id (PK), repository_id (FK), source_type (file|symbol), source_id,
  target_type (file|symbol|external), target_id (nullable if external),
  target_label (for external packages), edge_type (imports|references|contains),
  INDEX(repository_id, edge_type), INDEX(source_type, source_id)

documents
  id (PK), repository_id (FK), path, title, content (text)

commits
  id (PK), repository_id (FK), hash, author, date, message
  INDEX(repository_id, date)

file_commits
  file_id (FK), commit_id (FK)  -- join table
  INDEX(file_id)

issues
  id (PK), repository_id (FK), number, title, state, labels (text[]), url, linked_files (text[] nullable)

chat_sessions
  id (PK), repository_id (FK), created_at

chat_messages
  id (PK), session_id (FK), role (user|assistant), content, created_at,
  tool_trace (jsonb, nullable) -- stores the agent plan/tool calls for debugging/transparency
```

**Indexing strategy:** B-tree indexes on all foreign keys and on `symbols.name` / `files.path` for fast lookup; `edges(repository_id, edge_type)` since nearly every graph query filters by both.

**Caching:** History Search and GitHub Metadata results are computed once at index time and stored (not fetched live per chat query) — this avoids rate-limiting the GitHub API and avoids re-walking git log on every question.

**Incremental updates:** explicitly out of MVP scope. Re-indexing means re-running the full pipeline and replacing rows for that `repository_id` in a transaction. This is simple and correct, at the cost of being slower than a true incremental system — an acceptable tradeoff given the one-month budget (see §18 Risks).

---

## 12. API Design (REST, FastAPI)

```
POST   /repositories                  Add a repo (url or local path) → starts indexing job
GET    /repositories                  List repos with status
GET    /repositories/{id}             Repo detail + summary stats
POST   /repositories/{id}/reindex     Trigger full re-index
DELETE /repositories/{id}             Remove repo and all derived data

GET    /repositories/{id}/graph       Query params: scope=whole|file|module, target=<path>
                                       → nodes/edges for the Graph View

GET    /repositories/{id}/search      Query params: q, type=symbol|file|text
                                       → structured search hits

POST   /repositories/{id}/chat        Body: {session_id?, message}
                                       → runs LangGraph flow, returns synthesized answer + tool_trace

GET    /repositories/{id}/files/{path}       File detail: symbols, imports, history, referencing files
GET    /repositories/{id}/issues             List cached issues (if GitHub configured)
GET    /repositories/{id}/architecture       Whole-repo module map + "key files" by centrality
```

All endpoints return JSON; indexing is async (job started, status polled via `GET /repositories/{id}`), since a full index can take minutes.

---

## 13. UI Design

Three MVP screens, deliberately fewer than the six originally scoped (see §0).

### 13.1 Dashboard + Repo Overview (merged)
- **Purpose:** entry point — add a repo, see indexing status, see a repo's high-level stats once ready (file count, language breakdown, key files by centrality, README rendered).
- **Components:** repo list/cards, "Add Repository" form, status badge (indexing/ready/failed), stat tiles, rendered README panel.
- **Interactions:** add repo → poll status → click into repo when ready.

### 13.2 Repository Chat + Architecture/Graph View (combined workspace)
- **Purpose:** the core product experience — ask a question, get a cited answer, and see the relevant part of the dependency graph highlighted alongside it.
- **Components:** chat panel (message list, input box, citations rendered as clickable file links), graph panel (NetworkX layout rendered via a lightweight graph visualization library, e.g. `react-flow` or `cytoscape.js`, both free/OSS) that highlights nodes/edges referenced in the current answer, a collapsible "tool trace" showing what the agents actually did (transparency, and a debugging aid during development).
- **Interactions:** ask question → see plan execute (optional streaming of intermediate steps) → answer appears with citations → clicking a citation highlights it in the graph and shows a code snippet.

### 13.3 Indexing Status / Repo Settings
- **Purpose:** operational view — see indexing progress/logs, reconfigure GitHub token, trigger re-index, delete repo.
- **Components:** progress indicator per pipeline stage (§7 steps), log/error panel, settings form.
- **Interactions:** trigger reindex, edit GitHub token, delete.

**Deferred to Future Work:** standalone Issue Explorer (issues are visible read-only in chat/context for MVP, not a dedicated browsing UI) and a "Learning View" (guided onboarding path/tour) — both good ideas, both real UI/UX projects in their own right that don't fit the four-week budget alongside everything above.

---

## 14. Sequence Diagram — "Ask a Question" Flow

```
User        Frontend       FastAPI        LangGraph        Postgres      Ollama
 |  types Q    |               |               |                |            |
 |------------>|  POST /chat   |               |                |            |
 |             |-------------->| start flow    |                |            |
 |             |               |-------------->| Planner call   |            |
 |             |               |               |--------------------------->|
 |             |               |               |<---------------------------|
 |             |               |               | plan (JSON)    |            |
 |             |               |               | Search/Context |            |
 |             |               |               | tool calls----->|            |
 |             |               |               |<-----------------|            |
 |             |               |               | structured facts|            |
 |             |               |               | Synthesizer call|            |
 |             |               |               |--------------------------->|
 |             |               |               |<---------------------------|
 |             |               |               | final answer + citations   |
 |             |               |<--------------|                |            |
 |             |<--------------| response       |                |            |
 |<------------| render answer +|               |                |            |
 |             | highlight graph|               |                |            |
```

---

## 15. LangGraph Flow

```
        ┌───────────┐
        │  Planner   │
        └─────┬──────┘
              │ plan (list of steps)
       ┌──────┴───────┐
       ▼              ▼
 ┌───────────┐  ┌─────────────┐
 │  Search    │  │  Context    │      (run in parallel where plan allows;
 │  Agent     │  │  Agent      │       sequential where one depends on the other's output)
 └─────┬──────┘  └──────┬──────┘
       │                │
       └───────┬────────┘
               ▼
      ┌────────────────┐
      │ replan needed?  │──yes──▶ back to Planner (max 1 loop)
      └────────┬────────┘
               │ no
               ▼
      ┌────────────────┐
      │  Synthesizer    │
      └────────┬────────┘
               ▼
         final answer
```

State passed through the graph is a single typed object: `{query, plan, search_results, context_results, replans_used, final_answer}`. The one-loop cap on replanning is a deliberate latency bound — without it, a weak local model could plausibly loop indefinitely on an unanswerable query.

---

## 16. Folder Structure

```
repolens/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry
│   │   ├── api/                    # route modules (repositories, chat, graph, search)
│   │   ├── indexing/
│   │   │   ├── pipeline.py         # orchestrates §7 steps
│   │   │   ├── parser.py           # Tree-sitter wrapper
│   │   │   ├── import_graph.py
│   │   │   ├── reference_index.py  # ripgrep-backed
│   │   │   ├── git_history.py
│   │   │   └── github_sync.py
│   │   ├── agents/
│   │   │   ├── planner.py
│   │   │   ├── search_agent.py
│   │   │   ├── context_agent.py
│   │   │   ├── synthesizer.py
│   │   │   └── graph.py            # LangGraph wiring
│   │   ├── tools/                  # thin wrappers exposed to agents, §10
│   │   ├── models/                 # SQLAlchemy models, §11
│   │   └── db.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app/                        # Next.js app router
│   │   ├── dashboard/
│   │   ├── repo/[id]/chat/
│   │   └── repo/[id]/settings/
│   ├── components/
│   └── package.json
├── docker-compose.yml              # postgres + backend + ollama (optional convenience)
└── README.md
```

---

## 17. Development Roadmap (4 Weeks, Solo)

**Week 1 — Indexing Pipeline & Data Model**
- Postgres schema + migrations. Tree-sitter parsing for Python, then TS/JS. Import graph resolution. ripgrep reference index. Git history scan. Basic FastAPI CRUD + async indexing job. *Goal: can add a repo and see it fully indexed in the DB.*

**Week 2 — Tools & Agents**
- Implement all tools in §10 as callable functions with tests against a real sample repo. Build Planner, Search, Context, Synthesizer prompts and wire LangGraph flow. Get end-to-end chat working against the FastAPI backend (no frontend yet — test via API client). *Goal: `POST /chat` returns a correct, cited answer for basic questions.*

**Week 3 — Frontend**
- Next.js scaffold, Dashboard/Repo Overview screen, Chat + Graph View screen (integrate `react-flow`/`cytoscape.js`), Indexing Status screen. Wire to backend APIs. *Goal: full happy-path usable end to end in the browser.*

**Week 4 — GitHub Integration, Polish, Hardening**
- PyGithub metadata sync, error handling and edge cases (empty repos, parse failures, huge files), replanning-loop tuning against actual local model behavior (this will need real iteration — local models will surprise you), basic auth/token handling for GitHub, documentation, demo repo walkthrough, bug fixing buffer.

**Should Have (attempt if ahead of schedule, cut without shame if not):**
- Standalone Issue Explorer view.
- Streaming intermediate agent steps to the frontend (nicer UX, not required for correctness).
- Multi-repo comparison / switching without full page reload.
- Basic "key files to read first" heuristic surfaced explicitly on the Overview page (this is cheap — it's just centrality on the graph already built — worth doing if time allows).

**Future Work (explicitly out of the 4-week build):**
- Incremental indexing on commit/webhook.
- Vector/semantic search layer.
- True call-graph resolution.
- Multi-language support beyond Python/TS/JS.
- "Learning View" guided onboarding paths.
- PR-diff-aware issue analysis.
- Multi-user / team deployment with auth and permissions.
- Optional graph database if repo/graph scale outgrows in-memory NetworkX.

---

## 18. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Local model quality is inconsistent at tool-selection and JSON-formatting tasks | Broken agent flow, malformed plans | Constrain Planner output with a strict JSON schema + retry-on-parse-failure; keep prompts small and single-purpose; test early against the actual target models, not assumptions about them |
| Import/reference resolution is noisier than expected (dynamic imports, re-exports, monorepo path aliases) | Misleading graph, wrong answers | Explicitly flag unresolved imports as "external/unresolved" rather than guessing; document this as a known limitation rather than silently failing |
| Four weeks is optimistic even with descoping | Incomplete MVP | Week-by-week goals in §17 are ordered so that each week ends with something demoable; if behind, cut from "Should Have" first, then Week 4 polish, never from Weeks 1–2 core pipeline |
| ripgrep-based reference index produces false positives (common names) | Cluttered/confusing "references" results | Cap result counts, sort by proximity/relevance heuristics (same module first), label clearly as "textual references" not "confirmed usages" in the UI |
| Solo developer, no code review | Bugs ship, architecture drift | Keep tools/agents small and independently testable (§10 outputs are structured JSON specifically so they're easy to unit test without an LLM in the loop) |
| GitHub API rate limits (unauthenticated or low-quota token) | Metadata sync fails/partial | Metadata sync is optional and cached at index time only, not on the request path; fail gracefully and mark as unavailable rather than blocking indexing |

---

## 19. Technical Tradeoffs

- **Relational store + in-memory graph vs. dedicated graph database:** chosen for zero added infrastructure at MVP scale; revisit only if graphs grow beyond comfortable in-memory size (see §8).
- **Reference graph vs. true call graph:** chosen because correctness of a real call graph is unattainable solo in a month; an honestly-labeled reference graph is more useful than a mislabeled, subtly-wrong call graph.
- **No embeddings/vector search vs. semantic search:** chosen because symbol/text/import search already answers most "where is X" questions with zero hallucination risk, and adding a vector DB is meaningful setup and maintenance cost for a capability that mostly matters at multi-repo or fuzzy-query scale, which is out of scope.
- **Full re-index vs. incremental:** chosen because incremental correctness (knowing exactly what changed and cascading graph updates) is a harder problem than the indexing pipeline itself; full re-index is slower but trivially correct.
- **4 agents vs. 5+:** chosen to reduce local-LLM round trips per query, directly improving latency and reducing compound failure probability, at the cost of the Context agent being slightly less single-purpose than a maximally "clean" design would prefer.
- **REST over WebSockets for chat:** chosen for simplicity; streaming intermediate steps (a nicer UX) is explicitly a Should-Have, not required for MVP correctness, and REST polling/response is far less to build and debug solo.

---

## 20. Future Improvements

- Incremental indexing via git webhook/commit-hook triggers.
- Optional semantic search layer (local embedding model, e.g. a small sentence-transformer run via `sentence-transformers` on CPU) once the deterministic layer's limitations are well understood in practice.
- True call-graph construction for Python (feasible with more time via static analysis of `ast`, harder for JS due to dynamic dispatch).
- Multi-repo support with cross-repo reference resolution (e.g., monorepo packages, or a company's internal package ecosystem).
- PR-diff-aware "what does this change affect" analysis, building on the reference graph once it's proven reliable.
- Guided "Learning View" — a curated, ordered tour of a repo for new hires, generated once from the architecture graph's centrality/entry-point heuristics.
- Optional migration to a dedicated graph database if repo scale demands it.
- Team deployment: auth, per-user chat history, shared repo indexes across an organization.

---

## Closing Note

Every cut in §0 was made to protect the same thing: a working, honest, locally-run product at the end of four weeks, rather than a partially-working ambitious one. The riskiest technical bet in the original brief — the call graph — has been replaced with something weaker but real and clearly labeled as such. Everything else follows the brief's own stated philosophy: the intelligence comes from deterministic engineering, and the LLM's only job is to orchestrate and explain.
