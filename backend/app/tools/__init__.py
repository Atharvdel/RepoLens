"""RepoLens agent tools (SDD §10).

Thin, deterministic, side-effect-free query functions the agent orchestrator
(SDD §9) calls name-by-name — never an LLM, never an agent framework *inside*
the tools themselves. Each returns structured JSON-shaped data (plain dicts via
``dataclasses.asdict``) so agents and the Synthesizer work from facts, not
paraphrased summaries (SDD §10: "structured JSON, never free text").

Layers in place:

* **Search tools** (the §9.2 Search Agent dispatches these): Symbol Search +
  File Search are thin Postgres wrappers; Text Search is the ripgrep-backed free
  text tool — the inverse posture of the reference-index stage's whole-word
  symbol-name matching.
* **Context tools** (the §9.3 Context Agent dispatches these): two NetworkX
  graph tools over the indexed import edges (Architecture Query — module map +
  key files by centrality, whole-repo or focused; Dependency Graph Query —
  BFS neighborhood of a file) and two cached-metadata reads (History Search
  over ``commits``/``file_commits``; GitHub Metadata Loader over ``issues``).
  The graph tools are live against the flask import graph today; the two
  metadata tools read tables that exist in the committed schema but are not
  yet populated by the §7 git-history / PyGithub indexing steps — they return a
  clear empty until those stages land, and are proven correct by synthetic-DB
  tests.

All share the injected-``Session`` / owning-no-transaction posture of the
indexing ``index_*`` stages (SDD §7 step 9): a tool reads, returns, and commits
nothing. The graph/metadata tools additionally split a pure-core aggregation
(NetworkX BFS, commit aggregation, issues/PRs partition) from the live wrapper
that reads Postgres — the project's pure / live convention mirrored from the
indexing stages and :mod:`app.tools.text_search`.
"""
from app.tools.architecture import (
    ArchitectureResult,
    query_architecture,
)
from app.tools.dependency_graph import (
    DependencyGraphResult,
    query_dependency_graph,
)
from app.tools.file_history import (
    FileHistoryResult,
    query_file_history,
)
from app.tools.file_search import FileResult, search_files
from app.tools.github_metadata import (
    GitHubMetadataResult,
    query_github_metadata,
)
from app.tools.symbol_search import SymbolResult, search_symbols
from app.tools.text_search import TextHit, search_text

__all__ = [
    "ArchitectureResult",
    "DependencyGraphResult",
    "FileHistoryResult",
    "FileResult",
    "GitHubMetadataResult",
    "SymbolResult",
    "TextHit",
    "query_architecture",
    "query_dependency_graph",
    "query_file_history",
    "query_github_metadata",
    "search_files",
    "search_symbols",
    "search_text",
]
