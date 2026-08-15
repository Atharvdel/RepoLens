"""RepoLens indexing pipeline (SDD §7).

Stage-by-stage implementation of the one-shot batch indexing flow described in
SDD §7: file walker, Tree-sitter parsers (Python, TS, JS), import graph,
ripgrep reference index, document parser, git history scan, GitHub sync,
and the unified pipeline orchestrator.
"""
from app.indexing.doc_indexer import index_documents, parse_documents
from app.indexing.git_history import extract_commits_and_touches, index_git_history
from app.indexing.github_sync import fetch_github_metadata, sync_github_metadata
from app.indexing.import_graph import (
    ImportResolver,
    IndexResult,
    ParsedImport,
    index_file_imports,
    parse_and_index_imports,
    parse_imports,
)
from app.indexing.parser import (
    ParsedSymbol,
    index_file_symbols,
    parse_and_index_file,
    parse_file,
)
from app.indexing.pipeline import (
    PipelineProgress,
    clear_repository_data,
    run_indexing_pipeline,
)
from app.indexing.reference_index import (
    DEFAULT_CAP,
    ReferenceHit,
    ReferenceIndexResult,
    find_and_index_references,
    find_references,
    index_symbol_references,
    ripgrep_available,
)
from app.indexing.ts_js_parser import (
    parse_ts_js_file_imports,
    parse_ts_js_file_symbols,
)
from app.indexing.walker import walk_repository

__all__ = [
    "DEFAULT_CAP",
    "ImportResolver",
    "IndexResult",
    "ParsedImport",
    "ParsedSymbol",
    "PipelineProgress",
    "ReferenceHit",
    "ReferenceIndexResult",
    "clear_repository_data",
    "extract_commits_and_touches",
    "fetch_github_metadata",
    "find_and_index_references",
    "find_references",
    "index_documents",
    "index_file_imports",
    "index_file_symbols",
    "index_git_history",
    "index_symbol_references",
    "parse_and_index_file",
    "parse_and_index_imports",
    "parse_documents",
    "parse_file",
    "parse_imports",
    "parse_ts_js_file_imports",
    "parse_ts_js_file_symbols",
    "ripgrep_available",
    "run_indexing_pipeline",
    "sync_github_metadata",
    "walk_repository",
]
