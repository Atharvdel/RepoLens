"""RepoLens indexing pipeline (SDD §7).

Stage-by-stage implementation of the one-shot batch indexing flow described in
SDD §7. Modules land incrementally as the pipeline is built: the file-walker
(§7 steps 1–2), the tree-sitter symbol parser (§7 step 3), the import graph
(§7 step 4), and the reference index (§7 step 5) exist today; git history and
GitHub metadata arrive in later steps.
"""
from app.indexing.import_graph import (
    IndexResult,
    ImportResolver,
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
from app.indexing.reference_index import (
    DEFAULT_CAP,
    ReferenceHit,
    ReferenceIndexResult,
    find_and_index_references,
    find_references,
    index_symbol_references,
    ripgrep_available,
)
from app.indexing.walker import walk_repository

__all__ = [
    "DEFAULT_CAP",
    "IndexResult",
    "ImportResolver",
    "ParsedImport",
    "ParsedSymbol",
    "ReferenceHit",
    "ReferenceIndexResult",
    "find_and_index_references",
    "find_references",
    "index_file_imports",
    "index_file_symbols",
    "index_symbol_references",
    "parse_and_index_file",
    "parse_and_index_imports",
    "parse_file",
    "parse_imports",
    "ripgrep_available",
    "walk_repository",
]
