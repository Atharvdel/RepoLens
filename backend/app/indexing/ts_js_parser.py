"""Tree-sitter parser for TypeScript and JavaScript files (SDD §0, §7 step 3, step 4).

Extracts top-level symbols (classes, functions, methods, exported arrow functions)
and import statements from `.ts`, `.tsx`, `.js`, and `.jsx` files.

Uses `tree-sitter-typescript` and `tree-sitter-javascript`.
Conforms to the `ParsedSymbol` and `ImportStatement` interfaces used by
`parser.py` and `import_graph.py`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from dataclasses import dataclass

import tree_sitter_javascript as ts_js
import tree_sitter_typescript as ts_ts
from tree_sitter import Language, Node, Parser

from app.indexing.parser import ParsedSymbol


@dataclass
class TSImport:
    module: str
    is_relative: bool
    line_number: int
    level: int = 0

# Initialize Languages & Parsers
_JS_LANG = Language(ts_js.language())
_TS_LANG = Language(ts_ts.language_typescript())
_TSX_LANG = Language(ts_ts.language_tsx())

_PARSER_JS = Parser(_JS_LANG)
_PARSER_TS = Parser(_TS_LANG)
_PARSER_TSX = Parser(_TSX_LANG)


def _get_parser_for_path(path: Path | str) -> Parser:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".tsx":
        return _PARSER_TSX
    elif suffix == ".ts":
        return _PARSER_TS
    elif suffix in {".jsx", ".js", ".mjs", ".cjs"}:
        return _PARSER_JS
    return _PARSER_TS


def _line(node: Node, end: bool = False) -> int:
    point = node.end_point if end else node.start_point
    return point[0] + 1


def _clean_docstring(comment: str) -> str:
    """Clean JSDoc /** ... */ comments."""
    s = comment.strip()
    if s.startswith("/**") and s.endswith("*/"):
        lines = s[3:-2].splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("*"):
                stripped = stripped[1:].strip()
            cleaned.append(stripped)
        return "\n".join(cleaned).strip()
    return s


def _extract_leading_comment(node: Node, source_bytes: bytes) -> str | None:
    """Look for preceding comment or JSDoc sibling node."""
    curr: Node | None = node
    while curr:
        prev = curr.prev_sibling
        if prev and prev.type in {"comment", "jsdoc"}:
            return _clean_docstring(prev.text.decode("utf-8", errors="replace"))
        if curr.parent and curr.parent.type in {"export_statement", "export_default_declaration", "decorated_definition"}:
            curr = curr.parent
        else:
            break
    return None


def _make_method(node: Node, source_bytes: bytes) -> ParsedSymbol | None:
    """Extract a method_definition from a class body."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = name_node.text.decode("utf-8", errors="replace")
    doc = _extract_leading_comment(node, source_bytes)
    return ParsedSymbol(
        name=name,
        kind="method",
        line_start=_line(node),
        line_end=_line(node, end=True),
        docstring=doc,
        methods=[],
    )


def _make_class(node: Node, source_bytes: bytes) -> ParsedSymbol | None:
    """Extract a class_declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = name_node.text.decode("utf-8", errors="replace")
    doc = _extract_leading_comment(node, source_bytes)

    body = node.child_by_field_name("body")
    methods: list[ParsedSymbol] = []
    if body:
        for child in body.children:
            if child.type == "method_definition":
                m = _make_method(child, source_bytes)
                if m:
                    methods.append(m)

    return ParsedSymbol(
        name=name,
        kind="class",
        line_start=_line(node),
        line_end=_line(node, end=True),
        docstring=doc,
        methods=methods,
    )


def _make_function(node: Node, source_bytes: bytes) -> ParsedSymbol | None:
    """Extract a function_declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = name_node.text.decode("utf-8", errors="replace")
    doc = _extract_leading_comment(node, source_bytes)
    return ParsedSymbol(
        name=name,
        kind="function",
        line_start=_line(node),
        line_end=_line(node, end=True),
        docstring=doc,
        methods=[],
    )


def _unwrap_export_or_decorator(node: Node) -> Node:
    if node.type in {"export_statement", "export_default_declaration"}:
        for child in node.children:
            if child.type in {"class_declaration", "function_declaration", "lexical_declaration", "variable_declaration"}:
                return child
    return node


def parse_ts_js_symbols_from_source(source_bytes: bytes, filename: str = "file.ts") -> list[ParsedSymbol]:
    """Parse TypeScript/JavaScript source bytes and extract top-level symbols."""
    parser = _get_parser_for_path(filename)
    tree = parser.parse(source_bytes)
    symbols: list[ParsedSymbol] = []

    for child in tree.root_node.children:
        unwrapped = _unwrap_export_or_decorator(child)
        if unwrapped.type == "class_declaration":
            cls = _make_class(unwrapped, source_bytes)
            if cls:
                symbols.append(cls)
        elif unwrapped.type in {"function_declaration", "generator_function_declaration"}:
            fn = _make_function(unwrapped, source_bytes)
            if fn:
                symbols.append(fn)
        elif unwrapped.type in {"lexical_declaration", "variable_declaration"}:
            # Check for top-level `const fn = () => {}` or `const fn = function() {}`
            for decl in unwrapped.children:
                if decl.type == "variable_declarator":
                    name_node = decl.child_by_field_name("name")
                    value_node = decl.child_by_field_name("value")
                    if name_node and value_node and value_node.type in {"arrow_function", "function_expression"}:
                        name = name_node.text.decode("utf-8", errors="replace")
                        doc = _extract_leading_comment(child, source_bytes)
                        symbols.append(
                            ParsedSymbol(
                                name=name,
                                kind="function",
                                line_start=_line(decl),
                                line_end=_line(decl, end=True),
                                docstring=doc,
                                methods=[],
                            )
                        )

    return symbols


def parse_ts_js_file_symbols(path: Path | str) -> list[ParsedSymbol]:
    """Read a TS/JS file and parse its top-level symbols."""
    p = Path(path).resolve()
    try:
        content = p.read_bytes()
        return parse_ts_js_symbols_from_source(content, filename=p.name)
    except Exception:
        return []


def _clean_string_literal(raw: str) -> str:
    s = raw.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')) or (s.startswith('`') and s.endswith('`')):
        return s[1:-1]
    return s


def parse_ts_js_imports_from_source(source_bytes: bytes, filename: str = "file.ts") -> list[TSImport]:
    """Parse import statements and require calls from TS/JS source."""
    parser = _get_parser_for_path(filename)
    tree = parser.parse(source_bytes)
    imports: list[TSImport] = []

    def _walk_node(node: Node):
        if node.type == "import_statement":
            # e.g. import ... from 'source'
            source_node = node.child_by_field_name("source")
            if source_node:
                raw_mod = source_node.text.decode("utf-8", errors="replace")
                cleaned = _clean_string_literal(raw_mod)
                if cleaned:
                    is_rel = False
                    level = 0
                    if cleaned.startswith("@/"):
                        is_rel = True
                        level = -1
                    elif cleaned.startswith("./") or cleaned.startswith("../") or cleaned == "." or cleaned == "..":
                        is_rel = True
                        parts = cleaned.split("/")
                        up_count = sum(1 for p in parts if p == "..")
                        level = 1 + up_count

                    imports.append(
                        TSImport(
                            module=cleaned,
                            is_relative=is_rel,
                            line_number=_line(node),
                            level=level,
                        )
                    )
        elif node.type == "call_expression":
            # e.g. require('./foo')
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn and fn.text == b"require" and args and args.children:
                for arg in args.children:
                    if arg.type == "string":
                        cleaned = _clean_string_literal(arg.text.decode("utf-8", errors="replace"))
                        if cleaned:
                            is_rel = False
                            level = 0
                            if cleaned.startswith("@/"):
                                is_rel = True
                                level = -1
                            elif cleaned.startswith("./") or cleaned.startswith("../") or cleaned == "." or cleaned == "..":
                                is_rel = True
                                parts = cleaned.split("/")
                                up_count = sum(1 for p in parts if p == "..")
                                level = 1 + up_count

                            imports.append(
                                TSImport(
                                    module=cleaned,
                                    is_relative=is_rel,
                                    line_number=_line(node),
                                    level=level,
                                )
                            )

        for child in node.children:
            _walk_node(child)

    _walk_node(tree.root_node)
    return imports


def parse_ts_js_file_imports(path: Path | str) -> list[TSImport]:
    """Read a TS/JS file and parse its imports."""
    p = Path(path).resolve()
    try:
        content = p.read_bytes()
        return parse_ts_js_imports_from_source(content, filename=p.name)
    except Exception:
        return []
