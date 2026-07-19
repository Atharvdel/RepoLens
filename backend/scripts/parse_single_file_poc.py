"""THROWAWAY tree-sitter proof-of-concept (SDD §7 step 4: parsing).

Not part of the application — a single-file harness proving the tree-sitter
pipeline against one real .py file before we generalize it into
``app/indexing/parser.py``. Touches NO database: no SQLAlchemy, no session,
no models. Pure parse-and-print, so we can eyeball the output against the
source file and decide whether tree-sitter-python is trustworthy here.

What it extracts, recursively through nesting:

* class names  — ``class`` definitions
* function/method names — ``def`` definitions (a ``def`` directly inside a
  ``class`` block is labeled ``method``; everything else is ``function``)
* line ranges — the definition *body* span (``def``/``class`` line through
  the last line of its block), 1-indexed inclusive, exactly what an editor
  shows. Decorator lines are reported separately because tree-sitter keeps
  them as siblings of the definition, not part of its span.
* docstrings — the first statement of a block that is an ``expression_statement``
  made up solely of ``string`` literals (covers implicit-concatenation docstrings
  too). The raw text is shown, newline-collapsed and truncated.
* decorators — each ``@...`` preceding a def/class, with its own line range.

Run from the ``backend/`` directory with the project venv::

    .venv/Scripts/python scripts/parse_single_file_poc.py
    .venv/Scripts/python scripts/parse_single_file_poc.py path/to/other.py

With no argument it defaults to the file this POC is scoped to prove:
``C:\\Users\\Atharv Sharma\\Desktop\\Work\\flask\\src\\flask\\app.py``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser

# Default target (overridable via argv[1]) — the file this POC must prove.
DEFAULT_TARGET = r"C:\Users\Atharv Sharma\Desktop\Work\flask\src\flask\app.py"

# The two definition node types we surface; everything else is structural noise.
DEFINITION_TYPES = {"class_definition", "function_definition"}

# A def whose body lives directly inside a class block is a method, not a
# function. Detected by inspecting the *def's* parent node type.
CLASS_BODY = "block"  # the field "body" of a class/function is always a `block`


@dataclass
class Decorator:
    """One ``@name(...)`` above a definition. Range is its own line span."""
    text: str
    start_line: int
    end_line: int


@dataclass
class Symbol:
    """A class, function, or method extracted from the tree."""
    kind: str          # "class" | "method" | "function"
    name: str
    start_line: int   # 1-indexed, inclusive — the def/class line
    end_line: int     # 1-indexed, inclusive — last line of the body
    decorators: list[Decorator] = field(default_factory=list)
    docstring_text: Optional[str] = None
    docstring_range: Optional[tuple[int, int]] = None
    children: list[Symbol] = field(default_factory=list)


def _line(node: Node, end: bool = False) -> int:
    """1-indexed line of a node (row 0 in tree-sitter == line 1 in an editor)."""
    point = node.end_point if end else node.start_point
    return point[0] + 1


def _docstring_of(block: Optional[Node], src: bytes) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """Pull the docstring out of a class/function body block, if it has one.

    A docstring is the block's first statement AND that statement is an
    ``expression_statement`` made up exclusively of ``string`` literals (this
    excludes ``"x" = something`` style oddities and statements that merely
    start with a string). Implicit-concatenation docstrings (several adjacent
    string literals in one expression) are handled because *every* child of the
    expression must be a string.
    """
    if block is None:
        return None, None
    kids = list(block.children)
    if not kids or kids[0].type != "expression_statement":
        return None, None
    inner = list(kids[0].children)
    if not inner or not all(c.type == "string" for c in inner):
        return None, None
    raw = b"".join(c.text for c in inner).decode("utf-8", errors="replace")
    return raw, (_line(kids[0]), _line(kids[0], end=True))


def _decorators_of(node: Node) -> list[Decorator]:
    """Decorators attached to a definition that's wrapped in a
    ``decorated_definition``. Returns ``[]`` for a bare (undecorated) def."""
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return []
    out: list[Decorator] = []
    for child in parent.named_children:
        if child.type == "decorator":
            out.append(
                Decorator(
                    text=child.text.decode("utf-8", errors="replace").strip(),
                    start_line=_line(child),
                    end_line=_line(child, end=True),
                )
            )
    return out


def _classify(parent: Optional[Node]) -> str:
    """``method`` when the def's parent is a class body, else ``function``."""
    if parent is not None and parent.type == "block":
        grand = parent.parent
        if grand is not None and grand.type == "class_definition":
            return "method"
    return "function"


def _extract(node: Node, src: bytes) -> Optional[Symbol]:
    """Turn one ``function_definition`` / ``class_definition`` node into a
    Symbol, recursing into its body for nested classes, methods, and
    inner functions. Returns ``None`` for anything that isn't a definition."""
    if node.type not in DEFINITION_TYPES:
        return None

    kind = "class" if node.type == "class_definition" else _classify(node.parent)
    name_node = node.child_by_field_name("name")
    body = node.child_by_field_name("body")
    name = name_node.text.decode("utf-8", errors="replace") if name_node is not None else "<anon>"

    doc_text, doc_range = _docstring_of(body, src)

    sym = Symbol(
        kind=kind,
        name=name,
        start_line=_line(node),
        end_line=_line(node, end=True),
        decorators=_decorators_of(node),
        docstring_text=doc_text,
        docstring_range=doc_range,
    )

    # Recurse: only nested defs/classes matter; decorators get re-extracted
    # from their own ``decorated_definition`` wrapper if present.
    if body is not None:
        for child in body.children:
            target = child
            if child.type == "decorated_definition":
                defs = [c for c in child.named_children if c.type in DEFINITION_TYPES]
                target = defs[0] if defs else None
            if target is not None:
                inner = _extract(target, src)
                if inner is not None:
                    sym.children.append(inner)
    return sym


def _flatten(sym: Symbol, depth: int = 0) -> list[tuple[int, Symbol]]:
    """Indentation-ordered (depth, symbol) pairs for tree-style printing."""
    out = [(depth, sym)]
    for child in sym.children:
        out.extend(_flatten(child, depth + 1))
    return out


def _collapse(text: str, width: int = 78) -> str:
    """Make a docstring printable on one line: strip quotes, squeeze
    whitespace, truncate."""
    s = text.strip()
    for q in ('"""', "'''"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            s = s[len(q):-len(q)]
            break
    single = " ".join(s.split())  # collapse all runs of whitespace incl. newlines
    if len(single) > width:
        single = single[: width - 3] + "..."
    return single


def _print_symbol(depth: int, sym: Symbol) -> None:
    indent = "  " * depth
    rng = f"L{sym.start_line}-{sym.end_line}"
    print(f"{indent}{sym.kind:<7} {sym.name:<28} {rng}")
    for dec in sym.decorators:
        print(f"{indent}    @{dec.text}  (L{dec.start_line})")
    if sym.docstring_text:
        ds, de = sym.docstring_range
        preview = _collapse(sym.docstring_text)
        print(f"{indent}    docstring L{ds}-{de}: {preview!r}")
    else:
        print(f"{indent}    docstring: (none)")


def parse_file(path: Path) -> tuple[list[Symbol], int, int, int]:
    """Parse ``path`` with tree-sitter-python; return ``(symbols, loc,
    error_count, total_symbols)``. Symbols are the top-level module-level
    classes/functions; nested ones hang off ``.children``."""
    src = path.read_bytes()
    language = Language(tsp.language())
    parser = Parser(language)
    tree = parser.parse(src)
    root = tree.root_node

    # Count ERROR nodes anywhere in the tree — a non-zero count means the
    # grammar couldn't make sense of part of the file, which is exactly what
    # we want surfaced before trusting the extracted symbols.
    error_count = 0
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "ERROR" or n.is_missing:
            error_count += 1
        stack.extend(n.children)

    symbols: list[Symbol] = []
    for child in root.children:
        target = child
        if child.type == "decorated_definition":
            defs = [c for c in child.named_children if c.type in DEFINITION_TYPES]
            target = defs[0] if defs else None
        if target is not None:
            sym = _extract(target, src)
            if sym is not None:
                symbols.append(sym)

    loc = len(src.decode("utf-8", errors="replace").splitlines())
    total = sum(len(list(_flatten(s))) for s in symbols)
    return symbols, loc, error_count, total


def main(argv: list[str]) -> int:
    # Force UTF-8 output so non-ASCII in docstrings survives the (often
    # cp1252/cp437) Windows console. Errors="replace" keeps a stray byte from
    # ever crashing the print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass  # older runtime / redirected stream — best-effort, ignore
    target = Path(argv[1]) if len(argv) > 1 else Path(DEFAULT_TARGET)
    if not target.is_file():
        print(f"ERROR: target is not a file: {target}", file=sys.stderr)
        return 1

    symbols, loc, error_count, total = parse_file(target)

    # Header — everything you need to orient before reading the symbol list.
    rel = target.as_posix()
    n_class = sum(1 for s in _iter_all(symbols) if s.kind == "class")
    n_method = sum(1 for s in _iter_all(symbols) if s.kind == "method")
    n_func = sum(1 for s in _iter_all(symbols) if s.kind == "function")
    print(f"=== tree-sitter-python parse: {rel} ===")
    print(f"source : {loc} lines  |  parse errors: {error_count}")
    print(
        f"symbols: {total} total "
        f"({n_class} class, {n_method} method, {n_func} function incl. nested)"
    )
    print("(nothing written to DB -- this is a proof-of-concept)")
    print()

    for sym in symbols:
        for depth, s in _flatten(sym):
            _print_symbol(depth, s)
    print()
    print(f"=== done: {total} symbols, {error_count} parse errors ===")
    return 0


def _iter_all(symbols: list[Symbol]):
    for s in symbols:
        for _, s2 in _flatten(s):
            yield s2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
