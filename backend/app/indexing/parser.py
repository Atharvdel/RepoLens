"""Tree-sitter parsing stage of the RepoLens indexing pipeline (SDD §7 step 3).

Parses a single Python source file with ``tree-sitter-python`` and extracts the
symbols that matter for navigation and symbol search:

* **top-level classes**  (``class`` at module level)  → ``kind="class"``
* **top-level functions** (``def`` / ``async def`` at module level) → ``kind="function"``
* **methods** (a ``def`` / ``async def`` *directly* inside a class body) → ``kind="method"``,
  linked to its enclosing class via ``parent_symbol_id``

and *nothing else*. Concretely, **functions nested inside other functions are
skipped** — they are never written to the ``symbols`` table. Nested classes and
class-/function-local assignments are out of scope here too (the SDD lists
top-level variables, imports, and reference-graph edges as separate, later
pipeline steps; this module is symbols-only).

Scoping rationale (SDD §7 step 3 / §10 "Repository Parser"): the symbol index
answers "where is X defined / which methods does this class have". A helper
``def`` buried inside a method body is implementation detail, not an addressable
entry point, so including it would just add noise to search results. We also
stop at exactly one level of nesting (class → method): a ``def`` nested inside a
method, or inside a top-level function, is not extracted.

Two cooperating pieces:

* :func:`parse_file`  — *pure*: reads bytes from disk, returns a list of
  :class:`ParsedSymbol` dataclass instances. No DB, no session. This is the
  unit-testable core and the SDD §10 "Repository Parser" tool surface
  (``{symbols: [{name, kind, line_start, line_end, docstring}]}``).
* :func:`index_file_symbols` — *persists*: takes a parsed list plus a ``file_id``
  and a :class:`~sqlalchemy.orm.Session`, adds ``Symbol`` rows, and returns the
  count added. It does **not commit** — the caller owns the transaction, exactly
  like :func:`app.indexing.walker.walk_repository` (SDD §7 step 9: persist &
  mark ready as one atomic unit). It *may* ``flush`` to assign primary keys so
  methods can foreign-key back to their enclosing class row via
  ``parent_symbol_id``; flushing is transactional and rolls back with the
  caller's session.

:func:`parse_and_index_file` glues the two together for the future
``pipeline.py`` orchestrator.

``file_id`` must reference a ``files`` row that is already *persisted* (flushed
or committed) at call time — the ``symbols.file_id`` FK is non-deferrable, so a
still-pending-in-session ``File`` without an id would violate it on flush. The
realistic pipeline commits the walker's ``File`` rows first, then parses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_python as tsp
from sqlalchemy.orm import Session
from tree_sitter import Language, Node, Parser

from app.models import Symbol

# The two definition node kinds tree-sitter-python exposes that we surface as
# symbols. Everything else under a module/class body (assignments, imports,
# comments, expression statements, ``if __name__ == ...`` blocks) is structural
# noise we do not index here.
DEFINITION_TYPES: frozenset[str] = frozenset({"class_definition", "function_definition"})

# Node type that wraps a decorated def/class so we can peel the decorator(s) off
# and get at the real definition underneath. Decorators themselves are dropped:
# the ``symbols`` table has no decorator column (SDD §11), so a method's
# ``line_start`` is its ``def`` line, not its (optional) decorator line(s).
DECORATED_WRAPPER = "decorated_definition"


@dataclass
class ParsedSymbol:
    """One extracted symbol, in the shape the SDD §10 parser tool emits.

    ``methods`` is populated only for ``kind == "class"`` (the methods defined
    directly in that class body); top-level functions have an empty ``methods``
    list. The list is carried structurally (not by name) so the writer can link
    each method row to its enclosing class row by id without any name-matching
    that could collide on duplicate class names.
    """

    name: str
    kind: str  # "class" | "function" | "method"
    line_start: int
    line_end: int
    docstring: Optional[str]
    methods: list["ParsedSymbol"] = field(default_factory=list)


def _line(node: Node, end: bool = False) -> int:
    """1-indexed inclusive line of ``node`` (tree-sitter row 0 == editor line 1).

    Used for both ``line_start`` (the ``def``/``class`` line) and ``line_end``
    (the last line of the body) — exactly the span an editor highlights for a
    definition. Decorator lines are not included (they wrap the def, sibling to
    it in the tree, so they fall outside this node's range by construction).
    """
    point = node.end_point if end else node.start_point
    return point[0] + 1


def _name_of(node: Node) -> str:
    """The identifier of a ``def``/``class`` node, decoded as UTF-8.

    Every well-formed Python definition has a ``name`` field; the ``<anon>``
    fallback only guards against a malformed tree and is not expected in
    practice (the ``name`` column is NOT NULL, so we always emit something).
    """
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return "<anon>"
    return name_node.text.decode("utf-8", errors="replace")


def _clean_docstring(raw: str) -> str:
    """Strip one surrounding triple-quote pair from a docstring literal.

    Keeps the internal prose (newlines, indentation) verbatim — downstream
    search/tooling sees the actual docstring text rather than Python's string
    delimiters. A docstring built from implicit string concatenation (``"a"
    "b"``) has no single wrapping pair; for that rare case we return the raw
    joined text unchanged rather than guess at fragment boundaries.
    """
    s = raw.strip()
    for quote in ('"""', "'''"):
        if len(s) >= 2 * len(quote) and s.startswith(quote) and s.endswith(quote):
            return s[len(quote) : -len(quote)]
    return s


def _docstring_of(block: Optional[Node]) -> Optional[str]:
    """Return the docstring text of a class/function body block, if it has one.

    A docstring is the block's *first* statement, and that statement must be an
    ``expression_statement`` made up exclusively of ``string`` literals. The
    "all children are strings" test (not just "starts with a string") correctly
    admits implicit-concatenation docstrings (several adjacent string literals
    in one expression) and rejects statements that merely begin with a string.
    """
    if block is None:
        return None
    kids = list(block.children)
    if not kids or kids[0].type != "expression_statement":
        return None
    inner = list(kids[0].children)
    if not inner or not all(c.type == "string" for c in inner):
        return None
    raw = b"".join(c.text for c in inner).decode("utf-8", errors="replace")
    return _clean_docstring(raw)


def _unwrap_decorated(child: Node) -> Optional[Node]:
    """If ``child`` is a ``decorated_definition``, return the def/class it
    wraps; otherwise return ``child`` unchanged. Returns ``None`` for a
    decorated node that wraps neither a def nor a class (e.g. a decorated
    assignment) so the caller can skip it cleanly."""
    if child.type == DECORATED_WRAPPER:
        for inner in child.named_children:
            if inner.type in DEFINITION_TYPES:
                return inner
        return None
    return child


def _make_function(node: Node, kind: str) -> ParsedSymbol:
    """Build a ``ParsedSymbol`` for a ``function_definition`` node. Does not
    recurse into the body — we never index nested defs (SDD scope above), so a
    top-level function and a method are extracted the same way, differing only
    in ``kind``."""
    doc = _docstring_of(node.child_by_field_name("body"))
    return ParsedSymbol(
        name=_name_of(node),
        kind=kind,
        line_start=_line(node),
        line_end=_line(node, end=True),
        docstring=doc,
        methods=[],
    )


def _make_class(node: Node) -> ParsedSymbol:
    """Build a ``ParsedSymbol`` for a ``class_definition``, recursing exactly
    one level into its body to collect **methods** — defs (including
    ``async def`` and decorated defs) sitting *directly* in the class block.

    Anything else in the class body is skipped on purpose: a nested class is
    not a method (and not top-level, so it is dropped entirely per scope), and
    class-level assignments / ``if`` blocks / comments are not symbols. We do
    *not* descend into the method bodies, so a ``def`` nested inside a method
    is never emitted.
    """
    body = node.child_by_field_name("body")
    methods: list[ParsedSymbol] = []
    if body is not None:
        for child in body.children:
            target = _unwrap_decorated(child)
            # Only ``function_definition`` (bare or decorated) counts as a
            # method. A nested ``class_definition`` here is *not* a method and
            # is skipped — nested classes are outside this module's scope.
            if target is None or target.type != "function_definition":
                continue
            methods.append(_make_function(target, kind="method"))
    return ParsedSymbol(
        name=_name_of(node),
        kind="class",
        line_start=_line(node),
        line_end=_line(node, end=True),
        docstring=_docstring_of(body),
        methods=methods,
    )


def parse_file(path: Path | str) -> list[ParsedSymbol]:
    """Parse one Python file with tree-sitter and return its symbols.

    *Pure*: reads bytes from ``path`` (so the file's own encoding is honored —
    tree-sitter works on raw bytes), runs the Python grammar, and returns a
    list of :class:`ParsedSymbol`. No DB, no session, no side effects.

    Top-level (module-level) definitions only, one level deep into classes for
    methods. Functions nested inside functions (top-level or method) and nested
    classes are **not** returned. Raises ``OSError`` if ``path`` cannot be
    read; the caller decides how loud to be about a vanished file.

    Tree-sitter is error-recovering — a syntactically broken file still yields
    a best-effort tree, which we extract from as-is. Dedicated parse-error
    reporting / skipping is a later pipeline-hardening step (SDD lists error
    handling as a later phase); this module does not special-case ``ERROR``
    nodes.
    """
    src = Path(path).read_bytes()
    language = Language(tsp.language())
    parser = Parser(language)
    tree = parser.parse(src)
    root = tree.root_node

    symbols: list[ParsedSymbol] = []
    for child in root.children:
        target = _unwrap_decorated(child)
        if target is None or target.type not in DEFINITION_TYPES:
            continue
        if target.type == "function_definition":
            symbols.append(_make_function(target, kind="function"))
        else:  # class_definition
            symbols.append(_make_class(target))
    return symbols


def index_file_symbols(
    file_id: int, symbols: list[ParsedSymbol], session: Session
) -> int:
    """Add ``Symbol`` rows for ``symbols`` to ``session``; return the count.

    Does **not commit** — the caller owns the transaction (mirrors
    :func:`app.indexing.walker.walk_repository`). May ``flush`` so that class
    rows are assigned primary keys before their methods foreign-key back via
    ``parent_symbol_id``; flushing is transactional and rolls back with the
    caller's session.

    ``file_id`` must belong to an already-persisted ``files`` row — the
    ``symbols.file_id`` FK is enforced on flush, so a still-pending ``File``
    would be rejected.

    Only the three scoped kinds are ever written here (``parse_file`` cannot
    produce anything else), so a function nested inside another function is
    simply absent from ``symbols`` and therefore absent from the table — by
    construction, not by a runtime filter.
    """
    added = 0
    for sym in symbols:
        if sym.kind == "class":
            class_row = Symbol(
                file_id=file_id,
                name=sym.name,
                kind="class",
                line_start=sym.line_start,
                line_end=sym.line_end,
                docstring=sym.docstring,
                parent_symbol_id=None,
            )
            session.add(class_row)
            # Flush so the class row gets its PK; methods FK to it by id. One
            # flush per class is negligible at MVP scale (handful of classes
            # per file) and avoids building a name→id map that could collide
            # on duplicate class names.
            session.flush()
            added += 1
            for method in sym.methods:
                session.add(
                    Symbol(
                        file_id=file_id,
                        name=method.name,
                        kind="method",
                        line_start=method.line_start,
                        line_end=method.line_end,
                        docstring=method.docstring,
                        parent_symbol_id=class_row.id,
                    )
                )
                added += 1
        else:  # top-level function
            session.add(
                Symbol(
                    file_id=file_id,
                    name=sym.name,
                    kind="function",
                    line_start=sym.line_start,
                    line_end=sym.line_end,
                    docstring=sym.docstring,
                    parent_symbol_id=None,
                )
            )
            added += 1
    return added


def parse_and_index_file(
    path: Path | str, file_id: int, session: Session
) -> int:
    """Convenience: parse ``path`` then persist its symbols to ``session``.

    Composes :func:`parse_file` + :func:`index_file_symbols` for the future
    ``pipeline.py`` orchestrator. Same transaction-ownership rule applies — this
    adds rows and may flush; the caller commits.
    """
    return index_file_symbols(file_id, parse_file(path), session)
