"""Tests for the tree-sitter symbol parser (SDD §7 step 3).

Two layers, mirroring the module's pure-parse / persist split:

* ``test_parse_file_*`` — pure, no DB. Exercise :func:`parse_file` against a
  crafted source file and assert the **scope rule** the parser must enforce:
  top-level classes, top-level functions, and methods (defs directly inside a
  class body) are extracted; everything nested inside a function (or inside a
  method) and any nested class is dropped. These run anywhere — no Postgres.
* ``test_index_file_symbols_*`` — live-Postgres round-trip (same container the
  smoke test uses). Confirms the writer links methods to their class row via
  ``parent_symbol_id`` and that the nested-out names never reach the table.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_parser.py -v
"""
import sqlalchemy as sa

from app.db import SessionLocal
from app.indexing import index_file_symbols, parse_file
from app.models import File, Repository, Symbol

# A single source file exercising every scoping decision at once. Line numbers
# are not asserted literally (brittle); what matters is *which* names surface.
SRC = '''"""Module docstring at top — not a symbol."""


def top_func(a):
    """Top-level function docstring."""
    def inner():
        """Nested inside a function — must NOT be indexed."""
        return 1
    return inner


class Widget:
    """Widget class docstring."""

    attr = 1  # class-level assignment — not a symbol

    def do(self):
        """Method docstring."""
        def helper():
            """Nested inside a method — must NOT be indexed."""
            return 2
        return helper

    class Nested:
        """Nested class — not top-level, not a method — must NOT be indexed."""
        pass


@some_decorator
def dec_func():
    """Decorated top-level function — decorator dropped, function kept."""
    pass
'''


def _all_names(symbols) -> set[str]:
    """Every name the parser surfaced — top-level symbols plus their methods.
    Used to assert forbidden (nested) names are absent everywhere."""
    names: set[str] = set()
    for sym in symbols:
        names.add(sym.name)
        for method in sym.methods:
            names.add(method.name)
    return names


def test_parse_file_extracts_only_top_level_and_methods(tmp_path):
    """The scope rule: top-level function + top-level class + its methods are
    returned; a nested function, a function nested in a method, and a nested
    class are all omitted."""
    path = tmp_path / "widget.py"
    path.write_text(SRC, encoding="utf-8")

    symbols = parse_file(path)

    # Three top-level symbols, in source order.
    top_names = [s.name for s in symbols]
    assert top_names == ["top_func", "Widget", "dec_func"], top_names

    top_func, widget, dec_func = symbols

    # ── top-level function ─────────────────────────────────────────────────
    assert top_func.kind == "function"
    assert top_func.methods == []
    assert top_func.docstring == "Top-level function docstring."
    assert top_func.line_start <= top_func.line_end

    # ── top-level class ────────────────────────────────────────────────────
    assert widget.kind == "class"
    assert widget.docstring == "Widget class docstring."
    # Exactly one method; the class-level assignment, the nested class, and the
    # method-local def must all be absent from .methods.
    method_names = [m.name for m in widget.methods]
    assert method_names == ["do"], method_names
    do = widget.methods[0]
    assert do.kind == "method"
    assert do.docstring == "Method docstring."
    assert do.line_start > widget.line_start  # method is inside the class span

    # ── decorated top-level function ───────────────────────────────────────
    assert dec_func.kind == "function"
    assert dec_func.methods == []
    assert dec_func.docstring == (
        "Decorated top-level function — decorator dropped, function kept."
    )

    # ── forbidden (nested) names must be nowhere ───────────────────────────
    all_names = _all_names(symbols)
    for forbidden in ("inner", "helper", "Nested"):
        assert forbidden not in all_names, (
            f"{forbidden!r} surfaced — nested symbol was not skipped: {all_names}"
        )


def test_parse_file_drops_decorators_from_line_span(tmp_path):
    """A decorated function's ``line_start`` is its ``def`` line, not the
    decorator line — decorators are siblings of the def in tree-sitter, so they
    fall outside the definition node's range. ``dec_func`` starts on its ``def``
    line (the 2nd line of its two-line decorated block)."""
    path = tmp_path / "dec.py"
    path.write_text("@some_decorator\ndef dec_func():\n    pass\n", encoding="utf-8")
    (sym,) = parse_file(path)
    assert sym.name == "dec_func"
    assert sym.kind == "function"
    assert sym.line_start == 2  # the `def` line, not the `@` line


def test_index_file_symbols_links_methods_to_class(tmp_path):
    """Live-DB round-trip: parse the crafted file, persist its symbols, read
    them back in a fresh session. Methods must carry ``parent_symbol_id``
    pointing at their class row; top-level functions/classes have NULL; the
    nested-out names must be absent from the table."""
    path = tmp_path / "widget.py"
    path.write_text(SRC, encoding="utf-8")

    repo_id = None
    file_id = None
    try:
        # ── set up: a persisted Repository + File so there's a real file_id ─
        with SessionLocal() as session:
            repo = Repository(url_or_path="file:///test/parser", name="test/parser",
                              status="indexing")
            session.add(repo)
            session.commit()
            session.refresh(repo)
            repo_id = repo.id

            f = File(repository_id=repo_id, path="fake/widget.py",
                     language="python", loc=SRC.count("\n"))
            session.add(f)
            session.commit()
            session.refresh(f)
            file_id = f.id

        # ── act: parse + persist symbols (caller commits) ──────────────────
        with SessionLocal() as session:
            symbols = parse_file(path)
            added = index_file_symbols(file_id, symbols, session)
            session.commit()
        # 4 rows: top_func, Widget, do (method), dec_func. Nested ones are absent.
        assert added == 4, added

        # ── read back through a fresh session (real DB hit, no identity-map) ─
        with SessionLocal() as session:
            rows = (
                session.execute(
                    sa.select(Symbol).where(Symbol.file_id == file_id)
                )
                .scalars()
                .all()
            )
            by_name = {r.name: r for r in rows}

            assert {r.name for r in rows} == {"top_func", "Widget", "do", "dec_func"}

            # Linkage: the class row has no parent; its method FKs to it.
            widget_row = by_name["Widget"]
            do_row = by_name["do"]
            assert widget_row.kind == "class"
            assert widget_row.parent_symbol_id is None
            assert do_row.kind == "method"
            assert do_row.parent_symbol_id == widget_row.id, (
                "method not linked to its class row via parent_symbol_id"
            )

            # Top-level functions: no parent, line ranges persisted.
            for fname in ("top_func", "dec_func"):
                row = by_name[fname]
                assert row.kind == "function"
                assert row.parent_symbol_id is None
                assert row.line_start <= row.line_end

            # The docstring landed stripped of its triple-quote delimiters.
            assert by_name["top_func"].docstring == "Top-level function docstring."

    finally:
        # Defensive cleanup: symbols → file → repo, so a failure never leaves
        # stray rows (mirrors the smoke test's cleanup posture).
        with SessionLocal() as session:
            if file_id is not None:
                session.execute(sa.delete(Symbol).where(Symbol.file_id == file_id))
                session.execute(sa.delete(File).where(File.id == file_id))
            if repo_id is not None:
                session.execute(sa.delete(Repository).where(Repository.id == repo_id))
            session.commit()
