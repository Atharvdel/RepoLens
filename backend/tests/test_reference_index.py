"""Tests for the reference-index stage (SDD Â§7 step 5).

Three layers, mirroring :mod:`tests.test_import_graph`:

* ``test_parse_ripgrep_output_*`` â€” *deterministic pure, no ripgrep, no DB*. The
  :func:`_parse_ripgrep_output` core is fed crafted ``path:line:col:text`` strings
  and asserts the parse, the leading ``./`` strip, the own-definition-line
  exclusion, malformed-line skipping, and the cap. These run anywhere â€” no
  ripgrep, no Postgres â€” and pin the logic independent of a ``rg`` binary.
* ``test_find_references_*`` â€” *real ripgrep, no DB* (skipped if ``rg`` is not on
  PATH). Drive :func:`find_references` end-to-end against a fixture repo on disk:
  a known symbol's references are found correctly (its own ``def``/``class`` line
  excluded; cross-file and same-file-non-binding references kept), and a common
  name repeated far past the cap does not blow past it.
* ``test_index_symbol_references_*`` â€” *live Postgres, no ripgrep*: the persist
  layer is fed crafted hits so it is exercised deterministically. Asserts one
  ``references`` edge per distinct referencing file (multiple hits in one file
  collapse), ``source_type="symbol"`` / ``target_type="file"`` / ``target_label``
  NULL, and hits whose path is not in the ``path -> file_id`` map are counted
  ``dropped`` and produce no edge.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_reference_index.py -v
"""
import pytest
import sqlalchemy as sa

from app.db import SessionLocal
from app.indexing import (
    DEFAULT_CAP,
    ReferenceHit,
    find_references,
    index_symbol_references,
    ripgrep_available,
)
from app.indexing.reference_index import _parse_ripgrep_output
from app.models import Edge, File, Repository, Symbol

# Tests that shell out to a real ripgrep binary are skipped when `rg` is not on
# PATH (e.g. CI images without it). The deterministic pure/persist tests above
# still cover the logic everywhere; these only verify the ripgrep wiring + flags.
rg_required = pytest.mark.skipif(
    not ripgrep_available(), reason="ripgrep binary not on PATH"
)


def _def_line(path, marker: str) -> int:
    """1-indexed line number of the first line whose stripped text equals
    ``marker`` â€” so a test can locate ``class Widget:`` without hard-coding line
    offsets (robust to a leading comment / shebang)."""
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() == marker:
            return i
    raise AssertionError(f"marker {marker!r} not found in {path}")


# â”€â”€â”€ deterministic pure parse (no ripgrep, no DB) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_parse_ripgrep_output_parses_strips_and_excludes_def_line():
    """The pure core: ripgrep's ``path:line:col:text`` text becomes hits, the
    leading ``./`` is stripped so paths line up with ``files.path``, the symbol's
    own binding line is dropped, malformed / non-numeric / blank lines are
    skipped, and a colon inside the matched text survives (maxsplit=3)."""
    raw = "\n".join(
        [
            "./pkg/thing.py:1:7:class Widget:",                          # def line â†’ EXCLUDED
            "./pkg/thing.py:5:12:        return Widget()",               # own-file, non-binding â†’ KEPT
            "./pkg/user.py:1:24:from pkg.thing import Widget",           # import ref â†’ KEPT
            "./pkg/user.py:4:10:    w = Widget()",                       # call ref â†’ KEPT
            "./pkg/bad.py:xx:1:    x = Widget",                          # non-numeric line â†’ SKIPPED
            "garbage_no_colon",                                          # not a match line â†’ SKIPPED
            "./pkg/other.py:9:3:    Widget   # in a comment: has a colon",  # colon in text â†’ KEPT
            "",                                                         # blank â†’ SKIPPED
        ]
    )

    hits = _parse_ripgrep_output(
        raw, definition_path="pkg/thing.py", definition_line=1, cap=None
    )

    # â”€â”€ the four real references survive; the binding line is gone â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    locations = {(h.path, h.line) for h in hits}
    assert locations == {
        ("pkg/thing.py", 5),  # self-recursive â€” own file, NOT the binding line
        ("pkg/user.py", 1),   # `from ... import Widget`
        ("pkg/user.py", 4),   # `w = Widget()`
        ("pkg/other.py", 9),  # a match whose line text contains a `:`
    }, locations
    assert ("pkg/thing.py", 1) not in locations, "own definition line was not excluded"

    # â”€â”€ field fidelity: the `./` prefix was stripped, column + text preserved â”€
    own = next(h for h in hits if h.path == "pkg/thing.py" and h.line == 5)
    assert own.column == 12
    assert "return Widget()" in own.line_text
    # A colon in the matched text is kept verbatim (maxsplit=3), not truncated.
    other = next(h for h in hits if h.path == "pkg/other.py")
    assert ": has a colon" in other.line_text


def test_parse_ripgrep_output_normalizes_windows_paths():
    """On Windows ripgrep prints ``.\\pkg\\thing.py`` (backslash separators plus a
    ``.\\`` prefix) instead of ``./pkg/thing.py``. The parse must normalize both
    forms to the same POSIX-relative path the rest of the pipeline â€”
    ``files.path`` and the caller's ``path -> file_id`` map â€” stores and expects,
    or hits silently miss the map and get dropped on persist (counted ``dropped``,
    no edge). The own-binding-line exclusion also rides on this: it compares the
    *normalized* path to the caller's POSIX ``definition_path``, so a
    Windows-emitted binding line must normalize *before* that check or it would
    survive as a false self-reference. Mirrors the import-graph / files stage
    POSIX invariant end-to-end on any platform. Pinned at the pure layer so the
    fix is verified without a ripgrep binary on PATH.
    """
    raw = "\n".join(
        [
            r".\pkg\thing.py:1:7:class Widget:",                 # def line â†’ EXCLUDED after normalize
            r".\pkg\thing.py:5:12:        return Widget()",       # own-file ref â†’ KEPT, path POSIX
            r".\pkg\user.py:1:24:from pkg.thing import Widget",   # cross-file import â†’ KEPT
            r".\pkg\user.py:4:10:    w = Widget()",              # cross-file call â†’ KEPT
        ]
    )

    hits = _parse_ripgrep_output(
        raw, definition_path="pkg/thing.py", definition_line=1, cap=None
    )

    # Every surviving hit's path is POSIX-normalized: forward slashes only, no
    # backslashes, no leading `.\` â€” i.e. it would now match a `{posix_path: id}`
    # map built from `files.path`.
    assert all(
        "/" in h.path and "\\" not in h.path and not h.path.startswith(".")
        for h in hits
    ), [h.path for h in hits]

    locations = {(h.path, h.line) for h in hits}
    assert locations == {
        ("pkg/thing.py", 5),  # self-reference on a non-binding line â€” KEPT
        ("pkg/user.py", 1),   # `from ... import Widget`
        ("pkg/user.py", 4),   # `w = Widget()`
    }, locations
    # The binding line (line 1) was excluded *because* normalization made its
    # path match the POSIX `definition_path` â€” the exact Windows regression this
    # pins: had the `.\\`-prefixed path not normalized, the equality check against
    # `pkg/thing.py` would fail and the binding would leak through as a self-ref.
    assert ("pkg/thing.py", 1) not in locations, "own binding line not excluded under Windows paths"


def test_parse_ripgrep_output_applies_cap():
    """With `cap` set, at most `cap` hits survive (the defensive truncation that
    mirrors ripgrep's `--max-count`); `cap=None` is unbounded. Excluding the
    definition line happens *before* the cap, so a binding within the first `cap`
    lines does not consume an edge slot."""
    raw = "\n".join(f"./pkg/common.py:{ln}:4:    x{ln} = spam" for ln in range(1, 61))

    bounded = _parse_ripgrep_output(raw, "pkg/common.py", definition_line=0, cap=50)
    assert len(bounded) == 50, len(bounded)
    # First-most (document order) survive; cap is an upper bound, never exceeded.
    assert all(h.path == "pkg/common.py" for h in bounded)
    assert [h.line for h in bounded] == list(range(1, 51))

    unbounded = _parse_ripgrep_output(raw, "pkg/common.py", definition_line=0, cap=None)
    assert len(unbounded) == 60, len(unbounded)


# â”€â”€â”€ real ripgrep (skipped if rg absent) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@rg_required
def test_find_references_finds_known_symbol(tmp_path):
    """A known symbol's references are found correctly end-to-end via real
    ripgrep: the cross-file call, the same-file recursive reference, and the
    import all surface; the symbol's own ``class`` binding line is excluded."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    thing = pkg / "thing.py"
    thing.write_text(
        'class Widget:\n'
        '    """A widget."""\n'
        '\n'
        '    def make(self):\n'
        '        return Widget()\n',  # line 5: a self-reference inside the own file
        encoding="utf-8",
    )
    user = pkg / "user.py"
    user.write_text(
        'from pkg.thing import Widget\n'   # line 1: import reference
        '\n'
        '\n'
        'def use():\n'
        '    w = Widget()\n'               # line 5: a genuine cross-file call
        '    return w\n',
        encoding="utf-8",
    )

    def_line = _def_line(thing, "class Widget:")
    hits = find_references(
        "Widget", tmp_path, definition_path="pkg/thing.py",
        definition_line=def_line, cap=DEFAULT_CAP,
    )

    locations = {(h.path, h.line) for h in hits}

    # The symbol's own binding line is NOT recorded as a reference to itself.
    assert ("pkg/thing.py", def_line) not in locations

    # The cross-file call and the import reference both surface.
    assert ("pkg/user.py", 5) in locations, locations  # w = Widget()
    assert ("pkg/user.py", 1) in locations, locations  # from pkg.thing import Widget

    # The same-file recursive reference (line 5, NOT the binding line) is kept â€”
    # recursion inside a body is a real usage; only the binding line is excluded.
    assert ("pkg/thing.py", 5) in locations, locations
    return_hit = next(h for h in hits if h.path == "pkg/thing.py" and h.line == 5)
    assert "return Widget()" in return_hit.line_text

    # No stray matches outside the two fixture files.
    assert {h.path for h in hits} <= {"pkg/thing.py", "pkg/user.py"}


@rg_required
def test_find_references_caps_common_name(tmp_path):
    """A name repeated far past the cap does not blow past it: ripgrep's
    ``--max-count`` stops early and the returned hit count stays <= cap even when
    the on-disk occurrence count is many times the cap."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    common = pkg / "common.py"
    # 60 lines, each a distinct whole-word occurrence of `spam` (a deliberately
    # uncommon name so the only matches are the ones we wrote).
    common.write_text(
        "\n".join(f"    v{i} = spam  #{i}" for i in range(60)) + "\n",
        encoding="utf-8",
    )
    on_disk = common.read_text(encoding="utf-8").count("spam")
    assert on_disk == 60, on_disk  # sanity: the fixture really is "many times the cap"

    cap = 5
    # def_line=0 (no line 0 exists) isolates the cap from the def-line exclusion.
    hits = find_references(
        "spam", tmp_path, definition_path="pkg/common.py",
        definition_line=0, cap=cap,
    )

    assert len(hits) <= cap, (len(hits), cap)
    # The cap actually binds: 60 on-disk occurrences collapse to at most `cap` hits.
    assert len(hits) == cap, (len(hits), cap)


# â”€â”€â”€ live Postgres persist (no ripgrep; crafted hits) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _setup_repo_and_symbol(session) -> tuple[int, int, dict[str, int]]:
    """Persist a small repo with two files plus a ``Widget`` symbol row, and
    return ``(repo_id, symbol_id, {posix_path: file_id})``. The persist test feeds
    :func:`index_symbol_references` crafted hits (no real ripgrep, no disk files)
    â€” only path + id matter to the pathâ†’file_id map."""
    repo = Repository(url_or_path="file:///test/refs", name="test/refs", status="indexing")
    session.add(repo)
    session.commit()
    session.refresh(repo)

    thing = File(repository_id=repo.id, path="proj/thing.py", language="python", loc=10)
    user = File(repository_id=repo.id, path="proj/user.py", language="python", loc=10)
    session.add_all([thing, user])
    session.flush()  # assign file PKs so the symbol can FK to thing.id

    sym = Symbol(
        file_id=thing.id, name="Widget", kind="class",
        line_start=1, line_end=5, docstring=None, parent_symbol_id=None,
    )
    session.add(sym)
    session.flush()
    session.commit()

    path_to_file_id = {thing.path: thing.id, user.path: user.id}
    return repo.id, sym.id, path_to_file_id


def _cleanup(session, repo_id: int | None) -> None:
    """edges â†’ symbols â†’ files â†’ repo, so a mid-test failure leaves no stray rows
    (mirrors test_import_graph's teardown order)."""
    if repo_id is not None:
        session.execute(sa.delete(Edge).where(Edge.repository_id == repo_id))
        session.execute(
            sa.delete(Symbol).where(
                Symbol.file_id.in_(
                    sa.select(File.id).where(File.repository_id == repo_id)
                )
            )
        )
        session.execute(sa.delete(File).where(File.repository_id == repo_id))
        session.execute(sa.delete(Repository).where(Repository.id == repo_id))
        session.commit()


def test_index_symbol_references_dedupes_to_one_edge_per_file():
    """Live-DB round-trip: the persist layer writes exactly one ``references``
    edge per distinct file the hits resolve to. Multiple hits in the same file
    collapse (the graph node is the file, not each textual occurrence); a hit
    whose path is not in the ``path -> file_id`` map is counted ``dropped`` and
    yields no edge (not flagged external â€” that is for unresolved *package*
    imports). Emits ``source_type="symbol"`` â†’ ``target_type="file"`` with
    ``target_label`` NULL, and a self-adjacency (the symbol referenced in its own
    file on a non-binding line) is kept like any other reference."""
    repo_id = None
    try:
        with SessionLocal() as session:
            repo_id, sym_id, path_to_file_id = _setup_repo_and_symbol(session)
            thing_id = path_to_file_id["proj/thing.py"]
            user_id = path_to_file_id["proj/user.py"]

            hits = [
                # Own-file, non-binding line â†’ a self-adjacency edge to thing_id.
                ReferenceHit("proj/thing.py", 5, 12, "        return Widget()"),
                # user.py, two distinct lines â†’ collapses to ONE edge to user_id.
                ReferenceHit("proj/user.py", 1, 24, "from proj.thing import Widget"),
                ReferenceHit("proj/user.py", 4, 10, "    w = Widget()"),
                # A file the walker never indexed â†’ dropped, no edge, not external.
                ReferenceHit("proj/extra.py", 7, 8, "    Widget()"),
            ]

            result = index_symbol_references(
                sym_id, hits, path_to_file_id, repo_id, session
            )
            session.commit()

        # 2 distinct resolvable files (thing, user); the third hit is dropped.
        assert result.added == 2, result
        assert result.dropped == 1, result

        # Read back through a fresh session (real DB hit, no identity map).
        with SessionLocal() as session:
            rows = (
                session.execute(
                    sa.select(Edge)
                    .where(
                        Edge.source_id == sym_id,
                        Edge.source_type == "symbol",
                        Edge.edge_type == "references",
                    )
                )
                .scalars()
                .all()
            )

            assert len(rows) == 2, [r.target_id for r in rows]
            assert all(r.target_type == "file" for r in rows), rows
            assert all(r.target_label is None for r in rows), rows  # file targets carry no label

            # One edge per distinct referencing file â€” the two user.py hits did
            # NOT produce two edges. The set is {thing_id, user_id}, proving both
            # the cross-file reference and the own-file self-adjacency were kept
            # and the unindexed extra.py produced nothing.
            assert {r.target_id for r in rows} == {thing_id, user_id}, (
                [r.target_id for r in rows], thing_id, user_id
            )
    finally:
        with SessionLocal() as session:
            _cleanup(session, repo_id)

