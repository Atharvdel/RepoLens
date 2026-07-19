"""Tests for the Text Search tool (SDD §10).

Same pure / live split as :mod:`tests.test_reference_index`, adapted to the
free-text tool (which has no persist layer — it is a pure read, SDD §10 — so the
"live" tier is *real ripgrep against the on-disk flask clone* plus *live
Postgres for repo-root resolution*, rather than a crafted-hit persist round-trip):

* ``test_parse_text_search_output_*`` — *deterministic pure, no ripgrep, no DB*:
  feed crafted ``path:line:text`` strings to the parse core and pin the parse,
  the leading ``./`` / ``.\\`` strip, the colon-in-text survival (``maxsplit=2``),
  malformed / non-numeric / blank-line skipping, and the cap. A Windows-path
  regression test mirrors
  :func:`tests.test_reference_index.test_parse_ripgrep_output_normalizes_windows_paths`
  — pinned here so a regression in *this tool's own copy* of the normalizer is
  caught without a ripgrep binary on PATH. (No own-binding-line exclusion here:
  Text Search keeps every match — the def-line drop is a
  :mod:`app.indexing.reference_index` whole-word-references concern, not free
  text's.)
* ``test_search_text_*`` — *live: real ripgrep against the on-disk flask clone
  AND live Postgres for repo-root resolution* (skipped cleanly when ripgrep is
  not on PATH or flask isn't indexed / isn't on disk). Text Search resolves
  ``repo_root`` from the ``repositories`` row (unlike the throwaway driver
  scripts, which hardcode ``REPO_PATH``) and confirms it against the ``files``
  table — so these drive the *tool* (:func:`search_text`), not just the shell
  core. A known flask docstring phrase is found in the expected file; a nonsense
  query returns an empty list, not an error; and a broad query (``self``) is
  capped.

Run from the ``backend/`` directory::

    PYTHONPATH=. python -m pytest tests/test_text_search.py -v
"""
from pathlib import Path

import pytest
import sqlalchemy as sa

from app.db import SessionLocal
from app.indexing import ripgrep_available
from app.models import File, Repository
from app.tools.text_search import (
    DEFAULT_CAP,
    TextHit,
    _parse_text_search_output,
    search_text,
)

# ─── ripgrep availability ─────────────────────────────────────────────────────
# Tests that shell out to ripgrep skip when `rg` is not on PATH (e.g. CI images
# without it), exactly as :mod:`tests.test_reference_index` does. The pure parse
# tests above still cover the logic everywhere.

rg_required = pytest.mark.skipif(
    not ripgrep_available(), reason="ripgrep binary not on PATH"
)


# ─── flask readiness probe (skip, not error, when flask isn't indexed/on-disk) ─
# Text Search shells out to ripgrep against the *on-disk* clone (unlike the
# pure-DB symbol/file tools), so readiness also requires the clone to exist at
# the path the walker stored. Defensively swallows any DB / filesystem error to
# `(False, None)` so collection in a fresh environment skips cleanly — mirrors
# the per-file `_flask_indexed()` precedent in test_symbol_search /
# test_file_search / test_reference_index (kept per-file, conftest couples no
# DB state).


def _flask_indexed() -> tuple[bool, int | None]:
    """``(ready, repo_id)``: ready iff a ``flask`` repo row exists, its
    ``url_or_path`` is a real on-disk directory, it has indexed ``files`` rows,
    and at least one of those files resolves under that root. Text Search needs
    both the DB row (for repo-root resolution) and the files on disk (to
    shell ripgrep at), so a missing clone or a moved repo skips rather than
    errors."""
    try:
        with SessionLocal() as session:
            repo = session.execute(
                sa.select(Repository).where(Repository.name == "flask")
            ).scalar_one_or_none()
            if repo is None:
                return False, None
            root = repo.url_or_path
            if not root or not Path(root).is_dir():
                return False, None
            sample_path = session.scalar(
                sa.select(File.path)
                .where(File.repository_id == repo.id)
                .order_by(File.path)
                .limit(1)
            )
            if not sample_path:
                return False, None
            if not (Path(root) / sample_path).exists():
                return False, None
            return True, repo.id
    except Exception:
        return False, None


_FLASK_READY, FLASK_REPO_ID = _flask_indexed()
flask_required = pytest.mark.skipif(
    not _FLASK_READY,
    reason="flask repo not indexed in DB / not on disk "
    "(run scripts/run_walker_once.py first)",
)


# ─── deterministic pure parse (no ripgrep, no DB) ─────────────────────────────


def test_parse_text_search_output_parses_strips_and_preserves_colon_in_text():
    """The pure core: ripgrep's ``path:line:text`` text becomes hits, the leading
    ``./`` is stripped so each ``file`` lines up with ``files.path``, a colon
    inside the matched text survives (``maxsplit=2`` so both colons in ``x: int``
    stay in ``matched_text``), and malformed / non-numeric / blank lines are
    skipped. No own-binding-line exclusion here — Text Search keeps every match
    (the def-line drop is the whole-word reference graph's concern, not free
    text's)."""
    raw = "\n".join(
        [
            "./pkg/thing.py:1:class Widget:",                       # path+line+text  → KEPT
            "./pkg/user.py:4:x: int = 1",                           # colon in text    → KEPT (both colons)
            "./pkg/bad.py:xx:nope",                                 # non-numeric line → SKIPPED
            "garbage_no_colon",                                     # not a match line → SKIPPED
            "./pkg/other.py:9:    w = Widget()  # has a: colon",    # colon in text    → KEPT
            "",                                                    # blank            → SKIPPED
        ]
    )

    hits = _parse_text_search_output(raw, cap=None)

    locations = {(h.file, h.line) for h in hits}
    assert locations == {
        ("pkg/thing.py", 1),
        ("pkg/user.py", 4),
        ("pkg/other.py", 9),
    }, locations

    # `./` prefix stripped; matched_text is the whole source line.
    thing = next(h for h in hits if h.file == "pkg/thing.py" and h.line == 1)
    assert thing.matched_text == "class Widget:", thing.matched_text

    # A colon inside the matched text survives verbatim (maxsplit=2 keeps it),
    # not truncated at the second colon.
    user = next(h for h in hits if h.file == "pkg/user.py" and h.line == 4)
    assert user.matched_text == "x: int = 1", user.matched_text
    other = next(h for h in hits if h.file == "pkg/other.py" and h.line == 9)
    assert other.matched_text == "    w = Widget()  # has a: colon", other.matched_text


def test_parse_text_search_output_normalizes_windows_paths():
    """On Windows ripgrep prints ``.\\pkg\\thing.py`` (backslash separators plus a
    ``.\\`` prefix) instead of ``./pkg/thing.py``. The parse must normalize both
    forms to the same POSIX-relative path ``files.path`` stores and the caller's
    UI expects, or a hit's ``file`` silently fails to align with the DB on
    Windows (and the hit reads as a foreign path). Mirrors
    :func:`tests.test_reference_index.test_parse_ripgrep_output_normalizes_windows_paths`;
    pinned at the pure layer so the regression is caught without a ripgrep
    binary on PATH — and pinned for *this tool's own copy* of the normalizer, so
    a divergence from the reference stage's copy is caught independently.
    """
    raw = "\n".join(
        [
            r".\pkg\thing.py:1:class Widget:",
            r".\pkg\user.py:4:    w = Widget()",
            r".\pkg\nested\deep.py:20:    deep()",
        ]
    )

    hits = _parse_text_search_output(raw, cap=None)

    # Every surviving hit's file is POSIX-normalized: forward slashes only, no
    # backslashes, no leading `.\` — i.e. it would now line up with `files.path`.
    assert all(
        "/" in h.file and "\\" not in h.file and not h.file.startswith(".")
        for h in hits
    ), [h.file for h in hits]

    locations = {(h.file, h.line) for h in hits}
    assert locations == {
        ("pkg/thing.py", 1),
        ("pkg/user.py", 4),
        ("pkg/nested/deep.py", 20),  # a backslashed *nested* path → forward slashes
    }, locations


def test_parse_text_search_output_applies_cap():
    """With ``cap`` set, at most ``cap`` hits survive — the authoritative global
    bound that backs up ripgrep's per-file ``--max-count`` (which cannot bound the
    grand total across files). First-most (document order) survive; ``cap=None``
    is unbounded. Identical cap semantics to the reference stage's parse."""
    raw = "\n".join(f"./pkg/common.py:{ln}:    v{ln} = spam" for ln in range(1, 61))

    bounded = _parse_text_search_output(raw, cap=50)
    assert len(bounded) == 50, len(bounded)
    assert all(h.file == "pkg/common.py" for h in bounded)
    assert [h.line for h in bounded] == list(range(1, 51))

    unbounded = _parse_text_search_output(raw, cap=None)
    assert len(unbounded) == 60, len(unbounded)


# ─── live: real ripgrep against flask + live Postgres repo-root resolution ────


@rg_required
@flask_required
def test_search_text_finds_known_string():
    """A known free-text string is found end-to-end via real ripgrep against the
    on-disk flask clone, with ``repo_root`` resolved from the ``repositories``
    row (not hardcoded) and confirmed against the ``files`` table before
    shelling out. The exact phrase ``URL rules, template configuration`` appears
    in flask's ``src/flask/app.py:113`` (the ``Flask`` class docstring), so that
    ``(file, line)`` is among the hits in the SDD §10 shape ``[file, line,
    matched_text]``. The same phrase also appears in
    ``src/flask/sansio/app.py:63`` (the shared base-class docstring), so the
    result is asserted to have >= 2 hits (the multi-file property) — but the
    *primary* anchor is app.py:113, not the sansio line, to avoid pinning a
    line number that may drift with flask's sansio edits.
    """
    with SessionLocal() as session:
        hits = search_text(FLASK_REPO_ID, "URL rules, template configuration", session)

    assert hits, "expected at least one text hit for the known flask docstring phrase"

    anchor = next(
        (h for h in hits if h.file == "src/flask/app.py" and h.line == 113), None
    )
    assert anchor is not None, [(h.file, h.line) for h in hits]
    # The matched text is the whole source line, containing the literal query.
    assert "URL rules, template configuration" in anchor.matched_text, anchor
    # Shape: every returned row is the SDD §10 TextHit, not a stray dict / ORM row.
    assert all(isinstance(h, TextHit) for h in hits), hits

    # The phrase is shared with sansio's base-class docstring, so >= 2 files match
    # — pinning the multi-file free-text property (not just whole-word / one file).
    assert len(hits) >= 2, [(h.file, h.line) for h in hits]


@rg_required
@flask_required
def test_search_text_caps_broad_query():
    """A broad / common query does not flood: with a small ``cap``, the returned
    hit count stays ``<= cap`` even though the on-disk occurrence count is far
    larger. ``self`` is among the most common substrings in flask — it appears in
    ``self.x``, ``(self)``, ``def f(self):``, and incidentally inside
    ``itself`` / ``yourself`` / ... (matching *inside* words is the intended
    not-just-whole-word behavior, so the on-disk count is many times the cap).
    ripgrep's per-file ``--max-count <cap>`` and the parser's global
    ``hits[:cap]`` slice both apply; the slice is the authoritative bound, so
    the result equals ``cap`` when occurrences saturate it (which ``self`` at
    cap=5 is guaranteed to). Same ``--max-count`` + slice mechanism as the
    reference stage's :func:`find_references`.
    """
    cap = 5
    with SessionLocal() as session:
        hits = search_text(FLASK_REPO_ID, "self", session, cap=cap)

    assert len(hits) <= cap, (len(hits), cap)   # the load-bearing invariant: never exceeds the cap
    assert len(hits) == cap, (len(hits), cap)   # saturated: "self" is far more common than cap=5
    assert all(isinstance(h, TextHit) for h in hits), hits


@rg_required
@flask_required
def test_search_text_nonsense_query_returns_empty_not_error():
    """A nonsense query returns an empty list — not an error, not None (ripgrep
    exit 1 → ``[]``; SDD §10: an empty result is a valid answer). Pins the
    no-matches path at the tool layer end-to-end, mirroring
    :func:`tests.test_symbol_search.test_search_symbols_nonsense_query_returns_empty_not_error`
    and the sibling file-search test.
    """
    with SessionLocal() as session:
        hits = search_text(FLASK_REPO_ID, "zzqqxxnotext9991", session)
    assert hits == [], hits
