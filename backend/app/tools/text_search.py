"""Text-search tool for the RepoLens agent layer (SDD Â§10).

The third thin tool the agent orchestrator (SDD Â§9) calls, after Symbol Search
and File Search: given a repository and a free-text query, shell out to
``ripgrep`` across the repo's Python files and return matching lines as
structured hits in the SDD Â§10 Text Search output shape
``[{file, line, matched_text}]`` (capped). Unlike :mod:`app.tools.symbol_search`
(a DB query on symbol *names*) this is *free text*: it matches anywhere in a
line â€” a code snippet, a string literal, a comment, an expression â€” not symbol
names and not whole words.

Scope contrast with :mod:`app.indexing.reference_index`: that stage matches a
symbol's name as a **whole word** (``rg -w --fixed-strings``) to build the
reference graph. This tool is the opposite on both axes: **not whole-word** (no
``-w`` â€” a free-text query for ``config`` should match ``reconfigure`` and
``get_config`` too) and **literal-by-default** (``--fixed-strings``, so an agent
searching ``widget(`` or ``a = b.c`` does not get an unbalanced-group regex
error). ripgrep's *normal* (regex) pattern matching is available via the
``regex=True`` opt-in when an agent deliberately wants alternation / anchors /
classes â€” see :func:`find_text`. The posture (literal-by-default, never
whole-word) is the direct reading of "search for the literal text â€¦ free text,
â€¦ not just whole-word"; the regex flag honors "support ripgrep's normal pattern
matching" on demand. Same reliability-first ethos as the rest of the stack (SDD
â€§10: "structured JSON, never free text"; "deterministic engineering, not LLM
summarization"): a tool an LLM calls with arbitrary text should not error on
that text's punctuation.

Three cooperating pieces (the same pure-shell / DB-resolve split as
:mod:`app.indexing.reference_index`'s pure / persist):

* :func:`find_text` â€” *pure wrt DB*: shells out to ripgrep for ``query`` under
  ``repo_root`` and parses the matches. *Takes* ``repo_root``, owns no session,
  commits nothing â€” the unit-testable core, the mirror of
  :func:`app.indexing.reference_index.find_references`.
* :func:`_parse_text_search_output` â€” the deterministic, no-ripgrep, no-DB heart
  of the parse: turn ``path:line:text`` text into hits, normalize each path to
  the POSIX-relative form (strip the leading ``./`` / ``.\\`` ripgrep emits and
  convert backslashes to forward slashes), and cap. Tested directly so the
  parse / normalize / cap logic is pinned independent of a ripgrep binary on
  PATH â€” exactly as ``_parse_ripgrep_output``'s pure tests pin the reference
  stage. Three-part (``path:line:text``) where the reference stage is four-part
  (``path:line:col:text``); Text Search's SDD shape has no column, so ``--column``
  is omitted and the split is ``maxsplit=2``.
* :func:`_normalize_ripgrep_path` â€” the path normalizer, a copy of
  :mod:`app.indexing.reference_index`'s (load-bearing on Windows: a ``.\\``-prefixed
  backslash path must become the POSIX form that lines up with ``files.path`` and
  the caller's UI). Kept local rather than imported private so this tool is
  self-contained; pinned again here by a Windows pure test so a regression in the
  copy is caught without a ripgrep binary.
* :func:`search_text` â€” the *tool* the agents call: takes a ``repository_id``
  (not a bare ``repo_root``) plus an injected ``Session``, **resolves**
  ``repo_root`` from the repository row's ``url_or_path`` and **confirms** it
  against the ``files`` table's stored relative paths (an indexed file must exist
  on disk under that root, else the repo was moved / the path is a remote URL),
  then delegates to :func:`find_text`. Reads only â€” no writes, no commit â€”
  matching the injected-session / owning-no-transaction posture of the sibling
  tools (:mod:`app.tools.symbol_search`, :mod:`app.tools.file_search`) and the
  indexing ``index_*`` stages (SDD Â§7 step 9). The user-facing repo_root story
  ("resolves repo_root from the files table's stored paths") lives HERE: the
  throwaway driver scripts hardcode ``REPO_PATH`` and pass it straight to
  ``find_references``; a real tool derives it from the DB instead.

Cap convention (reused from :mod:`app.indexing.reference_index`, SDD Â§13 risk row
"cap result counts"): :data:`DEFAULT_CAP` = 50, ripgrep ``--max-count <cap>``
bounds a single hot file's emitted matches (a broad query like ``self`` can
match thousands of lines in one file), and :func:`_parse_text_search_output`'s
``hits[:cap]`` is the **authoritative global cap** on the returned list (the
total across all files). ``--max-count`` is per-file, so it alone cannot bound the
grand total; the Python slice does. ``cap=None`` is unbounded. The value tracks
:mod:`app.indexing.reference_index`'s ``DEFAULT_CAP``; the semantics differ
(per-query here vs. per-symbol there), so it is a local copy rather than an
import to keep the layers decoupled and avoid a change tuned for one silently
shifting the other.

ripgrep exit-code contract: ``0`` = matches found (parse stdout), ``1`` = no
matches (empty result, *not* an error), ``>= 2`` = real error (bad flag, missing
binary, binary file) â†’ raised as :class:`RuntimeError` â€” identical to the
reference stage. The ``rg`` binary is resolved from ``PATH`` (override via the
``ripgrep_bin`` argument); use :func:`app.indexing.ripgrep_available` to precheck
it (tests skip on absence).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import File, Repository

# The ripgrep binary name; resolved from PATH. Overridable per call so tests /
# the agent layer can point at a specific binary if needed. Mirrors
# :mod:`app.indexing.reference_index`.
RIPGREP_BIN: str = "rg"

# Default cap on returned hits (SDD Â§10 "capped result count"; SDD Â§13 risk row
# "cap result counts"). 50 tracks :mod:`app.indexing.reference_index`'s
# ``DEFAULT_CAP`` deliberately â€” same default, same ``--max-count`` + slice
# mechanism â€” but is a local copy: the cap here is per-query (one search across
# the whole repo), there per-symbol (one search per indexed symbol), so coupling
# them by import would let a value tuned for one silently shift the other.
# `None` means unbounded.
DEFAULT_CAP: int = 50


@dataclass
class TextHit:
    """One matching text line, in the SDD Â§10 Text Search output shape.

    Every field is a plain JSON type (str / int), so ``dataclasses.asdict`` is
    directly ``json.dumps``-able â€” the whole point of SDD Â§10's "structured
    JSON, never free text" contract: agents and the Synthesizer work from these
    facts, not paraphrased summaries.

    ``file`` is the matched line's POSIX-rel path (ripgrep's leading ``./`` /
    ``.\\`` stripped and backslashes converted to forward slashes, so it lines up
    directly with the ``files.path`` values in the DB on any platform). ``line``
    is the 1-indexed line number (ripgrep). ``matched_text`` is the whole matched
    source line (untrimmed) for display / citation â€” ripgrep prints the full
    line, not just the matched substring (no ``--only-matching``).
    """

    file: str
    line: int
    matched_text: str


# â”€â”€â”€ ripgrep availability â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Precheck lives in :mod:`app.indexing.reference_index` (``ripgrep_available``);
# this tool reuses it rather than duplicating the ``rg --version`` probe â€” one
# binary, one availability check, shared across every ripgrep-backed stage/tool.


# â”€â”€â”€ pure extraction (no DB session) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _normalize_ripgrep_path(path: str) -> str:
    """Normalize a ripgrep-emitted path to the POSIX-relative form stored in
    ``files.path`` â€” a copy of :func:`app.indexing.reference_index._normalize_ripgrep_path`.

    Two platform-dependent tweaks to the raw path ripgrep prints when searching
    ``.`` from the repo root:

    * **Strip a leading ``./`` or ``.\\``.** ripgrep prepends ``.`` + the
      platform's path separator, so ``./pkg/thing.py`` on POSIX and
      ``.\\pkg\\thing.py`` on Windows. Both are exactly two chars and dropped
      before the separator swap. Anything else keeps its leading bytes untouched.
    * **Convert backslashes to forward slashes.** On Windows ripgrep separates
      path segments with ``\\``; every other pipeline stage (``files.path`` from
      the walker, the symbol-search join, the reference hit's own path) stores
      and expects POSIX-style paths, so a Windows ``.\\pkg\\thing.py`` must become
      ``pkg/thing.py`` or the hit's ``file`` silently fails to align with
      ``files.path`` downstream. This swap is load-bearing on Windows; the prefix
      strip alone would leave ``pkg\\thing.py``.

    The caller runs ripgrep with ``cwd=repo_root`` searching ``.``, so one of the
    ``./`` / ``.\\`` forms is what we actually get. Kept local (not an import of
    the reference stage's private helper) so this tool is self-contained; a
    Windows pure test pins the copy directly.
    """
    if len(path) >= 2 and path[0] == "." and path[1] in ("/", "\\"):
        path = path[2:]
    return path.replace("\\", "/")


def _parse_text_search_output(raw: str, *, cap: int | None) -> list[TextHit]:
    """Parse ripgrep's ``path:line:text`` match lines into hits and cap the
    result. *Pure*: no subprocess, no DB, no filesystem â€” given ripgrep's stdout
    as a string it returns data. This is the deterministic, ripgrep-free core the
    pure tests pin (the mirror of ``_parse_ripgrep_output``).

    Blank lines and anything that does not split into three ``:``-delimited parts
    â€” the path, the line number, the matched text â€” are skipped (``maxsplit=2``
    so a ``:`` *inside* the matched line text survives; a source line like
    ``x: int`` parses path/line/``"x: int"`` and keeps both colons in
    ``matched_text``). Each path is POSIX-normalized via
    :func:`_normalize_ripgrep_path`. The cap is the **authoritative global**
    bound on the returned list (the total across all files); ``--max-count`` bound
    a single file upstream but cannot bound the grand total, so this slice does.
    Applied in document (ripgrep output) order; ``cap=None`` is unbounded.
    """
    hits: list[TextHit] = []
    for raw_line in raw.split("\n"):
        if not raw_line:
            continue
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            # Not a `path:line:text` match line (e.g. a truncated / prefixed line
            # ripgrep sometimes prints); skip rather than guess fields.
            continue
        path, line_s, text = parts
        path = _normalize_ripgrep_path(path)
        try:
            line = int(line_s)
        except ValueError:
            continue  # non-numeric line â€” not a match line, skip
        hits.append(TextHit(file=path, line=line, matched_text=text))
    if cap is not None:
        hits = hits[:cap]
    return hits


def find_text(
    query: str,
    repo_root: Path | str,
    *,
    cap: int | None = DEFAULT_CAP,
    regex: bool = False,
    ripgrep_bin: str = RIPGREP_BIN,
) -> list[TextHit]:
    """Run ripgrep for ``query`` across the repo's ``.py`` files and return the
    matching lines as hits, capped at ``cap``. *Pure wrt DB*: no session, no
    commit, no write â€” it shells out to ripgrep (so it touches the filesystem)
    but owns no database transaction. The unit-testable core; takes ``repo_root``
    the way :func:`app.indexing.reference_index.find_references` does (the future
    agent layer / tests resolve ``repo_root`` themselves and pass it in, or call
    :func:`search_text` which does the resolution).

    ripgrep flags:

    * ``(no ``-w``)`` â€” deliberately **not** whole-word: Text Search is free
      text, so ``config`` must match inside ``reconfigure`` too. This is the axis
      on which it contrasts the reference-index stage's ``-w``.
    * ``--fixed-strings`` (default, ``regex=False``) â€” treat ``query`` literally,
      so punctuation is not regex-interpreted (an agent searching ``widget(`` or
      ``a = b.c`` does not hit an unbalanced-group / dot-metachar surprise).
      Omitted when ``regex=True`` â€” then ripgrep's *normal* regex matching
      applies (alternation, anchors, classes) for an agent that deliberately
      wants a pattern. "Search for the literal text" + "support ripgrep's normal
      pattern matching" are both honored: literal by default, regex on opt-in.
    * ``--type py`` â€” only Python files (MVP scope: the indexed set is ``.py``).
      ripgrep respects ``.gitignore`` / ``.ignore`` by default, so it skips the
      VCS / venv / build dirs the file-walker also prunes; a hit in a ``.py`` file
      the walker did *not* index (e.g. one over the walker's 1 MB size cap) is a
      minor self-correcting divergence, the same posture the reference stage takes.
    * ``--line-number --no-heading --color never`` â€” the ``path:line:text``
      format :func:`_parse_text_search_output` parses (``--column`` is NOT passed
      â€” the SDD Â§10 Text Search shape is three fields, not four).
    * ``--max-count <cap>`` â€” stop after ``cap`` matching lines *per file* (the
      hot-file efficiency bound; a broad query like ``self`` matching thousands
      of lines in one file is capped at the source). Omitted when ``cap is None``.
      The authoritative *global* cap on the returned list is
      :func:`_parse_text_search_output`'s ``hits[:cap]`` slice (``--max-count``
      is per-file and cannot bound the grand total).
    * ``-- <query> .`` â€” run with ``cwd=repo_root`` searching ``.``, so the
      printed paths are repo-root-relative (a leading ``./`` / ``.\\`` is stripped
      and backslashes are converted to forward slashes in
      :func:`_normalize_ripgrep_path`) and line up with the POSIX ``files.path``.

    ripgrep exit: ``0`` = matches found (parse), ``1`` = no matches (empty
    result, *not* an error â€” returns ``[]``), ``>= 2`` = real error â†’ raised as
    :class:`RuntimeError` (bad flag, missing binary, binary fileâ€¦).
    """
    args = [
        ripgrep_bin,
        "--line-number",
        "--no-heading",
        "--color", "never",
        "-i",  # case-insensitive search
        "-g", "*.py",
        "-g", "*.ts",
        "-g", "*.tsx",
        "-g", "*.js",
        "-g", "*.jsx",
        "-g", "*.json",
        "-g", "*.md",
        "-g", "*.yaml",
        "-g", "*.yml",
    ]
    if not regex:
        args.append("--fixed-strings")
    if cap is not None:
        args += ["--max-count", str(cap)]
    # `--` ends option parsing so a (hypothetical) query beginning with `-` is
    # treated as the pattern, then `.` is the search root.
    args += ["--", query, "."]

    # cwd=repo_root + search "." â†’ paths print relative as `./rel/path.py` on
    # POSIX / `.\rel\path.py` on Windows, which _normalize_ripgrep_path converts
    # to the POSIX-rel form stored in `files.path`. text+utf-8 so non-ASCII in
    # source lines (paths, comments, docstrings) survives the parse.
    cp = subprocess.run(
        args,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if cp.returncode == 1:
        return []  # no matches â€” the normal no-result path
    if cp.returncode != 0:
        raise RuntimeError(
            f"ripgrep failed (exit {cp.returncode}) for query={query!r}: "
            f"{cp.stderr.strip()}"
        )
    return _parse_text_search_output(cp.stdout, cap=cap)


# â”€â”€â”€ repository-scoped wrapper (the tool agents call) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def search_text(
    repository_id: int,
    query: str,
    session: Session,
    *,
    cap: int | None = DEFAULT_CAP,
    regex: bool = False,
    ripgrep_bin: str = RIPGREP_BIN,
) -> list[TextHit]:
    """Text-search repository ``repository_id`` for ``query`` and return capped
    matches as :class:`TextHit` (SDD Â§10 shape). The tool the agents call: it
    resolves the on-disk ``repo_root`` from the repository row (not hardcoded, as
    the throwaway driver scripts do), **confirms** it against the ``files``
    table's stored relative paths, then delegates to :func:`find_text`.

    *Pure read*: no writes, no commit, owning no transaction â€” the same
    injected-session posture as :mod:`app.tools.symbol_search` /
    :mod:`app.tools.file_search` and the indexing ``index_*`` stages.

    repo_root resolution ("resolves repo_root from the files table's stored
    paths"):

    1. ``session.get(Repository, repository_id)``. A **missing repo row** returns
       ``[]`` â€” consistent with the sibling tools' repository-scoping (an
       unknown repo id has no indexed text), not an error.
    2. ``repo.url_or_path`` is the candidate ``repo_root`` â€” the local path the
       repo was cloned / added from (``run_walker_once.py`` stored it there). If
       it is absent or not a directory (a remote URL, or the on-disk clone moved
       / was deleted), raise :class:`FileNotFoundError` rather than silently
       ``[]`` â€” a *configured* repo whose root is unreachable is a real
       misconfiguration an agent / planner should see, not "no text matched".
    3. **Confirmation via the files table**: pick one indexed ``files.path``, and
       verify ``Path(repo_root) / files.path`` exists on disk. The files table
       stores paths *relative* to the repo root, so a mismatch means
       ``url_or_path`` and the indexed tree have drifted apart (repo moved after
       indexing; a stale row pointing at an old path) â€” raise
       :class:`RuntimeError` with both paths in the message. With no files rows
       indexed yet, there is nothing to search; proceed (``find_text`` returns
       ``[]``).

    ``query`` is passed straight to :func:`find_text` unstripped â€” leading /
    trailing whitespace is meaningful to a free-text search (a query of ``  `` is
    a search for two spaces, not "no query"), so unlike the symbol/file tools this
    does NOT strip. An empty-string query is likewise passed through (ripgrep
    treats it as match-everything, which with ``--fixed-strings`` is an error â€”
    surfaced as ``RuntimeError`` rather than swallowed).
    """
    from app.tools._path_resolve import resolve_repo_root

    repo = session.get(Repository, repository_id)
    if repo is None:
        return []

    repo_root = resolve_repo_root(repo, session)
    if not repo_root or not Path(repo_root).is_dir():
        raise FileNotFoundError(
            f"repository {repository_id!r} repo_root is not a usable directory: "
            f"{repo.url_or_path!r} (local path or managed clone not found)"
        )

    return find_text(
        query,
        repo_root,
        cap=cap,
        regex=regex,
        ripgrep_bin=ripgrep_bin,
    )
