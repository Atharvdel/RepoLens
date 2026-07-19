"""Reference-index stage of the RepoLens indexing pipeline (SDD §7 step 5).

Builds the symbol → usage-site graph: one ``edges`` row of ``edge_type =
"references"`` per **referencing file** per indexed symbol, found by shelling out
to ``ripgrep`` for the symbol's name as a whole word (``-w``) across the repo's
``.py`` files, excluding the symbol's own definition line, and capped per symbol
to bound the noise very common names would otherwise produce.

What one edge is:

* a symbol ``S`` defined at ``(file_F, line L)``
* is referenced in file ``G`` (``S``'s name appears as a whole word somewhere in
  ``G`` on a line that is not ``G == F`` AND ``line == L``)
* → one ``edges`` row: ``source_type="symbol"``, ``source_id=S.id`` /
  ``target_type="file"``, ``target_id=G.id`` (``edge_type="references"``)

Multiple whole-word hits inside the *same* referencing file collapse to one
edge — the graph node is the file, not each textual occurrence (mirrors the
import-graph stage writing one ``imports`` edge per file→module dependency
rather than one per imported name). A symbol used 30 times in ``G`` is still a
single ``S → G`` relationship; recording 30 identical rows would just dwarf the
cap's purpose and bloat the graph.

Scope / honesty notes (the SDD is explicit that this is a *reference graph*, not
a resolved call graph — SDD §8, §13 risk row "ripgrep-based reference index
produces false positives"):

* **Name-based, not resolved.** A whole-word textual match is recorded as a
  reference even when it is actually a *different* symbol that merely shares the
  name (two ``Config`` classes in different files; a local variable ``run`` vs.
  a method ``run``). Only the symbol's *own* binding line is excluded — we do
  not (and cannot, without real semantic resolution) exclude other same-named
  bindings. The SDD accepts this as a known, capped-and-labeled trade-off: the
  UI presents these as "textual references", not confirmed usages.
* **Comments / string literals are not filtered.** SDD §7 step 5 mentions
  excluding matches inside strings/comments "where cheap to detect"; doing it
  reliably is not cheap (it needs a real tokenizer), so this MVP stage keeps all
  whole-word matches. Same posture as the import graph's PEP 420 limitation
  (§18): flag rather than guess. A dedicated false-positive reduction pass is a
  later hardening step.
* **Path parsing.** ripgrep is run in plain ``path:line:col:text`` mode
  (``--line-number --column --no-heading --color never``). Paths containing a
  ``:`` would break the ``split(":", 3)`` parse (a NUL/``--json`` mode would be
  robust to it); real Python repos do not have colons in source paths, and this
  is noted here rather than over-engineered. Fixture-grounded tests exercise the
  parse directly so regressions surface without a repo on disk.

Three cooperating pieces (same pure / persist split as :mod:`app.indexing.parser`
and :mod:`app.indexing.import_graph`):

* :func:`find_references` — *pure* (no DB session): shells out to ``ripgrep`` for
  ``name`` under ``repo_root``, parses the matches, drops the symbol's own
  definition line, and caps the result. Returns a list of :class:`ReferenceHit`
  (line granularity — one per matching line, *before* the file-level dedup the
  persist layer applies). This is the unit-testable core.
* :func:`_parse_ripgrep_output` — the deterministic, no-ripgrep, no-DB heart of
  the parse: turn ``path:line:col:text`` text into hits, normalize each path to
  the POSIX-relative form (strip the leading ``./`` or ``.\\`` ripgrep emits when
  searching ``.`` and convert backslashes to forward slashes — see
  :func:`_normalize_ripgrep_path`), exclude ``(definition_path, definition_line)``,
  and cap. Tested directly so the parse/normalize/exclude/cap logic is pinned
  independent of a ripgrep binary being on PATH.
* :func:`index_symbol_references` — *persists*: takes the hits, resolves each
  hit path to a ``file_id`` via the caller-supplied ``path -> file_id`` map (the
  analog of :class:`ImportResolver`, but trivial enough to stay a plain dict),
  and writes one ``references`` edge per distinct referencing file. Returns a
  :class:`ReferenceIndexResult` (edges added / hits dropped). Does **not commit**
  — the caller owns the transaction, exactly like
  :func:`app.indexing.walker.walk_repository` and
  :func:`app.indexing.import_graph.index_file_imports` (SDD §7 step 9). No
  ``flush`` is needed: ``target_id`` (a previously-persisted file) and
  ``repository_id`` (a committed repo) always reference rows with PKs, and the
  source ``symbol_id`` is likewise already-committed.

:func:`find_and_index_references` glues the two together for the future
``pipeline.py`` orchestrator.

ripgrep exit-code contract: ``0`` = matches found (parse stdout), ``1`` = no
matches (empty result, *not* an error), ``>= 2`` = real error (bad flag, missing
binary, etc.) → raised as :class:`RuntimeError`. The ``rg`` binary is resolved
from ``PATH`` (override via the ``ripgrep_bin`` argument); use
:func:`ripgrep_available` to precheck it.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Edge

# The ripgrep binary name; resolved from PATH. Overridable per call so tests /
# the throwaway driver can point at a specific binary if needed.
RIPGREP_BIN: str = "rg"

# Default per-symbol cap on parse-ready hits (SDD §7 step 5 / §13 risk row: cap
# result counts to keep very common names like `self` / `data` / `__init__` from
# drowning the graph). Applied both via ripgrep `--max-count` (so rg stops early
# — important for ``self``-tier names that match thousands of lines) and again in
# :func:`_parse_ripgrep_output` (so the cap holds even on crafted/injected raw
# output). `None` means unbounded.
DEFAULT_CAP: int = 50


@dataclass
class ReferenceHit:
    """One textual whole-word occurrence of a symbol name, line granularity.

    The pure layer (:func:`find_references`) returns these *before* the
    file-level dedup the persist layer applies — i.e. ``find_references`` may
    return several hits in the same file (one per matching line); they collapse
    to a single ``references`` edge on persist.

    ``path`` is POSIX-relative to the repo root (ripgrep's leading ``./`` or
    ``.\\`` stripped and backslashes converted to forward slashes, so it lines up
    directly with the ``files.path`` values in the DB and the caller's
    ``path -> file_id`` map on any platform). ``line`` is 1-indexed (ripgrep);
    ``column`` is the 0-indexed byte column of the match within the line;
    ``line_text`` is the matched line's source (untrimmed) for diagnostics/tests.
    """

    path: str
    line: int
    column: int
    line_text: str


@dataclass
class ReferenceIndexResult:
    """Tally returned by :func:`index_symbol_references`.

    ``added`` is the number of ``references`` edges written (one per distinct
    referencing *file* that resolved). ``dropped`` counts hits whose path was not
    in the caller's ``path -> file_id`` map — a match ripgrep surfaced in a file
    the walker never indexed (a pruned dir, a non-indexed file, etc.). Such a hit
    cannot become a ``target_type="file"`` edge and is *not* flagged external
    (``external`` is for unresolved *package* imports, not file targets), so it is
    simply skipped and counted here for the driver's report.
    """

    added: int
    dropped: int

    def __add__(self, other: "ReferenceIndexResult") -> "ReferenceIndexResult":
        return ReferenceIndexResult(
            added=self.added + other.added,
            dropped=self.dropped + other.dropped,
        )


# ─── ripgrep availability ────────────────────────────────────────────────────


def ripgrep_available(ripgrep_bin: str = RIPGREP_BIN) -> bool:
    """True when a runnable ripgrep binary is on ``PATH``.

    Used by tests (to skip cleanly when rg is absent) and by the throwaway driver
    (to bail with a clear message rather than failing once per symbol). Cheap:
    one ``rg --version`` subprocess; a missing binary raises ``FileNotFoundError``
    which we swallow.
    """
    try:
        cp = subprocess.run(
            [ripgrep_bin, "--version"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return cp.returncode == 0


# ─── pure extraction ──────────────────────────────────────────────────────────


def _normalize_ripgrep_path(path: str) -> str:
    """Normalize a ripgrep-emitted path to the POSIX-relative form stored in
    ``files.path``, so the hit lines up with the caller's ``path -> file_id`` map
    regardless of platform.

    Two platform-dependent tweaks to the raw path ripgrep prints when searching
    ``.`` from the repo root:

    * **Strip a leading ``./`` or ``.\\``.** ripgrep prepends ``.`` + the
      platform's path separator, so ``./pkg/thing.py`` on POSIX and
      ``.\\pkg\\thing.py`` on Windows. Both are exactly two chars and are dropped
      before the separator swap. Anything else (already-relative, or an absolute
      form some ripgrep builds emit) keeps its leading bytes untouched.
    * **Convert backslashes to forward slashes.** On Windows ripgrep separates
      path segments with ``\\``; every other pipeline stage (``files.path`` from
      the walker, the import graph's target resolution, this stage's caller-built
      ``path -> file_id`` map) stores and expects POSIX-style paths, so a Windows
      ``.\\pkg\\thing.py`` must become ``pkg/thing.py`` or its hits silently fail
      to match the map and get dropped on persist (no edge, counted ``dropped``).
      This swap is load-bearing on Windows; the prefix strip alone would leave
      ``pkg\\thing.py``, which still misses the POSIX map.

    The caller runs ripgrep with ``cwd=repo_root`` searching ``.``, so one of the
    ``./`` / ``.\\`` forms is what we actually get — the prefix guard is
    defensive, but the backslash-to-slash swap is not.
    """
    if len(path) >= 2 and path[0] == "." and path[1] in ("/", "\\"):
        path = path[2:]
    return path.replace("\\", "/")


def _parse_ripgrep_output(
    raw: str,
    definition_path: str,
    definition_line: int,
    cap: int | None,
) -> list[ReferenceHit]:
    """Parse ripgrep's ``path:line:col:text`` match lines into hits, excluding the
    symbol's own definition line and capping the result. *Pure*: no subprocess, no
    DB, no filesystem — given ripgrep's stdout as a string it returns data. This
    is the deterministic, ripgrep-free core that the pure tests pin.

    Blank lines and anything that does not split into four ``:``-delimited parts
    (a malformed match / a stray line) are skipped. The ``(definition_path,
    definition_line)`` hit — the symbol's own ``def``/``class`` binding line,
    where ``line`` is the 1-indexed ``line_start`` tree-sitter recorded — is
    dropped so a definition is never recorded as a self-reference. A self-recursive
    call inside the body (a different line) survives, as intended.

    The cap is applied *after* the def-line exclusion (so a definition sitting
    within the first ``cap`` lines does not waste an edge slot) and is a defensive
    truncation — the real bound is ripgrep's ``--max-count``; this just keeps the
    contract honest on crafted/injected raw output.
    """
    hits: list[ReferenceHit] = []
    for raw_line in raw.split("\n"):
        if not raw_line:
            continue
        parts = raw_line.split(":", 3)
        if len(parts) < 4:
            # Not a `path:line:col:text` match line (e.g. a truncated/prefixed
            # line ripgrep sometimes prints); skip rather than guess fields.
            continue
        path, line_s, col_s, text = parts
        path = _normalize_ripgrep_path(path)
        try:
            line = int(line_s)
            col = int(col_s)
        except ValueError:
            continue  # non-numeric line/column — not a match line, skip
        if path == definition_path and line == definition_line:
            continue  # the symbol's own binding — not a reference to itself
        hits.append(ReferenceHit(path=path, line=line, column=col, line_text=text))
    if cap is not None:
        hits = hits[:cap]
    return hits


def find_references(
    name: str,
    repo_root: Path | str,
    definition_path: str,
    definition_line: int,
    cap: int | None = DEFAULT_CAP,
    ripgrep_bin: str = RIPGREP_BIN,
) -> list[ReferenceHit]:
    """Run ripgrep for ``name`` (whole-word) across the repo's ``.py`` files and
    return the matching lines as hits, minus the symbol's own definition line and
    capped at ``cap``. *Pure wrt DB*: no session, no commit — it shells out to
    ripgrep, so it touches the filesystem but owns no database transaction. This
    is the unit-testable core for the future ``pipeline.py`` orchestrator.

    ripgrep flags:

    * ``-w`` — whole-word match (``Flask`` does not match ``FlaskClass``; the
      graph's node is the *name*, not a substring of a larger identifier).
    * ``--fixed-strings`` — treat ``name`` literally (Python identifiers have no
      regex metachars, but this is belt-and-suspenders against accidental regex
      interpretation).
    * ``--type py`` — only Python files (MVP scope: the symbols come from
      ``.py``, and a ``target_type="file"`` edge must resolve to an indexed
      ``.py`` file).
    * ``--line-number --column --no-heading --color never`` — the
      ``path:line:col:text`` text format :func:`_parse_ripgrep_output` parses.
    * ``--max-count <cap>`` — stop after ``cap`` matching lines (the efficiency
      bound; ``self``-tier names that match thousands of lines are capped at the
      source rather than after reading the whole repo). Omitted when ``cap is
      None`` (unbounded).
    * ``-w -F -- <name> .`` — run with ``cwd=repo_root`` searching ``.``, so the
      printed paths are repo-root-relative (a leading ``./`` / ``.\\`` is stripped
      and backslashes are converted to forward slashes in
      :func:`_parse_ripgrep_output`) and line up with the POSIX ``files.path``.

    ripgrep respects ``.gitignore`` / ``.ignore`` by default, so it skips VCS /
    venv / build dirs the file-walker also prunes; a hit in a file the walker did
    *not* index simply fails to resolve on persist and is dropped — the graph
    never gains an edge to a non-indexed file. (A minor divergence from the
    walker's hardcoded ``PRUNED_DIRS`` is self-correcting, never wrong.)

    Raises ``FileNotFoundError`` if the ripgrep binary is missing (let the caller
    decide — the driver prechecks with :func:`ripgrep_available`, tests skip) and
    ``RuntimeError`` on ripgrep exit code ``>= 2`` (a genuine failure: bad flag,
    binary file, etc.). Exit code ``1`` (no matches) is the normal no-result path
    and returns ``[]``.
    """
    args = [
        ripgrep_bin,
        "-w",
        "--fixed-strings",
        "--type", "py",
        "--line-number",
        "--column",
        "--no-heading",
        "--color", "never",
    ]
    if cap is not None:
        args += ["--max-count", str(cap)]
    # `--` ends option parsing so a (hypothetical) name beginning with `-` is
    # treated as the pattern, then `.` is the search root.
    args += ["--", name, "."]

    # cwd=repo_root + search "." → paths print relative as `./rel/path.py` on
    # POSIX / `.\rel\path.py` on Windows, which _normalize_ripgrep_path converts
    # to the POSIX-rel form stored in `files.path` (strip the `.`-prefix, swap
    # `\` → `/`). text+utf-8 so non-ASCII in source lines (paths, comments)
    # survives the parse.
    cp = subprocess.run(
        args,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # rg exit: 0 = matches found, 1 = no matches, >= 2 = error.
    if cp.returncode == 1:
        return []
    if cp.returncode != 0:
        raise RuntimeError(
            f"ripgrep failed (exit {cp.returncode}) for name={name!r}: "
            f"{cp.stderr.strip()}"
        )
    return _parse_ripgrep_output(
        cp.stdout, definition_path, definition_line, cap
    )


# ─── persistence wrapper ──────────────────────────────────────────────────────


def index_symbol_references(
    symbol_id: int,
    hits: list[ReferenceHit],
    path_to_file_id: dict[str, int],
    repository_id: int,
    session: Session,
) -> ReferenceIndexResult:
    """Write one ``edge_type="references"`` row per distinct *file* in ``hits``
    that resolves via ``path_to_file_id``. Returns a :class:`ReferenceIndexResult`
    (edges added / hits dropped). Does **not commit** — the caller owns the
    transaction (mirrors :func:`app.indexing.walker.walk_repository` and
    :func:`app.indexing.import_graph.index_file_imports`, SDD §7 step 9).

    ``path_to_file_id`` is the trivial analog of :class:`ImportResolver`: a plain
    ``{posix_path: file_id}`` map the caller builds once from the repo's ``files``
    rows. Here resolution is a dict lookup, so a dedicated class would be ceremony
    — the import graph earns its resolver class with real (relative-import,
    src-layout) logic; the reference graph only needs an O(1) path→id lookup.

    Every emitted edge is ``source_type="symbol"`` (the ``symbol_id``) →
    ``target_type="file"`` (the resolved ``file_id``) with ``target_label`` NULL
    (``target_label`` is for *external package* targets per SDD §11; a reference
    to a repo file carries the file's ``id``, not a label). A hit whose path is
    absent from the map (a file the walker never indexed) yields no edge — it is
    counted ``dropped`` and skipped, not flagged external. Dedup is by resolved
    ``target_id`` (identical to dedup-by-path since the map is 1:1), so N hits in
    the same referencing file collapse to the single ``symbol → file`` edge the
    graph models — the cap's job is to bound this, not to inflate it.

    No ``flush`` is needed: ``symbol_id``, ``target_id`` and ``repository_id``
    all reference already-persisted rows. A ``references`` edge whose target file
    is the symbol's *own* file (the name appears on a non-binding line in its own
    module) is a legitimate self-adjacency and is emitted like any other — only
    the binding line itself was excluded upstream by :func:`find_references`.
    """
    added = 0
    dropped = 0
    seen: set[int] = set()
    for hit in hits:
        target_id = path_to_file_id.get(hit.path)
        if target_id is None:
            dropped += 1
            continue
        if target_id in seen:
            continue  # one references edge per distinct referencing file
        seen.add(target_id)
        session.add(
            Edge(
                repository_id=repository_id,
                source_type="symbol",
                source_id=symbol_id,
                target_type="file",
                target_id=target_id,
                target_label=None,
                edge_type="references",
            )
        )
        added += 1
    return ReferenceIndexResult(added=added, dropped=dropped)


def find_and_index_references(
    symbol_id: int,
    name: str,
    repo_root: Path | str,
    definition_path: str,
    definition_line: int,
    path_to_file_id: dict[str, int],
    repository_id: int,
    session: Session,
    cap: int | None = DEFAULT_CAP,
    ripgrep_bin: str = RIPGREP_BIN,
) -> ReferenceIndexResult:
    """Convenience: :func:`find_references` then :func:`index_symbol_references`.

    Composes the shell-out + persist for the future ``pipeline.py`` orchestrator.
    The throwaway batch driver calls the two pieces separately so it can inspect
    ``len(hits)`` and count how many symbols saturated the cap; this glue is for
    the path that does not need that detail. Same transaction-ownership rule as
    the other glues — this adds rows and returns; the caller commits.
    """
    hits = find_references(
        name, repo_root, definition_path, definition_line, cap=cap,
        ripgrep_bin=ripgrep_bin,
    )
    return index_symbol_references(
        symbol_id, hits, path_to_file_id, repository_id, session
    )
