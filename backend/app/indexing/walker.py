"""File-walker stage of the RepoLens indexing pipeline (SDD §7, steps 1–2).

Walks a repository tree from its root, prunes VCS / virtual-env / build-artifact
directories so they are never descended into (SDD §7 step 2: exclude
``node_modules``, ``.git``, build artifacts, binaries, and oversized files), and
records each in-scope ``.py`` file as a row in the ``files`` table with ``path``,
``language`` (``"python"`` for the MVP), ``loc`` (line count), and
``last_modified`` (filesystem mtime).

Scope BY DESIGN (SDD §0, §7 step 3): Python only here — TypeScript / JavaScript
indexing lands later. Parsing, the import/reference graph, git history, and
GitHub metadata are later pipeline steps and are NOT handled here.

The walker never commits: it adds ``File`` rows to the session it is handed so
the caller owns the transaction, matching SDD §7 step 9 (persist & mark ready as
one atomic unit) and keeping the walker trivially composable inside
``pipeline.py`` when that orchestrator is added.

Paths are stored relative to the repository root using POSIX separators, so
they are stable across platforms and identical to the paths the downstream
symbol-search / ripgrep / graph tools will reference.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import File

# Directory basenames that are never descended into (SDD §7 step 2). Matched
# against a single path segment, so a vendored copy of e.g. ``node_modules``
# anywhere in the tree is skipped the same way as the top-level one. This is a
# curated "common ignore patterns" set for the MVP; honoring a real ``.gitignore``
# is a later refinement (the SDD lists it, the user asked for the walk first).
PRUNED_DIRS: frozenset[str] = frozenset(
    {
        # VCS state — never source.
        ".git", ".hg", ".svn", ".bzr",
        # Python bytecode caches.
        "__pycache__",
        # Virtualenvs (project-local or committed by mistake).
        "venv", ".venv", "env", ".env",
        # Node dependency / asset dirs.
        "node_modules", "bower_components", "jspm_packages",
        # Build / packaging output.
        "build", "dist", "_build", "out", "target",
        ".eggs", "eggs",
        # Coverage / type-check / lint caches & tmps.
        ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox", ".nox",
        "htmlcov", "coverage",
        # Editor / OS clutter.
        ".idea", ".vscode",
    }
)

# Hard size ceiling per the SDD §7 step 2 default (configurable later). Files
# past this are skipped to bound parse time and row size; vendored or generated
# blobs that balloon a repo are the usual reason this trips.
MAX_FILE_SIZE_BYTES: int = 1 * 1024 * 1024

# Extension → RepoLens language slug, for the languages in MVP scope (SDD §0).
# Add ``.ts``, ``.tsx``, ``.js``, ``.jsx`` here when JS/TS indexing lands.
INDEXED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
}


def should_prune_dir(name: str) -> bool:
    """True when a directory named ``name`` (a single path segment) must not be
    descended into during the walk."""
    return name in PRUNED_DIRS


def language_for(suffix: str) -> str | None:
    """Map a file extension (lowercased, with leading dot) to its RepoLens
    language slug, or ``None`` if the extension is outside MVP indexing scope."""
    return INDEXED_EXTENSIONS.get(suffix.lower())


def _count_lines(path: Path) -> int:
    """Return the editor-style line count of ``path``.

    ``splitlines`` tolerates CRLF and a missing final newline and matches what a
    developer sees in their editor — the most intuitive reading of "LOC". The
    caller has already enforced the size guard, so the file is bounded.
    """
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def walk_repository(root: Path | str, repository_id: int, session: Session) -> int:
    """Walk ``root``, add one ``files`` row per in-scope ``.py`` file to
    ``session``, and return the number of rows added.

    Does NOT commit — the caller owns the transaction (SDD §7 step 9). Raises
    ``NotADirectoryError`` if ``root`` is not a directory.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"repository root is not a directory: {root_path}")

    added = 0
    for current_dir, dirnames, filenames in os.walk(root_path):
        # Prune in place so os.walk does not recurse into excluded dirs. This is
        # both faster and avoids indexing vendored trees that dwarf the repo.
        dirnames[:] = [d for d in dirnames if not should_prune_dir(d)]

        for filename in filenames:
            file_path = Path(current_dir, filename)
            language = language_for(file_path.suffix)
            if language is None:
                continue  # not a language we index (Python-only for the MVP)

            try:
                stat = file_path.stat()
            except OSError:
                continue  # raced (file vanished mid-walk); skip
            if stat.st_size > MAX_FILE_SIZE_BYTES:
                continue

            rel_path = file_path.relative_to(root_path)
            session.add(
                File(
                    repository_id=repository_id,
                    # POSIX separators keep paths stable cross-platform and
                    # identical to what ripgrep / symbol search return later.
                    path=rel_path.as_posix(),
                    language=language,
                    loc=_count_lines(file_path),
                    last_modified=datetime.fromtimestamp(stat.st_mtime),
                )
            )
            added += 1

    return added
