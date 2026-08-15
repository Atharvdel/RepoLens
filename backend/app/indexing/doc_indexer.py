"""Documentation parser stage of the RepoLens indexing pipeline (SDD §7 step 6, §10).

Extracts README, CONTRIBUTING, and `docs/**/*.md` / `doc/**/*.md` documentation files
from the repository tree into the `documents` table for retrieval, citation,
and the repository overview screen.

Follows the RepoLens indexing pattern:
- `parse_documents(root)` is pure: reads files from disk, extracts title + content.
- `index_documents(root, repository_id, session)` persists `Document` rows into the session.
  Does NOT commit -- the caller owns the transaction (SDD §7 step 9).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import Document

# Documentation file patterns scanned at the root
ROOT_DOC_PATTERNS = {
    "readme.md",
    "readme.rst",
    "readme.txt",
    "readme",
    "contributing.md",
    "contributing.rst",
    "contributing.txt",
    "contributing",
    "license.md",
    "architecture.md",
    "changelog.md",
}

# Directories searched recursively for documentation
DOC_DIRS = {"docs", "doc", "documentation"}
DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".txt"}


@dataclass
class ParsedDocument:
    """In-memory representation of an extracted documentation file."""

    path: str
    title: str
    content: str


def _extract_title(content: str, default: str) -> str:
    """Extract the first top-level Markdown / RST heading from content, or default."""
    lines = content.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Markdown # Heading
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
        # RST Underlined Heading (e.g. Header \n ======)
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if len(next_line) >= 3 and all(c in "=-~^" for c in next_line) and len(next_line) >= len(stripped):
                if stripped and not stripped.startswith("="):
                    return stripped
    return default


def parse_documents(root: Path | str) -> list[ParsedDocument]:
    """Scan ``root`` for documentation files and parse their content and titles.

    Pure function wrt DB: reads disk, returns list of ParsedDocument.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        return []

    docs: list[ParsedDocument] = []
    seen_paths: set[str] = set()

    for current_dir, dirnames, filenames in os.walk(root_path):
        # Prune hidden and non-relevant directories
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in {
                "node_modules", "venv", ".venv", "__pycache__", "build", "dist",
                ".next", ".nuxt", ".turbo", ".vercel", ".output",
            }
        ]

        rel_dir = Path(current_dir).relative_to(root_path)
        is_root = str(rel_dir) == "."
        is_in_doc_dir = any(part.lower() in DOC_DIRS for part in rel_dir.parts)

        for filename in filenames:
            file_path = Path(current_dir, filename)
            rel_file_path = file_path.relative_to(root_path).as_posix()
            lower_name = filename.lower()
            ext = file_path.suffix.lower()

            should_include = False
            if is_root and lower_name in ROOT_DOC_PATTERNS:
                should_include = True
            elif is_in_doc_dir and ext in DOC_EXTENSIONS:
                should_include = True
            elif ext in DOC_EXTENSIONS and any(term in lower_name for term in ["readme", "contributing", "guide", "spec", "architecture"]):
                should_include = True

            if should_include and rel_file_path not in seen_paths:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    if not content.strip() or len(content) > 5 * 1024 * 1024:
                        continue
                    title = _extract_title(content, default=filename)
                    docs.append(
                        ParsedDocument(
                            path=rel_file_path,
                            title=title,
                            content=content,
                        )
                    )
                    seen_paths.add(rel_file_path)
                except Exception:
                    continue

    # If no README was found, check for package.json or pyproject.toml at root or 1st subfolder
    has_readme = any("readme" in d.path.lower() for d in docs)
    if not has_readme:
        for pkg_candidate in list(root_path.glob("package.json")) + list(root_path.glob("*/package.json")):
            try:
                content = pkg_candidate.read_text(encoding="utf-8", errors="replace")
                rel_path = pkg_candidate.relative_to(root_path).as_posix()
                if content.strip() and rel_path not in seen_paths:
                    import json
                    pkg = json.loads(content)
                    desc = pkg.get("description", "")
                    deps = list((pkg.get("dependencies") or {}).keys())
                    scripts = list((pkg.get("scripts") or {}).keys())
                    summary = f"# Project Overview ({pkg.get('name', 'Package')})\n\n"
                    if desc:
                        summary += f"Description: {desc}\n\n"
                    if deps:
                        summary += f"Key Dependencies: {', '.join(deps)}\n\n"
                    if scripts:
                        summary += f"Available Scripts: {', '.join(scripts)}\n\n"
                    docs.append(
                        ParsedDocument(
                            path=rel_path,
                            title=f"Package Configuration ({rel_path})",
                            content=summary + "```json\n" + content[:2000] + "\n```",
                        )
                    )
                    seen_paths.add(rel_path)
                    break
            except Exception:
                pass

    # Sort docs: README first, then alphabetically by path
    def _sort_key(d: ParsedDocument) -> tuple[int, str]:
        p = d.path.lower()
        if "readme" in p:
            return (0, p)
        if "package.json" in p or "pyproject" in p:
            return (1, p)
        if "contributing" in p:
            return (2, p)
        return (3, p)

    docs.sort(key=_sort_key)
    return docs


def index_documents(root: Path | str, repository_id: int, session: Session) -> int:
    """Extract documentation files from ``root`` and add ``Document`` rows to ``session``.

    Does NOT commit -- caller owns the transaction.
    Returns the number of Document rows created.
    """
    parsed = parse_documents(root)
    count = 0
    for doc in parsed:
        row = Document(
            repository_id=repository_id,
            path=doc.path,
            title=doc.title,
            content=doc.content,
        )
        session.add(row)
        count += 1
    return count
