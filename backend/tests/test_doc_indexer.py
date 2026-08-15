"""Tests for doc_indexer.py (SDD §7 step 6, §10)."""
import tempfile
from pathlib import Path

from app.indexing.doc_indexer import _extract_title, parse_documents


def test_extract_title_markdown():
    md = "# Project Title\n\nSome introductory text..."
    assert _extract_title(md, default="Fallback") == "Project Title"


def test_extract_title_rst():
    rst = "Project Title\n=============\n\nSome text..."
    assert _extract_title(rst, default="Fallback") == "Project Title"


def test_extract_title_fallback():
    raw = "No header at all here, just plain text."
    assert _extract_title(raw, default="default.txt") == "default.txt"


def test_parse_documents_finds_readme_and_docs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create README.md at root
        (root / "README.md").write_text("# RepoLens\n\nArchitecture intelligence.", encoding="utf-8")
        # Create CONTRIBUTING.md at root
        (root / "CONTRIBUTING.md").write_text("## How to contribute\n\nSubmit a PR.", encoding="utf-8")
        # Create docs/guide.md
        docs_dir = root / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# User Guide\n\nStep 1: Install.", encoding="utf-8")
        # Create an ignored file
        (root / "ignored.bin").write_bytes(b"\x00\x01\x02")

        docs = parse_documents(root)
        paths = [d.path for d in docs]
        assert "README.md" in paths
        assert "CONTRIBUTING.md" in paths
        assert "docs/guide.md" in paths

        readme = next(d for d in docs if d.path == "README.md")
        assert readme.title == "RepoLens"
        assert "Architecture intelligence" in readme.content
