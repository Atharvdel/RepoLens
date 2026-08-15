"""Tests for git_history.py and github_sync.py (SDD §7 step 7, step 8)."""
import tempfile
from pathlib import Path

import git
import pytest

from app.indexing.git_history import extract_commits_and_touches
from app.indexing.github_sync import extract_linked_files


def test_extract_linked_files():
    known = {"src/flask/app.py", "src/flask/blueprints.py", "docs/conf.py"}
    text = "This issue relates to `src/flask/app.py` and possibly `blueprints.py`."
    linked = extract_linked_files(text, known)
    assert linked is not None
    assert "src/flask/app.py" in linked
    assert "src/flask/blueprints.py" in linked


def test_extract_git_history_on_real_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        repo = git.Repo.init(repo_dir)

        # Create a file and commit
        f1 = repo_dir / "file1.py"
        f1.write_text("print('hello')", encoding="utf-8")
        repo.index.add(["file1.py"])
        repo.index.commit("Initial commit", author=git.Actor("Test Author", "test@example.com"))

        # Create second commit touching file2
        f2 = repo_dir / "file2.py"
        f2.write_text("x = 1", encoding="utf-8")
        repo.index.add(["file2.py"])
        repo.index.commit("Second commit", author=git.Actor("Test Author 2", "test2@example.com"))

        path_to_id = {"file1.py": 101, "file2.py": 102}
        try:
            commits = extract_commits_and_touches(repo_dir, path_to_id)
        finally:
            repo.close()

        assert len(commits) == 2
        assert any(c.message == "Second commit" and 102 in c.touched_file_ids for c in commits)
        assert any(c.message == "Initial commit" and 101 in c.touched_file_ids for c in commits)
