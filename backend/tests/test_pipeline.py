"""End-to-end integration tests for indexing pipeline.py (SDD §7, §16)."""
import tempfile
from pathlib import Path

import git
import pytest
import sqlalchemy as sa

from app.db import SessionLocal
from app.indexing.pipeline import clear_repository_data, run_indexing_pipeline
from app.models import (
    Commit,
    Document,
    Edge,
    File,
    Repository,
    Symbol,
    file_commits,
)


def test_full_pipeline_on_temp_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        repo_git = git.Repo.init(repo_path)

        # Create README.md
        (repo_path / "README.md").write_text("# Demo Project\n\nSample architecture.", encoding="utf-8")

        # Create Python files
        src_dir = repo_path / "src" / "demo"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("", encoding="utf-8")

        app_py = src_dir / "app.py"
        app_py.write_text(
            'from demo.utils import helper\n\nclass Server:\n    """Main server class."""\n    def start(self):\n        return helper()\n',
            encoding="utf-8",
        )

        utils_py = src_dir / "utils.py"
        utils_py.write_text(
            "def helper():\n    return 42\n",
            encoding="utf-8",
        )

        # Commit files to git
        repo_git.index.add(["README.md", "src/demo/__init__.py", "src/demo/app.py", "src/demo/utils.py"])
        repo_git.index.commit("Initial commit", author=git.Actor("Dev", "dev@example.com"))

        with SessionLocal() as session:
            # Create repository record
            db_repo = Repository(
                url_or_path=str(repo_path),
                name="demo-repo",
                status="indexing",
            )
            session.add(db_repo)
            session.commit()
            session.refresh(db_repo)
            repo_id = db_repo.id

            try:
                # Run the full pipeline
                progress = run_indexing_pipeline(repo_id, session)
                assert progress.stage == "ready"
                assert progress.files_indexed >= 3
                assert progress.symbols_indexed >= 2
                assert progress.docs_indexed >= 1
                assert progress.commits_indexed >= 1

                # Verify DB state
                refreshed_repo = session.get(Repository, repo_id)
                assert refreshed_repo.status == "ready"
                assert refreshed_repo.indexed_at is not None

                files = session.scalars(sa.select(File).where(File.repository_id == repo_id)).all()
                assert len(files) >= 3

                symbols = session.scalars(
                    sa.select(Symbol).join(File, Symbol.file_id == File.id).where(File.repository_id == repo_id)
                ).all()
                sym_names = [s.name for s in symbols]
                assert "Server" in sym_names
                assert "helper" in sym_names

                docs = session.scalars(sa.select(Document).where(Document.repository_id == repo_id)).all()
                assert any(d.title == "Demo Project" for d in docs)

                commits = session.scalars(sa.select(Commit).where(Commit.repository_id == repo_id)).all()
                assert len(commits) >= 1

            finally:
                # Cleanup DB rows
                clear_repository_data(repo_id, session)
                session.delete(session.get(Repository, repo_id))
                session.commit()
                repo_git.close()
