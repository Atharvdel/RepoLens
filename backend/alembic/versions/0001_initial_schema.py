"""initial schema — RepoLens tables per SDD §11

Creates the 10 RepoLens tables and their indexes in Postgres, matching the
ORM models under app/models one-to-one (so `alembic check` reports no drift):

    repositories, files, symbols, edges, documents, commits, file_commits,
    issues, chat_sessions, chat_messages

Revision ID: 0001
Revises:
Create Date: 2026-07-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── repositories ────────────────────────────────────────────────────────
    op.create_table(
        "repositories",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("url_or_path", sa.String, nullable=False),
        sa.Column("name", sa.String, nullable=False),
        sa.Column("default_branch", sa.String, nullable=True),
        # indexing | ready | failed (enforced by the application layer).
        sa.Column("status", sa.String, nullable=False),
        sa.Column("indexed_at", sa.DateTime, nullable=True),
        sa.Column("github_owner", sa.String, nullable=True),
        sa.Column("github_repo", sa.String, nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── files ───────────────────────────────────────────────────────────────
    # Note: §11 specifies UNIQUE(repository_id, path); the unique constraint's
    # B-tree index (leading column repository_id) doubles as the FK index, so no
    # separate index on repository_id is added here (matches the ORM model).
    op.create_table(
        "files",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.Integer, nullable=False),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("language", sa.String, nullable=False),
        sa.Column("loc", sa.Integer, nullable=False),
        sa.Column("last_modified", sa.DateTime, nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("repository_id", "path", name="uq_files_repository_id_path"),
    )

    # ── symbols ──────────────────────────────────────────────────────────────
    op.create_table(
        "symbols",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("file_id", sa.Integer, nullable=False),
        sa.Column("name", sa.String, nullable=False),
        # class | function | method | variable.
        sa.Column("kind", sa.String, nullable=False),
        sa.Column("line_start", sa.Integer, nullable=False),
        sa.Column("line_end", sa.Integer, nullable=False),
        sa.Column("docstring", sa.Text, nullable=True),
        # Self-referential: methods nested inside a class symbol.
        sa.Column("parent_symbol_id", sa.Integer, nullable=True),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["parent_symbol_id"], ["symbols.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_symbols_file_id", "symbols", ["file_id"])
    op.create_index("ix_symbols_name", "symbols", ["name"])
    op.create_index("ix_symbols_parent_symbol_id", "symbols", ["parent_symbol_id"])

    # ── edges ────────────────────────────────────────────────────────────────
    op.create_table(
        "edges",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.Integer, nullable=False),
        # file | symbol
        sa.Column("source_type", sa.String, nullable=False),
        sa.Column("source_id", sa.Integer, nullable=False),
        # file | symbol | external
        sa.Column("target_type", sa.String, nullable=False),
        sa.Column("target_id", sa.Integer, nullable=True),
        sa.Column("target_label", sa.String, nullable=True),
        # imports | references | contains
        sa.Column("edge_type", sa.String, nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_edges_repository_id_edge_type", "edges", ["repository_id", "edge_type"])
    op.create_index("ix_edges_source_type_source_id", "edges", ["source_type", "source_id"])

    # ── documents ────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.Integer, nullable=False),
        sa.Column("path", sa.String, nullable=False),
        sa.Column("title", sa.String, nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_repository_id", "documents", ["repository_id"])

    # ── commits ──────────────────────────────────────────────────────────────
    op.create_table(
        "commits",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.Integer, nullable=False),
        sa.Column("hash", sa.String, nullable=False),
        sa.Column("author", sa.String, nullable=False),
        sa.Column("date", sa.DateTime, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_commits_repository_id_date", "commits", ["repository_id", "date"])

    # ── issues ───────────────────────────────────────────────────────────────
    op.create_table(
        "issues",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.Integer, nullable=False),
        sa.Column("number", sa.Integer, nullable=False),
        sa.Column("title", sa.String, nullable=False),
        # open | closed (GitHub state).
        sa.Column("state", sa.String, nullable=False),
        sa.Column("labels", postgresql.ARRAY(sa.String), nullable=False),
        sa.Column("url", sa.String, nullable=False),
        sa.Column("linked_files", postgresql.ARRAY(sa.String), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_issues_repository_id", "issues", ["repository_id"])

    # ── chat_sessions ────────────────────────────────────────────────────────
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("repository_id", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_sessions_repository_id", "chat_sessions", ["repository_id"])

    # ── chat_messages ────────────────────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer, nullable=False),
        # user | assistant
        sa.Column("role", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        # Agent plan / tool-call trace (SDD §15) for the UI tool-trace panel.
        sa.Column("tool_trace", postgresql.JSONB, nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])

    # ── file_commits (join table) ────────────────────────────────────────────
    op.create_table(
        "file_commits",
        sa.Column("file_id", sa.Integer, nullable=False),
        sa.Column("commit_id", sa.Integer, nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"]),
        sa.ForeignKeyConstraint(["commit_id"], ["commits.id"]),
        sa.PrimaryKeyConstraint("file_id", "commit_id"),
    )
    # §11 explicitly specifies INDEX(file_id); redundant with the composite PK
    # (whose leading column is file_id) but included to honor the spec.
    op.create_index("ix_file_commits_file_id", "file_commits", ["file_id"])


def downgrade() -> None:
    op.drop_table("file_commits")
    op.drop_index("ix_chat_messages_session_id", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_chat_sessions_repository_id", table_name="chat_sessions")
    op.drop_table("chat_sessions")
    op.drop_index("ix_issues_repository_id", table_name="issues")
    op.drop_table("issues")
    op.drop_index("ix_commits_repository_id_date", table_name="commits")
    op.drop_table("commits")
    op.drop_index("ix_documents_repository_id", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_edges_source_type_source_id", table_name="edges")
    op.drop_index("ix_edges_repository_id_edge_type", table_name="edges")
    op.drop_table("edges")
    op.drop_index("ix_symbols_parent_symbol_id", table_name="symbols")
    op.drop_index("ix_symbols_name", table_name="symbols")
    op.drop_index("ix_symbols_file_id", table_name="symbols")
    op.drop_table("symbols")
    op.drop_table("files")
    op.drop_table("repositories")
