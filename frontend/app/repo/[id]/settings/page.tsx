"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileCode,
  FolderGit2,
  GitCommit,
  GitFork,
  Github,
  KeyRound,
  Layers,
  Loader2,
  RefreshCw,
  Search,
  Settings,
  Trash2,
} from "lucide-react";
import Navbar from "@/components/Navbar";
import { api, RepositoryDetail } from "@/lib/api";

export default function RepoSettingsPage() {
  const params = useParams();
  const router = useRouter();
  const repoId = Number(params.id);

  const [repo, setRepo] = useState<RepositoryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const fetchRepo = async () => {
    try {
      const data = await api.getRepository(repoId);
      setRepo(data);
    } catch (err) {
      console.error("Failed to load repo:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRepo();
    const interval = setInterval(fetchRepo, 3000);
    return () => clearInterval(interval);
  }, [repoId]);

  const handleReindex = async () => {
    setReindexing(true);
    try {
      await api.reindexRepository(repoId);
      await fetchRepo();
    } catch (err) {
      console.error("Failed to trigger reindex:", err);
    } finally {
      setReindexing(false);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await api.deleteRepository(repoId);
      router.push("/");
    } catch (err) {
      console.error("Failed to delete repository:", err);
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="h-7 w-7 text-primary-500 animate-spin" />
        </div>
      </div>
    );
  }

  if (!repo) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Navbar />
        <div className="flex-1 flex items-center justify-center text-text-muted">
          Repository not found
        </div>
      </div>
    );
  }

  const pipelineStages = [
    { title: "1. Repository Clone / Pull", desc: "Local path verification or Git remote clone", icon: FolderGit2 },
    { title: "2. File System Walker", desc: "Walks eligible source files and calculates line counts", icon: FileCode },
    { title: "3. AST Tree-sitter Parser", desc: "Extracts functions, classes, methods, and docstrings", icon: Layers },
    { title: "4. Import Graph Indexer", desc: "Resolves internal and external package import dependencies", icon: GitFork },
    { title: "5. Symbol Reference Indexer", desc: "Ripgrep scans for cross-file caller and reference edges", icon: Search },
    { title: "6. Documentation Parser", desc: "Indexes README, CONTRIBUTING, and markdown documentation", icon: CheckCircle2 },
    { title: "7. Git History & Contributor Sync", desc: "Parses Git commits and maps authors to files", icon: GitCommit },
    { title: "8. GitHub Sync", desc: "Fetches issues and pull requests linked to files", icon: Github },
    { title: "9. Status Ready", desc: "Transaction committed; graph & multi-agent query layer active", icon: CheckCircle2 },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background transition-colors">
      <Navbar currentRepo={repo} />

      <main className="flex-1 mx-auto w-full max-w-4xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-xl font-bold text-foreground">Repository Settings & Status</h1>
          <p className="text-xs text-text-muted mt-1">
            Manage background indexing pipeline, GitHub synchronization tokens, and repository data.
          </p>
        </div>

        {/* Indexing Status Card */}
        <div className="surface-card rounded-xl p-6 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-primary-500/10 text-primary-500 border border-primary-500/20">
                <Clock className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-sm font-semibold text-foreground">Pipeline Indexing Status</h2>
                <p className="text-xs text-text-muted">
                  Current state: <strong className="text-foreground capitalize">{repo.status}</strong>
                </p>
              </div>
            </div>

            <button
              onClick={handleReindex}
              disabled={reindexing || repo.status === "indexing"}
              className="flex items-center gap-2 bg-primary-600 hover:bg-primary-500 disabled:opacity-50 text-white text-xs font-medium px-3.5 py-2 rounded-lg shadow-sm transition-all"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${reindexing ? "animate-spin" : ""}`} />
              <span>{reindexing ? "Triggering..." : "Re-Index Repository"}</span>
            </button>
          </div>

          {/* 9-Stage Pipeline Tracker */}
          <div className="border-t border-surface-border pt-5">
            <h3 className="text-xs font-semibold text-foreground mb-3">9-Stage Automated Pipeline</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {pipelineStages.map((st, i) => {
                const Icon = st.icon;
                const isReady = repo.status === "ready";
                return (
                  <div
                    key={i}
                    className="p-3 rounded-lg bg-surface-raised border border-surface-border flex items-start gap-3"
                  >
                    <div className="p-1.5 rounded-md bg-surface border border-surface-border text-primary-500 mt-0.5">
                      <Icon className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-xs text-foreground">{st.title}</span>
                        {isReady && <CheckCircle2 className="h-3 w-3 text-accent-emerald" />}
                      </div>
                      <p className="text-[11px] text-text-muted mt-0.5">{st.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* GitHub Token Card */}
        <div className="surface-card rounded-xl p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-surface-raised border border-surface-border text-foreground">
              <KeyRound className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-foreground">GitHub Integration Token</h2>
              <p className="text-xs text-text-muted">
                Optional Personal Access Token (PAT) for fetching GitHub Issues and Pull Requests.
              </p>
            </div>
          </div>

          <div className="flex gap-2">
            <input
              type="password"
              placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
              className="flex-1 px-3.5 py-2 rounded-lg bg-surface border border-surface-border text-foreground text-xs placeholder:text-text-subtle focus:outline-none focus:border-primary-500 font-mono"
            />
            <button className="px-4 py-2 rounded-lg bg-surface-raised hover:bg-surface-border text-foreground text-xs font-medium border border-surface-border transition-colors">
              Save Token
            </button>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="surface-card rounded-xl p-6 border-accent-rose/30 space-y-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-accent-rose/10 text-accent-rose border border-accent-rose/20">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-foreground">Danger Zone</h2>
              <p className="text-xs text-text-muted">
                Permanently delete this repository, parsed AST symbols, and graph tables.
              </p>
            </div>
          </div>

          <div className="pt-2">
            {confirmDelete ? (
              <div className="flex items-center gap-3">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="bg-accent-rose hover:bg-accent-rose/90 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors"
                >
                  {deleting ? "Deleting..." : "Confirm Delete"}
                </button>
                <button
                  onClick={() => setConfirmDelete(false)}
                  className="text-xs text-text-muted hover:text-foreground px-3 py-2"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirmDelete(true)}
                className="flex items-center gap-2 bg-accent-rose/10 hover:bg-accent-rose/20 text-accent-rose border border-accent-rose/20 text-xs font-semibold px-4 py-2 rounded-lg transition-colors"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>Delete Repository</span>
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
