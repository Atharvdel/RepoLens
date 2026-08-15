"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Boxes,
  Code,
  FileCode2,
  FolderGit2,
  GitGraph,
  Layers,
  Loader2,
  MessageSquareCode,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
} from "lucide-react";
import AddRepoModal from "@/components/AddRepoModal";
import Navbar from "@/components/Navbar";
import { api, RepositoryItem } from "@/lib/api";

export default function DashboardPage() {
  const [repositories, setRepositories] = useState<RepositoryItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const fetchRepos = async () => {
    try {
      const data = await api.listRepositories();
      setRepositories(data);
    } catch (err) {
      console.error("Failed to load repositories:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRepos();
    const interval = setInterval(() => {
      api.listRepositories().then((data) => {
        setRepositories(data);
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const filteredRepos = repositories.filter(
    (r) =>
      r.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.url_or_path.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen flex flex-col bg-background transition-colors">
      <Navbar onOpenAddModal={() => setIsAddModalOpen(true)} />

      {/* Header Banner */}
      <section className="border-b border-surface-border bg-surface px-4 py-12 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div className="max-w-2xl">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-primary-500/10 border border-primary-500/20 text-primary-500 text-xs font-mono mb-3">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Deterministic Code Intelligence</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
                Repository Knowledge Graph & Assistant
              </h1>
              <p className="mt-2 text-sm text-text-muted leading-relaxed">
                AST symbol extraction, dependency graph analysis, and multi-agent code comprehension with verified citations.
              </p>
            </div>

            <button
              onClick={() => setIsAddModalOpen(true)}
              className="flex items-center gap-2 bg-primary-600 hover:bg-primary-500 text-white font-medium text-xs px-4 py-2.5 rounded-lg shadow-sm transition-all hover:scale-[1.01] active:scale-[0.99]"
            >
              <Plus className="h-4 w-4" />
              <span>Index Repository</span>
            </button>
          </div>
        </div>
      </section>

      {/* Repositories Main List */}
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold text-foreground flex items-center gap-2">
              <FolderGit2 className="h-4 w-4 text-primary-500" />
              <span>Indexed Repositories ({repositories.length})</span>
            </h2>
            <button
              onClick={fetchRepos}
              title="Refresh repository list"
              className="p-1 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Search Filter */}
          <div className="relative flex items-center">
            <Search className="h-3.5 w-3.5 absolute left-3 text-text-subtle" />
            <input
              type="text"
              placeholder="Filter repositories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-surface border border-surface-border text-foreground placeholder:text-text-subtle text-xs rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary-500 w-full sm:w-64"
            />
          </div>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center p-20 text-center text-text-muted">
            <Loader2 className="h-7 w-7 text-primary-500 animate-spin mb-3" />
            <p className="text-xs">Loading repositories...</p>
          </div>
        ) : filteredRepos.length === 0 ? (
          <div className="surface-card rounded-2xl p-12 text-center max-w-lg mx-auto mt-6">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-primary-500/10 text-primary-500 border border-primary-500/20 mb-3">
              <Boxes className="h-6 w-6" />
            </div>
            <h3 className="text-sm font-semibold text-foreground">
              {searchQuery ? "No matching repositories" : "No repositories indexed yet"}
            </h3>
            <p className="mt-1.5 text-xs text-text-muted leading-relaxed">
              {searchQuery
                ? "Try searching for a different repository name."
                : "Add your local Git repository path or GitHub URL to extract symbols and build the graph."}
            </p>
            {!searchQuery && (
              <button
                onClick={() => setIsAddModalOpen(true)}
                className="mt-5 inline-flex items-center gap-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs font-medium px-3.5 py-2 rounded-lg shadow-sm transition-all"
              >
                <Plus className="h-3.5 w-3.5" />
                <span>Add Your First Repository</span>
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredRepos.map((repo) => (
              <div
                key={repo.id}
                className="group relative flex flex-col justify-between rounded-xl surface-card p-5 hover:border-primary-500/40 hover:shadow-card transition-all"
              >
                <div>
                  {/* Card Header */}
                  <div className="flex items-start justify-between gap-3 mb-3.5">
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-surface-raised border border-surface-border text-primary-500">
                        <FolderGit2 className="h-4 w-4" />
                      </div>
                      <div>
                        <Link
                          href={`/repo/${repo.id}`}
                          className="font-semibold text-sm text-foreground hover:text-primary-500 transition-colors truncate block max-w-[170px]"
                        >
                          {repo.name}
                        </Link>
                        <p className="text-[11px] font-mono text-text-subtle truncate max-w-[190px]">
                          {repo.url_or_path}
                        </p>
                      </div>
                    </div>

                    <span
                      className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded-full font-medium ${
                        repo.status === "ready"
                          ? "bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/20"
                          : repo.status === "indexing"
                          ? "bg-accent-amber/10 text-accent-amber border border-accent-amber/20 animate-pulse"
                          : "bg-accent-rose/10 text-accent-rose border border-accent-rose/20"
                      }`}
                    >
                      {repo.status}
                    </span>
                  </div>

                  {/* Metrics Row */}
                  <div className="grid grid-cols-3 gap-2 p-2.5 rounded-lg bg-surface-raised border border-surface-border text-xs mb-3.5">
                    <div>
                      <span className="text-text-subtle block text-[10px]">Files</span>
                      <span className="font-mono font-medium text-foreground">{repo.file_count}</span>
                    </div>
                    <div>
                      <span className="text-text-subtle block text-[10px]">Symbols</span>
                      <span className="font-mono font-medium text-primary-500">{repo.symbol_count}</span>
                    </div>
                    <div>
                      <span className="text-text-subtle block text-[10px]">LOC</span>
                      <span className="font-mono font-medium text-accent-emerald">
                        {repo.total_loc.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Card Actions Footer */}
                <div className="flex items-center justify-between pt-3 border-t border-surface-border text-xs">
                  <span className="text-[11px] text-text-subtle">
                    {repo.indexed_at ? `Indexed ${new Date(repo.indexed_at).toLocaleDateString()}` : "Pending"}
                  </span>
                  <div className="flex items-center gap-2">
                    <Link
                      href={`/repo/${repo.id}/chat`}
                      className="inline-flex items-center gap-1 text-primary-500 hover:text-primary-600 font-medium text-xs py-1 px-2 rounded-md hover:bg-surface-raised transition-colors"
                    >
                      <MessageSquareCode className="h-3.5 w-3.5" />
                      <span>Chat</span>
                    </Link>
                    <Link
                      href={`/repo/${repo.id}`}
                      className="inline-flex items-center gap-1 text-text-muted hover:text-foreground font-medium text-xs py-1 px-2 rounded-md hover:bg-surface-raised transition-colors"
                    >
                      <span>Overview</span>
                      <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      <AddRepoModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSuccess={(repo) => setRepositories([repo, ...repositories])}
      />
    </div>
  );
}
