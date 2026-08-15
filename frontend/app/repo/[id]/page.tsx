"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowRight,
  BookOpen,
  Boxes,
  Code2,
  FileCode,
  FileDown,
  FolderGit2,
  GitCommit,
  GitGraph,
  Layers,
  Loader2,
  MessageSquareCode,
  RotateCw,
  Settings,
  Sparkles,
} from "lucide-react";
import CodeViewerModal from "@/components/CodeViewerModal";
import HealthScorecard from "@/components/HealthScorecard";
import Navbar from "@/components/Navbar";
import ReadmeViewer from "@/components/ReadmeViewer";
import { api, RepositoryDetail } from "@/lib/api";

export default function RepoOverviewPage() {
  const params = useParams();
  const router = useRouter();
  const repoId = Number(params.id);

  const [repo, setRepo] = useState<RepositoryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedFileForModal, setSelectedFileForModal] = useState<string | null>(null);
  const [exportingReport, setExportingReport] = useState(false);

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

  const handleExportReport = async () => {
    if (!repo) return;
    setExportingReport(true);
    try {
      const reportData = await api.getRepositoryReport(repo.id);
      const blob = new Blob([reportData.markdown], { type: "text/markdown;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", reportData.filename || `${repo.name}_architecture_report.md`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export report:", err);
    } finally {
      setExportingReport(false);
    }
  };

  useEffect(() => {
    fetchRepo();
  }, [repoId]);

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
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
          <p className="text-text-muted">Repository not found</p>
          <Link href="/" className="mt-4 text-primary-500 text-sm hover:underline">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  // Calculate total LOC for language percentage bar
  const totalLangLoc = Object.values(repo.language_breakdown).reduce((a, b) => a + b, 0) || 1;

  return (
    <div className="min-h-screen flex flex-col bg-background transition-colors">
      <Navbar currentRepo={repo} />

      {/* Header Banner */}
      <section className="border-b border-surface-border bg-surface px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight">{repo.name}</h1>
                <span
                  className={`text-[10px] uppercase font-mono px-2.5 py-0.5 rounded-full font-medium ${
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
              <p className="text-xs font-mono text-text-muted">{repo.url_or_path}</p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={handleExportReport}
                disabled={exportingReport}
                className="flex items-center gap-2 surface-card hover:bg-surface-raised text-foreground text-xs font-medium px-3.5 py-2.5 rounded-lg border border-surface-border transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50"
                title="Download full executive architecture & analysis summary report in Markdown"
              >
                {exportingReport ? (
                  <Loader2 className="h-4 w-4 text-primary-500 animate-spin" />
                ) : (
                  <FileDown className="h-4 w-4 text-primary-500" />
                )}
                <span>{exportingReport ? "Generating Report..." : "Export Report"}</span>
              </button>

              <Link
                href={`/repo/${repo.id}/chat`}
                className="flex items-center gap-2 bg-primary-600 hover:bg-primary-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-sm transition-all hover:scale-[1.01]"
              >
                <MessageSquareCode className="h-4 w-4" />
                <span>Open Chat & Graph</span>
              </Link>

              <Link
                href={`/repo/${repo.id}/settings`}
                className="flex items-center gap-2 surface-card hover:bg-surface-raised text-foreground text-xs font-medium px-3.5 py-2.5 rounded-lg transition-colors"
              >
                <Settings className="h-4 w-4 text-text-muted" />
                <span>Settings</span>
              </Link>
            </div>
          </div>

          {/* Language Breakdown Bar */}
          {Object.keys(repo.language_breakdown).length > 0 && (
            <div className="mt-6 pt-6 border-t border-surface-border">
              <div className="flex items-center justify-between text-xs text-text-muted mb-2 font-mono">
                <span>Language Breakdown</span>
                <span>{repo.total_loc.toLocaleString()} Total LOC</span>
              </div>
              <div className="h-2 w-full flex rounded-full overflow-hidden bg-surface-raised border border-surface-border">
                {Object.entries(repo.language_breakdown).map(([lang, count], idx) => {
                  const pct = Math.max(2, Math.round((count / totalLangLoc) * 100));
                  const colors = ["#2563eb", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b"];
                  return (
                    <div
                      key={lang}
                      style={{ width: `${pct}%`, backgroundColor: colors[idx % colors.length] }}
                      title={`${lang}: ${count} LOC (${pct}%)`}
                      className="h-full"
                    />
                  );
                })}
              </div>
              <div className="flex flex-wrap items-center gap-4 mt-2 text-xs">
                {Object.entries(repo.language_breakdown).map(([lang, count], idx) => {
                  const pct = Math.round((count / totalLangLoc) * 100);
                  const colors = ["#2563eb", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b"];
                  return (
                    <div key={lang} className="flex items-center gap-1.5 text-text-muted text-[11px]">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: colors[idx % colors.length] }}
                      />
                      <span>{lang}</span>
                      <span className="font-mono text-text-subtle font-medium">({pct}%)</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
        {/* Metric Cards Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <div className="surface-card rounded-xl p-4">
            <div className="flex items-center gap-2.5 text-text-muted mb-2">
              <FileCode className="h-4 w-4 text-primary-500" />
              <span className="text-xs font-medium">Files</span>
            </div>
            <span className="font-mono text-xl font-bold text-foreground">{repo.file_count}</span>
          </div>

          <div className="surface-card rounded-xl p-4">
            <div className="flex items-center gap-2.5 text-text-muted mb-2">
              <Layers className="h-4 w-4 text-primary-500" />
              <span className="text-xs font-medium">AST Symbols</span>
            </div>
            <span className="font-mono text-xl font-bold text-primary-500">{repo.symbol_count}</span>
          </div>

          <div className="surface-card rounded-xl p-4">
            <div className="flex items-center gap-2.5 text-text-muted mb-2">
              <GitCommit className="h-4 w-4 text-accent-emerald" />
              <span className="text-xs font-medium">Git Commits</span>
            </div>
            <span className="font-mono text-xl font-bold text-accent-emerald">{repo.commits_count}</span>
          </div>

          <div className="surface-card rounded-xl p-4">
            <div className="flex items-center gap-2.5 text-text-muted mb-2">
              <BookOpen className="h-4 w-4 text-accent-purple" />
              <span className="text-xs font-medium">Docs Indexed</span>
            </div>
            <span className="font-mono text-xl font-bold text-accent-purple">{repo.docs_count}</span>
          </div>

          <div className="surface-card rounded-xl p-4">
            <div className="flex items-center gap-2.5 text-text-muted mb-2">
              <Sparkles className="h-4 w-4 text-accent-cyan" />
              <span className="text-xs font-medium">Key Files</span>
            </div>
            <span className="font-mono text-xl font-bold text-accent-cyan">{repo.key_files.length}</span>
          </div>
        </div>

        {/* Architecture Health & Tech Debt Scorecard */}
        <HealthScorecard repoId={repoId} onSelectFile={(path) => setSelectedFileForModal(path)} />

        {/* 2-Column Section: Key Central Files & Readme */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Key Central Architecture Files */}
          <div className="lg:col-span-1 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary-500" />
                <span>Central Files ({repo.key_files.length})</span>
              </h2>
              <span className="text-[10px] text-text-subtle font-mono">By in-degree rank</span>
            </div>

            <div className="space-y-2">
              {repo.key_files.length === 0 ? (
                <p className="text-xs text-text-muted surface-card p-4 rounded-xl text-center">
                  No key central files identified yet.
                </p>
              ) : (
                repo.key_files.map((file, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedFileForModal(file.path)}
                    className="w-full text-left p-3 rounded-xl surface-card hover:bg-surface-raised hover:border-primary-500/40 transition-all flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-2.5 truncate">
                      <FileCode className="h-4 w-4 text-primary-500 flex-shrink-0" />
                      <span className="font-mono text-xs text-foreground group-hover:text-primary-500 transition-colors truncate">
                        {file.path}
                      </span>
                    </div>
                    <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-primary-500/10 text-primary-500 border border-primary-500/20">
                      {file.in_degree} deps
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* README / Overview Viewer */}
          <div className="lg:col-span-2 space-y-4">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-primary-500" />
              <span>Repository Overview Documentation</span>
            </h2>
            <ReadmeViewer content={repo.readme_preview} />
          </div>
        </div>
      </main>

      <CodeViewerModal
        repoId={repoId}
        filePath={selectedFileForModal}
        onClose={() => setSelectedFileForModal(null)}
      />
    </div>
  );
}
