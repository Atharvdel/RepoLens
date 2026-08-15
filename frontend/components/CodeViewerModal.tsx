"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Code,
  Copy,
  ExternalLink,
  FileCode,
  Folder,
  FolderGit2,
  FolderOpen,
  GitCommit,
  Layers,
  Loader2,
  PanelLeft,
  Search,
  Users,
  X,
} from "lucide-react";
import { api, FileDetailResponse } from "@/lib/api";

interface CodeViewerModalProps {
  repoId: number;
  filePath: string | null;
  onClose: () => void;
}

export default function CodeViewerModal({
  repoId,
  filePath,
  onClose,
}: CodeViewerModalProps) {
  const [currentPath, setCurrentPath] = useState<string | null>(filePath);
  const [detail, setDetail] = useState<FileDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"code" | "symbols" | "history">("code");
  const [copied, setCopied] = useState(false);
  const [allFiles, setAllFiles] = useState<string[]>([]);
  const [showFileTree, setShowFileTree] = useState(true);
  const [fileFilter, setFileFilter] = useState("");
  const [expandedDirs, setExpandedDirs] = useState<Record<string, boolean>>({});

  // Sync current path on prop change
  useEffect(() => {
    setCurrentPath(filePath);
  }, [filePath]);

  // Load all repo files for the file tree
  useEffect(() => {
    if (!repoId) return;
    api.getGraph(repoId, { scope: "whole" })
      .then((g) => {
        const filePaths = g.nodes.map((n) => n.path).sort();
        setAllFiles(filePaths);
      })
      .catch(console.error);
  }, [repoId]);

  const lineRangeMatch = currentPath ? currentPath.match(/:(\d+)(?:-(\d+))?$/) : null;
  const targetStartLine = lineRangeMatch ? parseInt(lineRangeMatch[1], 10) : null;
  const targetEndLine = lineRangeMatch
    ? lineRangeMatch[2]
      ? parseInt(lineRangeMatch[2], 10)
      : targetStartLine
    : null;

  useEffect(() => {
    if (!currentPath) {
      setDetail(null);
      return;
    }

    setLoading(true);
    const cleanPath = currentPath.split(":")[0];

    api
      .getFileDetail(repoId, cleanPath, true)
      .then((data) => setDetail(data))
      .catch((err) => console.error("Error loading file detail:", err))
      .finally(() => setLoading(false));
  }, [repoId, currentPath]);

  const copyCode = () => {
    if (!detail?.content) return;
    navigator.clipboard.writeText(detail.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const filteredFiles = useMemo(() => {
    if (!fileFilter.trim()) return allFiles;
    const q = fileFilter.toLowerCase();
    return allFiles.filter((f) => f.toLowerCase().includes(q));
  }, [allFiles, fileFilter]);

  if (!filePath) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-6xl h-[88vh] flex flex-col rounded-2xl bg-surface border border-surface-border shadow-elevation overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-surface-border bg-surface-raised/40">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setShowFileTree(!showFileTree)}
              title="Toggle File Explorer Sidebar"
              className={`p-1.5 rounded-lg border transition-all ${
                showFileTree
                  ? "bg-primary-500/10 text-primary-500 border-primary-500/20"
                  : "bg-surface-raised text-text-muted border-surface-border hover:text-foreground"
              }`}
            >
              <PanelLeft className="h-4 w-4" />
            </button>
            <div className="min-w-0">
              <h2 className="text-xs sm:text-sm font-semibold text-foreground font-mono truncate">{currentPath}</h2>
              {detail && (
                <div className="flex items-center gap-2.5 text-[10px] text-text-muted mt-0.5 font-mono">
                  <span>Lang: <strong className="text-foreground">{detail.language}</strong></span>
                  <span>•</span>
                  <span>LOC: <strong className="text-foreground">{detail.loc}</strong></span>
                  <span>•</span>
                  <span>Symbols: <strong className="text-primary-500 font-bold">{detail.symbols.length}</strong></span>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            {/* Tabs */}
            <div className="flex items-center gap-1 bg-surface-raised p-1 rounded-xl border border-surface-border text-xs">
              <button
                onClick={() => setActiveTab("code")}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === "code"
                    ? "bg-surface text-foreground shadow-sm font-semibold border border-surface-border/60"
                    : "text-text-muted hover:text-foreground"
                }`}
              >
                Source Code
              </button>
              <button
                onClick={() => setActiveTab("symbols")}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === "symbols"
                    ? "bg-surface text-foreground shadow-sm font-semibold border border-surface-border/60"
                    : "text-text-muted hover:text-foreground"
                }`}
              >
                Symbols ({detail?.symbols.length || 0})
              </button>
              <button
                onClick={() => setActiveTab("history")}
                className={`px-3 py-1 rounded-lg font-medium transition-all ${
                  activeTab === "history"
                    ? "bg-surface text-foreground shadow-sm font-semibold border border-surface-border/60"
                    : "text-text-muted hover:text-foreground"
                }`}
              >
                History ({detail?.recent_commits.length || 0})
              </button>
            </div>

            {/* Copy Button */}
            {detail?.content && (
              <button
                onClick={copyCode}
                title="Copy code"
                className="p-1.5 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised border border-transparent hover:border-surface-border transition-colors"
              >
                {copied ? <Check className="h-4 w-4 text-accent-emerald" /> : <Copy className="h-4 w-4" />}
              </button>
            )}

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors ml-1"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content Body: Split Explorer & Editor */}
        <div className="flex-1 flex overflow-hidden bg-background">
          {/* Collapsible File Tree Sidebar */}
          {showFileTree && (
            <aside className="w-64 border-r border-surface-border bg-surface/50 flex flex-col flex-shrink-0 animate-in slide-in-from-left duration-200">
              {/* File Search */}
              <div className="p-2.5 border-b border-surface-border">
                <div className="relative flex items-center">
                  <Search className="h-3 w-3 absolute left-2.5 text-text-subtle pointer-events-none" />
                  <input
                    type="text"
                    placeholder="Filter files..."
                    value={fileFilter}
                    onChange={(e) => setFileFilter(e.target.value)}
                    className="w-full bg-surface-raised border border-surface-border text-foreground placeholder-text-subtle text-[11px] rounded-lg pl-7 pr-2.5 py-1 focus:outline-none focus:ring-1 focus:ring-primary-500 font-mono"
                  />
                </div>
              </div>

              {/* File List */}
              <div className="flex-1 overflow-y-auto p-2 space-y-0.5 font-mono text-[11px]">
                {filteredFiles.length === 0 ? (
                  <div className="p-4 text-center text-text-muted text-[11px]">No matching files</div>
                ) : (
                  filteredFiles.map((file) => {
                    const isSelected = currentPath?.split(":")[0] === file;
                    const fileName = file.split("/").pop() || file;
                    const dirName = file.split("/").slice(0, -1).join("/");

                    return (
                      <button
                        key={file}
                        onClick={() => setCurrentPath(file)}
                        className={`w-full text-left px-2.5 py-1.5 rounded-lg flex items-center gap-2 transition-all truncate group ${
                          isSelected
                            ? "bg-primary-500/15 text-primary-600 dark:text-primary-400 font-semibold border border-primary-500/20"
                            : "text-text-muted hover:bg-surface-raised hover:text-foreground"
                        }`}
                        title={file}
                      >
                        <FileCode className={`h-3.5 w-3.5 flex-shrink-0 ${isSelected ? "text-primary-500" : "text-text-subtle group-hover:text-primary-500"}`} />
                        <div className="truncate flex-1">
                          <span className="truncate">{fileName}</span>
                          {dirName && (
                            <span className="text-[9px] text-text-subtle ml-1.5 opacity-60 truncate">
                              ({dirName})
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })
                )}
              </div>

              {/* Sidebar Footer */}
              <div className="p-2 border-t border-surface-border text-[10px] text-text-subtle font-mono flex items-center justify-between">
                <span>{allFiles.length} files indexed</span>
                <FolderGit2 className="h-3.5 w-3.5 text-text-muted" />
              </div>
            </aside>
          )}

          {/* Main Viewer Area */}
          <main className="flex-1 overflow-y-auto p-5 bg-background">
            {loading ? (
              <div className="h-full flex items-center justify-center">
                <Loader2 className="h-8 w-8 text-primary-500 animate-spin" />
              </div>
            ) : !detail ? (
              <div className="h-full flex flex-col items-center justify-center text-center text-text-muted">
                <FileCode className="h-10 w-10 mb-2 text-text-subtle" />
                <p className="text-sm">Unable to load file details</p>
              </div>
            ) : (
            <>
              {/* Tab 1: Source Code */}
              {activeTab === "code" && (
                <div className="h-full">
                  {detail.content ? (
                    <div className="rounded-xl bg-surface border border-surface-border font-mono text-xs overflow-x-auto shadow-sm">
                      <table className="w-full border-collapse">
                        <tbody>
                          {detail.content.split("\n").map((line, idx) => {
                            const lineNum = idx + 1;
                            const isHighlighted =
                              targetStartLine !== null &&
                              lineNum >= targetStartLine &&
                              (targetEndLine === null || lineNum <= targetEndLine);

                            return (
                              <tr
                                key={lineNum}
                                className={`transition-colors ${
                                  isHighlighted
                                    ? "bg-primary-500/15 text-primary-700 dark:text-primary-100 font-medium"
                                    : "hover:bg-surface-raised/60 text-foreground"
                                }`}
                              >
                                <td className="py-0.5 px-3 text-right text-text-subtle select-none text-[11px] w-12 border-r border-surface-border/40 font-mono">
                                  {lineNum}
                                </td>
                                <td className="py-0.5 px-4 font-mono text-xs whitespace-pre leading-relaxed">
                                  {line || " "}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="p-8 text-center text-text-muted">
                      <p className="text-xs">Raw file content not directly readable or file outside local tree.</p>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: Symbols & Imports */}
              {activeTab === "symbols" && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <Layers className="h-3.5 w-3.5 text-primary-500" /> Extracted Symbols ({detail.symbols.length})
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                      {detail.symbols.map((sym) => (
                        <div
                          key={sym.id}
                          className="p-3 rounded-xl surface-card flex flex-col justify-between"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-xs font-semibold text-foreground">{sym.name}</span>
                            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-primary-500/10 text-primary-500 border border-primary-500/20">
                              {sym.kind}
                            </span>
                          </div>
                          <div className="mt-2 text-[11px] text-text-subtle font-mono">
                            lines {sym.line_start}–{sym.line_end}
                          </div>
                          {sym.docstring && (
                            <p className="mt-1.5 text-[11px] text-text-muted line-clamp-2 italic">
                              "{sym.docstring}"
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Outgoing Imports */}
                  <div>
                    <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2">
                      Import Dependencies ({detail.imports.length})
                    </h3>
                    <div className="flex flex-wrap gap-1.5">
                      {detail.imports.map((imp, i) => (
                        <span
                          key={i}
                          className="px-2.5 py-1 rounded-lg bg-surface border border-surface-border text-xs font-mono text-foreground"
                        >
                          {imp}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Tab 3: Git History & Contributors */}
              {activeTab === "history" && (
                <div className="space-y-6">
                  {/* Top Contributors */}
                  <div>
                    <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <Users className="h-3.5 w-3.5 text-accent-emerald" /> Top Contributors
                    </h3>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                      {detail.top_contributors.map((c, i) => (
                        <div key={i} className="p-3 rounded-xl surface-card">
                          <span className="font-semibold text-xs text-foreground block">{c.author}</span>
                          <span className="text-[11px] text-text-muted font-mono">{c.commits} commits</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Recent Commits */}
                  <div>
                    <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
                      <GitCommit className="h-3.5 w-3.5 text-accent-cyan" /> Recent Commits Touching This File
                    </h3>
                    <div className="space-y-2">
                      {detail.recent_commits.map((cm, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-xl surface-card flex items-start justify-between gap-4"
                        >
                          <div>
                            <p className="text-xs text-foreground font-medium">{cm.message || "No commit message"}</p>
                            <div className="flex items-center gap-2 mt-1 text-[11px] text-text-muted font-mono">
                              <span>{cm.author}</span>
                              <span>•</span>
                              <span>{cm.date ? new Date(cm.date).toLocaleDateString() : ""}</span>
                            </div>
                          </div>
                          <span className="text-[10px] font-mono bg-surface-raised px-2 py-1 rounded border border-surface-border text-text-muted">
                            {cm.hash.substring(0, 7)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          </main>
        </div>
      </div>
    </div>
  );
}
