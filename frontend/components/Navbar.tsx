"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Code2,
  FolderGit2,
  Layers,
  MessageSquareCode,
  Moon,
  Plus,
  Settings,
  Sun,
} from "lucide-react";
import { RepositoryItem } from "@/lib/api";
import { useTheme } from "@/components/ThemeProvider";

interface NavbarProps {
  currentRepo?: RepositoryItem | null;
  onOpenAddModal?: () => void;
}

export default function Navbar({ currentRepo, onOpenAddModal }: NavbarProps) {
  const pathname = usePathname();
  const { resolvedTheme, toggleTheme } = useTheme();

  const isRepoPage = currentRepo && pathname.startsWith(`/repo/${currentRepo.id}`);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-surface-border bg-surface/90 backdrop-blur-md transition-colors">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Left Section: Brand & Breadcrumbs */}
        <div className="flex items-center gap-5">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600 text-white shadow-sm transition-transform group-hover:scale-105">
              <Layers className="h-4 w-4" />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-semibold tracking-tight text-foreground text-sm flex items-center gap-1.5">
                Repo<span className="text-primary-500">Lens</span>
              </span>
              <span className="text-[10px] font-mono font-medium text-text-muted bg-surface-raised px-1.5 py-0.5 rounded border border-surface-border">
                v1.0
              </span>
            </div>
          </Link>

          {/* Active repository pill */}
          {currentRepo && (
            <div className="hidden sm:flex items-center gap-2.5 text-xs text-text-muted border-l border-surface-border pl-5">
              <FolderGit2 className="h-3.5 w-3.5 text-primary-500" />
              <Link
                href={`/repo/${currentRepo.id}`}
                className="font-medium text-foreground hover:text-primary-500 transition-colors"
              >
                {currentRepo.name}
              </Link>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-medium capitalize ${
                  currentRepo.status === "ready"
                    ? "bg-accent-emerald/10 text-accent-emerald border border-accent-emerald/20"
                    : currentRepo.status === "indexing"
                    ? "bg-accent-amber/10 text-accent-amber border border-accent-amber/20 animate-pulse"
                    : "bg-accent-rose/10 text-accent-rose border border-accent-rose/20"
                }`}
              >
                {currentRepo.status}
              </span>
            </div>
          )}
        </div>

        {/* Center: Navigation Segment Tabs */}
        {isRepoPage && (
          <nav className="hidden md:flex items-center gap-1 bg-surface-raised p-1 rounded-xl border border-surface-border">
            <Link
              href={`/repo/${currentRepo.id}`}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                pathname === `/repo/${currentRepo.id}`
                  ? "bg-surface text-foreground shadow-sm font-semibold border border-surface-border/60"
                  : "text-text-muted hover:text-foreground"
              }`}
            >
              <Code2 className="h-3.5 w-3.5" />
              Overview
            </Link>

            <Link
              href={`/repo/${currentRepo.id}/chat`}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                pathname.includes("/chat")
                  ? "bg-surface text-foreground shadow-sm font-semibold border border-surface-border/60"
                  : "text-text-muted hover:text-foreground"
              }`}
            >
              <MessageSquareCode className="h-3.5 w-3.5 text-primary-500" />
              Chat & Graph
            </Link>

            <Link
              href={`/repo/${currentRepo.id}/settings`}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                pathname.includes("/settings")
                  ? "bg-surface text-foreground shadow-sm font-semibold border border-surface-border/60"
                  : "text-text-muted hover:text-foreground"
              }`}
            >
              <Settings className="h-3.5 w-3.5" />
              Settings
            </Link>
          </nav>
        )}

        {/* Right Section: Air-Gapped Status & Theme Toggle */}
        <div className="flex items-center gap-2.5">
          {/* Air-Gapped Local LLM Badge */}
          <div
            className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-surface-raised border border-surface-border text-text-muted"
            title="100% Air-Gapped & Local: All code analysis, parsing, and reasoning execute entirely on your machine. No code is transmitted to external servers."
          >
            <span className="h-2 w-2 rounded-full bg-accent-emerald animate-pulse" />
            <span className="text-foreground font-semibold">Local Ollama</span>
            <span className="text-[10px] text-text-subtle">· qwen3:8b</span>
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className="p-2 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised border border-transparent hover:border-surface-border transition-all"
            title={`Switch to ${resolvedTheme === "dark" ? "light" : "dark"} mode`}
          >
            {resolvedTheme === "dark" ? (
              <Sun className="h-4 w-4 text-accent-amber" />
            ) : (
              <Moon className="h-4 w-4 text-primary-600" />
            )}
          </button>

          {onOpenAddModal && (
            <button
              onClick={onOpenAddModal}
              className="flex items-center gap-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs font-medium px-3.5 py-1.5 rounded-lg shadow-sm transition-all hover:scale-[1.01] active:scale-[0.99]"
            >
              <Plus className="h-3.5 w-3.5" />
              <span>Add Repo</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
