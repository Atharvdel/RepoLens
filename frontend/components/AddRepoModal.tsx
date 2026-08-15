"use client";

import { useState } from "react";
import { FolderGit2, Github, KeyRound, Loader2, Sparkles, X } from "lucide-react";
import { api, RepositoryItem } from "@/lib/api";

interface AddRepoModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (repo: RepositoryItem) => void;
}

export default function AddRepoModal({ isOpen, onClose, onSuccess }: AddRepoModalProps) {
  const [urlOrPath, setUrlOrPath] = useState("");
  const [name, setName] = useState("");
  const [githubToken, setGithubToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!urlOrPath.trim()) {
      setError("Please provide a repository path or Git URL");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const created = await api.createRepository({
        url_or_path: urlOrPath.trim(),
        name: name.trim() || undefined,
        github_token: githubToken.trim() || undefined,
      });
      onSuccess(created);
      onClose();
    } catch (err: any) {
      setError(err.message || "Failed to add repository");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-lg rounded-2xl bg-surface border border-surface-border p-6 shadow-elevation">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-surface-border">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-primary-500/10 text-primary-500 border border-primary-500/20">
              <FolderGit2 className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-foreground">Add Repository</h2>
              <p className="text-xs text-text-muted">Index code structure, AST symbols, and git history</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          {error && (
            <div className="p-3 rounded-lg bg-accent-rose/10 border border-accent-rose/25 text-accent-rose text-xs">
              {error}
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-foreground mb-1.5">
              Repository Location <span className="text-accent-rose">*</span>
            </label>
            <input
              type="text"
              value={urlOrPath}
              onChange={(e) => setUrlOrPath(e.target.value)}
              placeholder="e.g. C:\work\my-project or https://github.com/owner/repo"
              className="w-full px-3.5 py-2.5 rounded-lg bg-surface border border-surface-border text-foreground text-xs placeholder:text-text-subtle focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-colors"
            />
            <p className="mt-1 text-[11px] text-text-muted">
              Supports local folder paths or remote GitHub HTTPS/SSH clone URLs.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-foreground mb-1.5">
              Display Name <span className="text-text-muted font-normal">(optional)</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Flask, Express, RepoLens"
              className="w-full px-3.5 py-2.5 rounded-lg bg-surface border border-surface-border text-foreground text-xs placeholder:text-text-subtle focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 transition-colors"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-foreground mb-1.5">
              GitHub Personal Access Token <span className="text-text-muted font-normal">(optional)</span>
            </label>
            <div className="relative flex items-center">
              <KeyRound className="h-3.5 w-3.5 absolute left-3 text-text-subtle" />
              <input
                type="password"
                value={githubToken}
                onChange={(e) => setGithubToken(e.target.value)}
                placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                className="w-full pl-9 pr-3.5 py-2.5 rounded-lg bg-surface border border-surface-border text-foreground text-xs placeholder:text-text-subtle focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 font-mono"
              />
            </div>
          </div>

          <div className="flex items-center justify-end gap-3 pt-3 border-t border-surface-border">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs font-medium text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !urlOrPath.trim()}
              className="flex items-center gap-2 bg-primary-600 hover:bg-primary-500 disabled:opacity-50 text-white text-xs font-medium px-4 py-2 rounded-lg shadow-sm transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Adding & Starting Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>Start Indexing</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
