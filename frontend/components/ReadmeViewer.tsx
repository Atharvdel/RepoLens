"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, FileText } from "lucide-react";

interface ReadmeViewerProps {
  content?: string | null;
  path?: string;
}

export default function ReadmeViewer({ content, path = "README.md" }: ReadmeViewerProps) {
  if (!content) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center text-text-muted surface-card rounded-xl">
        <BookOpen className="h-8 w-8 text-text-subtle mb-2.5" />
        <p className="text-xs font-medium text-foreground">No README document found</p>
        <p className="text-[11px] text-text-muted mt-0.5">This repository does not have a parsed README or overview doc.</p>
      </div>
    );
  }

  return (
    <div className="surface-card rounded-xl overflow-hidden shadow-card transition-colors">
      <div className="flex items-center gap-2 px-5 py-3 border-b border-surface-border bg-surface-raised/40">
        <FileText className="h-4 w-4 text-primary-500" />
        <span className="text-xs font-semibold text-foreground font-mono">{path}</span>
      </div>
      <div className="p-6 overflow-y-auto max-h-[600px] markdown-body bg-surface">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
}
