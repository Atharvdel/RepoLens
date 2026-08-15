"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  Layers,
  RotateCcw,
  Search,
  Terminal,
} from "lucide-react";

interface ToolTraceViewerProps {
  toolTrace?: Record<string, any> | null;
  nodeTrace?: string[];
  replansUsed?: number;
}

export default function ToolTraceViewer({
  toolTrace,
  nodeTrace,
  replansUsed = 0,
}: ToolTraceViewerProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!toolTrace && (!nodeTrace || nodeTrace.length === 0)) {
    return null;
  }

  const nodes = nodeTrace || toolTrace?.node_trace || [];
  const plan = toolTrace?.plan;
  const steps = plan?.steps || [];
  const searchResults = toolTrace?.search_results || [];
  const replans = replansUsed || toolTrace?.replans_used || 0;

  return (
    <div className="mt-3 rounded-xl border border-surface-border bg-surface-raised/60 overflow-hidden text-xs transition-colors">
      {/* Toggle Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2 hover:bg-surface-raised transition-colors text-text-muted hover:text-foreground"
      >
        <div className="flex items-center gap-2">
          <Cpu className="h-3.5 w-3.5 text-primary-500" />
          <span className="font-mono font-medium text-foreground">Agent Tool Trace</span>
          <span className="hidden sm:inline-flex items-center gap-1 text-[10px] text-text-subtle font-mono">
            ({nodes.join(" → ")})
          </span>
          {replans > 0 && (
            <span className="flex items-center gap-1 text-[10px] bg-accent-amber/10 text-accent-amber px-1.5 py-0.5 rounded border border-accent-amber/20">
              <RotateCcw className="h-2.5 w-2.5" /> replan: {replans}
            </span>
          )}
        </div>
        {isOpen ? (
          <ChevronDown className="h-3.5 w-3.5 text-text-subtle" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-text-subtle" />
        )}
      </button>

      {/* Expanded Content */}
      {isOpen && (
        <div className="p-3.5 border-t border-surface-border space-y-3 bg-surface font-mono text-[11px]">
          {/* Node Execution Sequence */}
          <div>
            <div className="text-text-muted font-semibold mb-1.5 flex items-center gap-1.5">
              <Layers className="h-3 w-3 text-primary-500" /> Execution Graph Flow
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              {nodes.map((node: string, idx: number) => (
                <div key={idx} className="flex items-center gap-1.5">
                  <span
                    className={`px-2 py-0.5 rounded-md border font-medium text-[10px] ${
                      node.includes("planner")
                        ? "bg-accent-purple/10 text-accent-purple border-accent-purple/20"
                        : node === "search"
                        ? "bg-primary-500/10 text-primary-500 border-primary-500/20"
                        : node === "context"
                        ? "bg-accent-cyan/10 text-accent-cyan border-accent-cyan/20"
                        : "bg-accent-emerald/10 text-accent-emerald border-accent-emerald/20"
                    }`}
                  >
                    {node}
                  </span>
                  {idx < nodes.length - 1 && <span className="text-text-subtle">→</span>}
                </div>
              ))}
            </div>
          </div>

          {/* Planned Steps */}
          {steps.length > 0 && (
            <div>
              <div className="text-text-muted font-semibold mb-1.5 flex items-center gap-1.5">
                <Terminal className="h-3 w-3 text-accent-cyan" /> Planned Tool Invocations ({steps.length})
              </div>
              <div className="space-y-1.5">
                {steps.map((st: any, i: number) => (
                  <div
                    key={i}
                    className="p-2 rounded-lg bg-surface-raised border border-surface-border text-foreground"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-primary-500 font-bold">#{i + 1}</span>
                      <span className="text-accent-purple uppercase text-[9px] px-1 py-0.2 rounded bg-accent-purple/10 border border-accent-purple/20">
                        {st.agent || "search"}
                      </span>
                      <span className="text-accent-amber font-semibold">{st.tool}</span>
                    </div>
                    <div className="mt-1 text-text-muted text-[10px] break-all">
                      args: <span className="text-foreground">{JSON.stringify(st.args || {})}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Search Hits Details */}
          {searchResults.length > 0 && (
            <div>
              <div className="text-text-muted font-semibold mb-1 flex items-center gap-1.5">
                <Search className="h-3 w-3 text-primary-500" /> Search Tool Hits
              </div>
              <div className="p-2 rounded-lg bg-surface-raised border border-surface-border text-text-muted space-y-1">
                {searchResults.map((sr: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-[10px]">
                    <span className="text-foreground font-medium">{sr.tool}</span>
                    <span className="text-primary-500 font-bold">{sr.hits?.length || 0} hits</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
