"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  ChevronRight,
  FileCode,
  Gauge,
  Layers,
  Link2,
  RotateCw,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import { api, ArchitectureHealthResponse } from "@/lib/api";

interface HealthScorecardProps {
  repoId: number;
  onSelectFile?: (path: string) => void;
}

export default function HealthScorecard({ repoId, onSelectFile }: HealthScorecardProps) {
  const [health, setHealth] = useState<ArchitectureHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [showHotspots, setShowHotspots] = useState(false);

  useEffect(() => {
    setLoading(true);
    api
      .getRepositoryHealth(repoId)
      .then((data) => setHealth(data))
      .catch((err) => console.error("Failed to load health report:", err))
      .finally(() => setLoading(false));
  }, [repoId]);

  if (loading) {
    return (
      <div className="surface-card rounded-2xl border border-surface-border p-6 animate-pulse space-y-4">
        <div className="h-6 w-48 bg-surface-raised rounded-lg" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="h-20 bg-surface-raised rounded-xl" />
          <div className="h-20 bg-surface-raised rounded-xl" />
          <div className="h-20 bg-surface-raised rounded-xl" />
          <div className="h-20 bg-surface-raised rounded-xl" />
        </div>
      </div>
    );
  }

  if (!health) return null;

  const getScoreColor = (score: number) => {
    if (score >= 85) return "text-accent-emerald border-accent-emerald/30 bg-accent-emerald/10";
    if (score >= 70) return "text-primary-500 border-primary-500/30 bg-primary-500/10";
    if (score >= 50) return "text-accent-amber border-accent-amber/30 bg-accent-amber/10";
    return "text-accent-rose border-accent-rose/30 bg-accent-rose/10";
  };

  return (
    <div className="surface-card rounded-2xl border border-surface-border p-6 shadow-subtle space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-primary-500/10 text-primary-500 border border-primary-500/20">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-foreground">Architecture Health & Tech Debt</h2>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-surface-raised text-text-muted border border-surface-border">
                Automated Graph Audit
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5">
              Continuous analysis of circular dependencies, coupling hotspots, and module encapsulation.
            </p>
          </div>
        </div>

        {/* Grade Dial Badge */}
        <div className="flex items-center gap-3 self-start sm:self-auto">
          <div className="text-right">
            <div className="text-[10px] uppercase font-mono tracking-wider text-text-subtle font-semibold">
              Health Grade
            </div>
            <div className="text-xs font-medium text-text-muted">
              Score: <strong className="text-foreground">{health.health_score}</strong>/100
            </div>
          </div>
          <div
            className={`h-12 w-12 rounded-2xl border flex items-center justify-center font-bold text-xl tracking-tight shadow-sm ${getScoreColor(
              health.health_score
            )}`}
          >
            {health.grade}
          </div>
        </div>
      </div>

      {/* 4 Health Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {health.metrics.map((m, idx) => (
          <div
            key={idx}
            className="p-3.5 rounded-xl bg-surface-raised/40 border border-surface-border flex flex-col justify-between space-y-2 hover:border-surface-border/80 transition-all"
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-text-muted">{m.name}</span>
              {m.status === "healthy" ? (
                <span className="flex items-center gap-1 text-[10px] font-medium text-accent-emerald bg-accent-emerald/10 px-1.5 py-0.5 rounded-full border border-accent-emerald/20">
                  <CheckCircle2 className="h-3 w-3" />
                  Pass
                </span>
              ) : (
                <span className="flex items-center gap-1 text-[10px] font-medium text-accent-amber bg-accent-amber/10 px-1.5 py-0.5 rounded-full border border-accent-amber/20">
                  <AlertTriangle className="h-3 w-3" />
                  Notice
                </span>
              )}
            </div>
            <div className="text-base font-bold text-foreground font-mono">{m.value}</div>
            <p className="text-[10px] text-text-subtle leading-relaxed line-clamp-2">{m.description}</p>
          </div>
        ))}
      </div>

      {/* Recommendations & Coupling Hotspots Accordion */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-2 border-t border-surface-border">
        {/* Recommendations */}
        <div className="space-y-2.5">
          <div className="text-xs font-semibold text-foreground flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-primary-500" />
            <span>Architectural Recommendations</span>
          </div>
          <div className="space-y-2">
            {health.recommendations.map((rec, i) => (
              <div
                key={i}
                className="text-xs p-2.5 rounded-xl bg-surface border border-surface-border text-text-muted flex items-start gap-2"
              >
                <div className="mt-0.5 flex-shrink-0">{rec.slice(0, 2)}</div>
                <div className="leading-relaxed">{rec.slice(2)}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Hotspots & Orphaned Files */}
        <div className="space-y-2.5">
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-foreground flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5 text-accent-amber" />
              <span>Coupling Hotspots ({health.hub_risks.length})</span>
            </div>
            <button
              onClick={() => setShowHotspots(!showHotspots)}
              className="text-[11px] text-primary-500 hover:underline flex items-center gap-0.5"
            >
              {showHotspots ? "Show Less" : "View All"}
              <ChevronRight className={`h-3 w-3 transform transition-transform ${showHotspots ? "rotate-90" : ""}`} />
            </button>
          </div>

          <div className="space-y-2">
            {health.hub_risks.slice(0, showHotspots ? 10 : 3).map((hub, i) => (
              <div
                key={i}
                onClick={() => onSelectFile?.(hub.path)}
                className="p-2.5 rounded-xl bg-surface border border-surface-border hover:border-primary-500/40 hover:bg-surface-raised/30 transition-all cursor-pointer flex items-center justify-between group"
              >
                <div className="min-w-0 pr-3">
                  <div className="text-xs font-mono text-foreground font-medium truncate group-hover:text-primary-500 transition-colors">
                    {hub.path}
                  </div>
                  <div className="text-[10px] text-text-subtle mt-0.5">
                    Imported by <strong className="text-foreground">{hub.in_degree} files</strong> • {hub.risk_level} Blast Radius
                  </div>
                </div>
                <div className="flex-shrink-0">
                  <span
                    className={`text-[9px] uppercase font-mono px-2 py-0.5 rounded-full font-bold ${
                      hub.risk_level === "High"
                        ? "bg-accent-rose/10 text-accent-rose border border-accent-rose/20"
                        : "bg-accent-amber/10 text-accent-amber border border-accent-amber/20"
                    }`}
                  >
                    {hub.risk_level} Risk
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
