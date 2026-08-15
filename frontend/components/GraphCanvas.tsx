"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  Code2,
  Compass,
  Copy,
  Download,
  Expand,
  FileCode,
  FileDown,
  Layers,
  Map,
  Maximize2,
  Minimize2,
  RefreshCw,
  Search,
  Sparkles,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { GraphEdge, GraphNode } from "@/lib/api";
import { useTheme } from "@/components/ThemeProvider";

interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  highlightedPath?: string | null;
  onSelectNode?: (node: GraphNode) => void;
  onAskAboutFile?: (path: string) => void;
}

// Module color palette
const MODULE_COLORS = [
  "#3b82f6", // blue
  "#10b981", // emerald
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#f59e0b", // amber
  "#ec4899", // pink
  "#14b8a6", // teal
  "#6366f1", // indigo
];

// Helper to extract a concise subsystem category from path
function getSubsystem(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  if (parts.length <= 1) return "root";

  // If starts with repo directory name (e.g. GitThatOffer/...), inspect sub-path
  const p = parts.length > 2 && !parts[0].includes(".") ? parts.slice(1) : parts;

  if (p[0] === "lib" || p[0] === "utils" || p[0] === "helpers") return "lib";
  if (p[0] === "app" && p[1] === "api") {
    return p.length > 3 ? `api/${p[2]}` : "api";
  }
  if (p[0] === "app" && p[1] === "admin") return "admin";
  if (p[0] === "app" && (p[1] === "components" || p[1] === "ui")) return "components";
  if (p[0] === "components" || p[0] === "ui") return "components";
  if (p[0] === "app") return "app-pages";
  if (p[0] === "pages") return "pages";
  if (p[0] === "src") return p[1] || "src";
  return p[0] || "root";
}

export default function GraphCanvas({
  nodes,
  edges,
  highlightedPath,
  onSelectNode,
  onAskAboutFile,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<GraphEdge | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedModule, setSelectedModule] = useState<string>("all");

  const isDark = resolvedTheme === "dark";

  const [copiedMermaid, setCopiedMermaid] = useState(false);
  const [showMinimap, setShowMinimap] = useState(true);
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Copy architecture graph as Mermaid.js syntax
  const handleCopyMermaid = () => {
    const sanitize = (id: string) => id.replace(/[^a-zA-Z0-9_]/g, "_");
    let mermaid = "graph TD\n";

    // Group nodes by subsystem
    modules.forEach((sub) => {
      mermaid += `  subgraph ${sanitize(sub)} ["📦 ${sub}"]\n`;
      nodes
        .filter((n) => (nodeSubsystemMap[n.id] || "root") === sub)
        .forEach((n) => {
          const basename = n.path.split("/").pop() || n.path;
          const star = n.in_degree && n.in_degree >= 3 ? " ⭐" : "";
          mermaid += `    ${sanitize(n.id)}["${basename}${star}"]\n`;
        });
      mermaid += `  end\n`;
    });

    edges.forEach((e) => {
      mermaid += `  ${sanitize(e.source)} --> ${sanitize(e.target)}\n`;
    });

    navigator.clipboard.writeText(mermaid);
    setCopiedMermaid(true);
    setTimeout(() => setCopiedMermaid(false), 2000);
  };

  // Export full-res vector SVG diagram
  const handleExportSVG = () => {
    if (!svgRef.current) return;
    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const blob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "repolens_architecture_diagram.svg";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Auto-focus on search node
  const handleSearchSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;
    const q = searchQuery.toLowerCase();
    const matched = nodes.find(
      (n) => n.path.toLowerCase().includes(q) || n.label.toLowerCase().includes(q)
    );
    if (matched && nodePositions[matched.id]) {
      const pos = nodePositions[matched.id];
      setPan({ x: -pos.x, y: -pos.y });
      setZoom(1.6);
      setSelectedNode(matched);
      setFocusedNodeId(matched.id);
      setTimeout(() => setFocusedNodeId(null), 3000);
    }
  };

  // Subsystem module assignment for each node
  const nodeSubsystemMap = useMemo(() => {
    const map: Record<string, string> = {};
    nodes.forEach((n) => {
      map[n.id] = getSubsystem(n.path);
    });
    return map;
  }, [nodes]);

  // Distinct subsystems / modules
  const modules = useMemo(() => {
    const set = new Set<string>();
    nodes.forEach((n) => set.add(nodeSubsystemMap[n.id] || "root"));
    return Array.from(set).sort();
  }, [nodes, nodeSubsystemMap]);

  // Assign color to each subsystem
  const moduleColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    modules.forEach((mod, idx) => {
      map[mod] = MODULE_COLORS[idx % MODULE_COLORS.length];
    });
    return map;
  }, [modules]);

  // Adjacency maps for instant connected-edge & connected-node lookups
  const adjacency = useMemo(() => {
    const incoming: Record<string, Set<string>> = {};
    const outgoing: Record<string, Set<string>> = {};
    const connectedEdges: Record<string, Set<string>> = {};

    nodes.forEach((n) => {
      incoming[n.id] = new Set();
      outgoing[n.id] = new Set();
      connectedEdges[n.id] = new Set();
    });

    edges.forEach((e) => {
      if (outgoing[e.source]) outgoing[e.source].add(e.target);
      if (incoming[e.target]) incoming[e.target].add(e.source);
      if (connectedEdges[e.source]) connectedEdges[e.source].add(e.id);
      if (connectedEdges[e.target]) connectedEdges[e.target].add(e.id);
    });

    return { incoming, outgoing, connectedEdges };
  }, [nodes, edges]);

  // Calculate Edge Similarity / Interaction Weight
  // - Intra-cluster edges have high similarity (0.8 - 1.0)
  // - Edges pointing to core hubs have medium-high similarity (0.6 - 0.8)
  // - Cross-system standard imports have standard weight (0.3 - 0.5)
  const edgeWeights = useMemo(() => {
    const weights: Record<string, { weight: number; label: string; color: string; isIntra: boolean }> = {};

    edges.forEach((e) => {
      const srcSub = nodeSubsystemMap[e.source] || "root";
      const tgtSub = nodeSubsystemMap[e.target] || "root";
      const isIntra = srcSub === tgtSub;

      let score = isIntra ? 0.85 : 0.45;
      if (e.type === "references") score += 0.15;

      let color = isDark ? "rgba(100, 116, 139, 0.4)" : "rgba(148, 163, 184, 0.5)";
      let label = "Cross-Module Import";

      if (score >= 0.8) {
        color = isDark ? "#10b981" : "#059669"; // Emerald/Dark green for tight intra-cluster
        label = "High Cohesion (Intra-Module)";
      } else if (score >= 0.6) {
        color = isDark ? "#06b6d4" : "#0284c7"; // Cyan for hub dependencies
        label = "Core Subsystem Dependency";
      } else {
        color = isDark ? "rgba(148, 163, 184, 0.35)" : "rgba(100, 116, 139, 0.45)";
      }

      weights[e.id] = {
        weight: score,
        label,
        color,
        isIntra,
      };
    });

    return weights;
  }, [edges, nodeSubsystemMap, isDark]);

  // Compact Cluster Island Layout
  const { nodePositions, clusterHulls } = useMemo(() => {
    const positions: Record<string, { x: number; y: number; r: number; subsystem: string }> = {};
    const hulls: Record<string, { x: number; y: number; width: number; height: number; count: number }> = {};
    if (nodes.length === 0) return { nodePositions: positions, clusterHulls: hulls };

    // Group nodes by subsystem
    const groups: Record<string, GraphNode[]> = {};
    nodes.forEach((n) => {
      const sub = nodeSubsystemMap[n.id] || "root";
      if (!groups[sub]) groups[sub] = [];
      groups[sub].push(n);
    });

    const groupKeys = Object.keys(groups);
    const numGroups = groupKeys.length;

    // Compact 2D Grid positioning for cluster centers
    const cols = Math.max(1, Math.ceil(Math.sqrt(numGroups * 1.25)));
    const cellWidth = 320;
    const cellHeight = 270;
    const totalWidth = cols * cellWidth;
    const totalRows = Math.ceil(numGroups / cols);
    const totalHeight = totalRows * cellHeight;

    groupKeys.forEach((sub, gIdx) => {
      const row = Math.floor(gIdx / cols);
      const col = gIdx % cols;
      const gCenterX = col * cellWidth - totalWidth / 2 + cellWidth / 2;
      const gCenterY = row * cellHeight - totalHeight / 2 + cellHeight / 2;

      const groupNodes = groups[sub];
      // Sort nodes in cluster so high in-degree hub nodes sit in the center
      const sortedNodes = [...groupNodes].sort((a, b) => (b.in_degree || 0) - (a.in_degree || 0));
      const count = sortedNodes.length;

      let minX = gCenterX;
      let maxX = gCenterX;
      let minY = gCenterY;
      let maxY = gCenterY;

      sortedNodes.forEach((node, nIdx) => {
        let x = gCenterX;
        let y = gCenterY;

        if (nIdx > 0) {
          // Golden-angle spiral packing for compact, clean cluster islands
          const radius = Math.min(105, 28 + (nIdx - 1) * 15);
          const angle = nIdx * 2.39996; // ~137.5 degrees
          x += Math.cos(angle) * radius;
          y += Math.sin(angle) * radius;
        }

        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);

        const nodeSize = Math.max(8, Math.min(20, 8 + Math.sqrt(node.loc || 10) * 0.8 + (node.in_degree || 0) * 1.2));
        positions[node.id] = { x, y, r: nodeSize, subsystem: sub };
      });

      const padding = 45;
      hulls[sub] = {
        x: minX - padding,
        y: minY - padding,
        width: Math.max(160, maxX - minX + padding * 2),
        height: Math.max(130, maxY - minY + padding * 2),
        count,
      };
    });

    return { nodePositions: positions, clusterHulls: hulls };
  }, [nodes, nodeSubsystemMap]);

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  // Zoom handlers
  const handleZoomIn = () => setZoom((z) => Math.min(3.5, z * 1.2));
  const handleZoomOut = () => setZoom((z) => Math.max(0.2, z / 1.2));
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    setZoom((z) => Math.min(3.5, Math.max(0.2, z * zoomFactor)));
  };
  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Center on highlighted citation node if provided
  useEffect(() => {
    if (!highlightedPath) return;
    const matchedNode = nodes.find(
      (n) =>
        n.path === highlightedPath ||
        n.path.endsWith(highlightedPath) ||
        highlightedPath.endsWith(n.path)
    );
    if (matchedNode && nodePositions[matchedNode.id]) {
      const pos = nodePositions[matchedNode.id];
      setPan({ x: -pos.x, y: -pos.y });
      setZoom(1.35);
    }
  }, [highlightedPath, nodes, nodePositions]);

  // Active interaction focus state
  const activeNode = hoveredNode || selectedNode;

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex flex-col overflow-hidden bg-background select-none transition-colors"
    >
      {/* Floating Header Toolbar */}
      <div className="absolute top-4 left-4 right-4 z-20 flex items-center justify-between pointer-events-none">
        {/* Search & Module filter */}
        <div className="flex items-center gap-2 pointer-events-auto bg-surface/90 backdrop-blur-md p-1.5 rounded-xl border border-surface-border shadow-card">
          <form onSubmit={handleSearchSubmit} className="relative flex items-center">
            <Search className="h-3.5 w-3.5 absolute left-3 text-text-subtle pointer-events-none" />
            <input
              type="text"
              placeholder="Search node & press Enter..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-surface-raised border border-surface-border text-foreground placeholder-text-subtle text-xs rounded-lg pl-8 pr-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary-500 w-52 font-mono"
            />
          </form>

          <select
            value={selectedModule}
            onChange={(e) => setSelectedModule(e.target.value)}
            className="bg-surface-raised border border-surface-border text-foreground text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary-500 font-mono"
          >
            <option value="all">All Subsystems ({modules.length})</option>
            {modules.map((mod) => (
              <option key={mod} value={mod}>
                {mod} ({clusterHulls[mod]?.count || 0})
              </option>
            ))}
          </select>
        </div>

        {/* Action Tools & View Controls */}
        <div className="flex items-center gap-2.5 pointer-events-auto">
          {/* Mermaid & SVG Export Buttons */}
          <div className="flex items-center gap-1 bg-surface/90 backdrop-blur-md p-1.5 rounded-xl border border-surface-border shadow-card">
            <button
              onClick={handleCopyMermaid}
              title="Copy Architecture Diagram in Mermaid.js syntax"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              {copiedMermaid ? (
                <>
                  <Check className="h-3.5 w-3.5 text-accent-emerald" />
                  <span className="text-accent-emerald font-semibold">Copied!</span>
                </>
              ) : (
                <>
                  <Code2 className="h-3.5 w-3.5 text-primary-500" />
                  <span className="hidden sm:inline">Mermaid</span>
                </>
              )}
            </button>

            <button
              onClick={handleExportSVG}
              title="Download vector SVG architecture diagram"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              <FileDown className="h-3.5 w-3.5 text-accent-cyan" />
              <span className="hidden sm:inline">Export SVG</span>
            </button>
          </div>

          {/* View Zoom Controls & Minimap Toggle */}
          <div className="flex items-center gap-1 bg-surface/90 backdrop-blur-md p-1.5 rounded-xl border border-surface-border shadow-card">
            <button
              onClick={() => setShowMinimap(!showMinimap)}
              title="Toggle Minimap HUD"
              className={`p-1.5 rounded-lg transition-colors ${
                showMinimap ? "text-primary-500 bg-primary-500/10" : "text-text-muted hover:text-foreground hover:bg-surface-raised"
              }`}
            >
              <Map className="h-4 w-4" />
            </button>
            <button
              onClick={handleZoomIn}
              title="Zoom In"
              className="p-1.5 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <button
              onClick={handleZoomOut}
              title="Zoom Out"
              className="p-1.5 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              onClick={resetView}
              title="Reset View"
              className="p-1.5 rounded-lg text-text-muted hover:text-foreground hover:bg-surface-raised transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* SVG Canvas */}
      <div
        className="flex-1 w-full h-full cursor-grab active:cursor-grabbing relative"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <svg
          ref={svgRef}
          className="w-full h-full"
          style={{
            backgroundColor: isDark ? "#090b10" : "#f8fafc",
          }}
        >
          {/* Subtle Grid Background & Markers */}
          <defs>
            <pattern id="canvas-grid" width="36" height="36" patternUnits="userSpaceOnUse">
              <circle
                cx="2"
                cy="2"
                r="1"
                fill={isDark ? "rgba(255, 255, 255, 0.07)" : "rgba(0, 0, 0, 0.07)"}
              />
            </pattern>

            {/* Standard arrow marker */}
            <marker
              id="graph-arrow-std"
              viewBox="0 0 10 10"
              refX="17"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path
                d="M 0 1.5 L 8 5 L 0 8.5 z"
                fill={isDark ? "rgba(148, 163, 184, 0.6)" : "rgba(100, 116, 139, 0.6)"}
              />
            </marker>

            {/* High-cohesion emerald arrow marker */}
            <marker
              id="graph-arrow-emerald"
              viewBox="0 0 10 10"
              refX="17"
              refY="5"
              markerWidth="6.5"
              markerHeight="6.5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10b981" />
            </marker>

            {/* Incoming dependency arrow marker (emerald) */}
            <marker
              id="graph-arrow-in"
              viewBox="0 0 10 10"
              refX="17"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#10b981" />
            </marker>

            {/* Outgoing dependency arrow marker (sky blue) */}
            <marker
              id="graph-arrow-out"
              viewBox="0 0 10 10"
              refX="17"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#38bdf8" />
            </marker>
          </defs>

          <rect width="100%" height="100%" fill="url(#canvas-grid)" />

          <g transform={`translate(${pan.x + 400}, ${pan.y + 300}) scale(${zoom})`}>
            {/* Render Cluster Hulls / Boundary Islands */}
            {Object.entries(clusterHulls).map(([sub, hull]) => {
              if (selectedModule !== "all" && sub !== selectedModule) return null;
              const color = moduleColorMap[sub] || "#3b82f6";

              return (
                <g key={`hull-${sub}`}>
                  <rect
                    x={hull.x}
                    y={hull.y}
                    width={hull.width}
                    height={hull.height}
                    rx={20}
                    ry={20}
                    fill={isDark ? "rgba(15, 23, 42, 0.45)" : "rgba(241, 245, 249, 0.65)"}
                    stroke={isDark ? "rgba(51, 65, 85, 0.5)" : "rgba(203, 213, 225, 0.8)"}
                    strokeWidth={1}
                    strokeDasharray="4 4"
                    className="transition-all"
                  />
                  {/* Cluster Title Tag */}
                  <g transform={`translate(${hull.x + 14}, ${hull.y + 18})`}>
                    <rect
                      x={-4}
                      y={-10}
                      width={sub.length * 7.5 + 40}
                      height={18}
                      rx={6}
                      fill={isDark ? "rgba(30, 41, 59, 0.85)" : "rgba(255, 255, 255, 0.9)"}
                      stroke={color}
                      strokeWidth={1}
                    />
                    <circle cx={4} cy={-1} r={3} fill={color} />
                    <text
                      x={12}
                      y={2}
                      fill={isDark ? "#f1f5f9" : "#1e293b"}
                      fontSize="9.5px"
                      fontFamily="'JetBrains Mono', monospace"
                      fontWeight="600"
                    >
                      {sub} ({hull.count})
                    </text>
                  </g>
                </g>
              );
            })}

            {/* Render Edges (with interaction weights & dynamic colors) */}
            {edges.map((e) => {
              const src = nodePositions[e.source];
              const tgt = nodePositions[e.target];
              if (!src || !tgt) return null;

              if (
                selectedModule !== "all" &&
                src.subsystem !== selectedModule &&
                tgt.subsystem !== selectedModule
              ) {
                return null;
              }

              const edgeMeta = edgeWeights[e.id] || { weight: 0.5, label: "Import", color: "#64748b", isIntra: false };

              // Interaction state
              const isSourceActive = activeNode && activeNode.id === e.source;
              const isTargetActive = activeNode && activeNode.id === e.target;
              const isHighlighted = isSourceActive || isTargetActive;
              const isDimmed = activeNode && !isHighlighted;
              const isHoveredEdge = hoveredEdge && hoveredEdge.id === e.id;

              let strokeColor = edgeMeta.color;
              let strokeWidth = edgeMeta.weight >= 0.8 ? 2.2 : 1.4;
              let marker = edgeMeta.weight >= 0.8 ? "url(#graph-arrow-emerald)" : "url(#graph-arrow-std)";
              let strokeOpacity = isDimmed ? 0.08 : edgeMeta.isIntra ? 0.75 : 0.45;

              if (isHighlighted || isHoveredEdge) {
                strokeOpacity = 1;
                if (isSourceActive) {
                  // Outgoing dependency (source -> target)
                  strokeColor = "#38bdf8"; // sky blue
                  strokeWidth = 2.8;
                  marker = "url(#graph-arrow-out)";
                } else {
                  // Incoming dependency (pointing to active target)
                  strokeColor = "#10b981"; // emerald green
                  strokeWidth = 3;
                  marker = "url(#graph-arrow-in)";
                }
              }

              return (
                <g key={e.id}>
                  {/* Invisible hit-area for hover */}
                  <line
                    x1={src.x}
                    y1={src.y}
                    x2={tgt.x}
                    y2={tgt.y}
                    stroke="transparent"
                    strokeWidth={12}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredEdge(e)}
                    onMouseLeave={() => setHoveredEdge(null)}
                  />
                  {/* Visual Edge */}
                  <line
                    x1={src.x}
                    y1={src.y}
                    x2={tgt.x}
                    y2={tgt.y}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeOpacity={strokeOpacity}
                    markerEnd={marker}
                    strokeDasharray={e.type === "references" ? "4 4" : "none"}
                    className="transition-all duration-150"
                  />
                </g>
              );
            })}

            {/* Render Nodes */}
            {nodes.map((node) => {
              const pos = nodePositions[node.id];
              if (!pos) return null;

              const isMatched =
                searchQuery &&
                (node.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
                  node.label.toLowerCase().includes(searchQuery.toLowerCase()));

              const isCitationActive =
                highlightedPath &&
                (node.path === highlightedPath ||
                  node.path.endsWith(highlightedPath) ||
                  highlightedPath.endsWith(node.path));

              const isDirectlyActive = activeNode?.id === node.id;
              const isConnectedNeighbor =
                activeNode &&
                (adjacency.incoming[activeNode.id]?.has(node.id) ||
                  adjacency.outgoing[activeNode.id]?.has(node.id));

              const isDimmed = activeNode && !isDirectlyActive && !isConnectedNeighbor;
              const mod = pos.subsystem;
              const color = moduleColorMap[mod] || "#3b82f6";

              if (selectedModule !== "all" && mod !== selectedModule) {
                return null;
              }

              const isHub = (node.in_degree || 0) >= 3;

              return (
                <g
                  key={node.id}
                  transform={`translate(${pos.x}, ${pos.y})`}
                  className="cursor-pointer transition-transform duration-150"
                  opacity={isDimmed ? 0.3 : 1}
                  onMouseEnter={() => setHoveredNode(node)}
                  onMouseLeave={() => setHoveredNode(null)}
                  onClick={() => {
                    setSelectedNode(node);
                    if (onSelectNode) onSelectNode(node);
                  }}
                >
                  {/* Subtle focus / citation glow */}
                  {(isCitationActive || isMatched || isDirectlyActive) && (
                    <circle
                      r={pos.r + 7}
                      fill="none"
                      stroke={isCitationActive ? "#06b6d4" : isDirectlyActive ? "#38bdf8" : color}
                      strokeWidth={isCitationActive ? 3 : 2}
                      className={isCitationActive ? "animate-ping opacity-60" : "opacity-90"}
                    />
                  )}

                  {/* Core Hub Badge indicator */}
                  {isHub && (
                    <circle
                      r={pos.r + 4}
                      fill="none"
                      stroke="#10b981"
                      strokeWidth={1.2}
                      strokeDasharray="2 2"
                      className="opacity-70"
                    />
                  )}

                  {/* Main Node Circle */}
                  <circle
                    r={pos.r}
                    fill={color}
                    fillOpacity={isDirectlyActive ? 1 : isHub ? 0.95 : 0.85}
                    stroke={isDark ? "#090b10" : "#ffffff"}
                    strokeWidth={isHub ? 2.5 : 1.5}
                  />

                  {/* Central Node Label */}
                  <text
                    y={pos.r + 13}
                    textAnchor="middle"
                    fill={
                      isCitationActive
                        ? "#0284c7"
                        : isDirectlyActive
                        ? isDark
                          ? "#f8fafc"
                          : "#0f172a"
                        : isDark
                        ? "#94a3b8"
                        : "#64748b"
                    }
                    fontSize={zoom > 0.8 ? "10px" : "8.5px"}
                    fontFamily="'JetBrains Mono', monospace"
                    fontWeight={isCitationActive || isDirectlyActive || isHub ? "600" : "400"}
                    className="pointer-events-none select-none"
                  >
                    {node.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Floating Edge Tooltip */}
      {hoveredEdge && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-30 pointer-events-none px-3.5 py-1.5 rounded-xl bg-surface/95 border border-primary-500/40 text-xs shadow-elevation backdrop-blur-md flex items-center gap-2">
          <span className="font-mono text-foreground font-medium">
            {nodes.find((n) => n.id === hoveredEdge.source)?.label || hoveredEdge.source}
          </span>
          <span className="text-emerald-500 font-semibold">──imports──►</span>
          <span className="font-mono text-foreground font-medium">
            {nodes.find((n) => n.id === hoveredEdge.target)?.label || hoveredEdge.target}
          </span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-raised border border-surface-border text-text-muted">
            {edgeWeights[hoveredEdge.id]?.label || "Dependency"}
          </span>
        </div>
      )}

      {/* Selected Node Interactive Inspector Card */}
      {selectedNode ? (
        <div className="absolute bottom-4 left-4 z-30 surface-card p-4 rounded-xl shadow-elevation max-w-sm border border-primary-500/40 bg-surface/95 backdrop-blur-md animate-in fade-in slide-in-from-bottom-2 duration-150">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="flex items-center gap-2 min-w-0">
              <FileCode className="h-4 w-4 text-primary-500 flex-shrink-0" />
              <span className="font-bold text-xs text-foreground truncate">{selectedNode.label}</span>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-text-subtle hover:text-foreground text-xs p-1 rounded hover:bg-surface-raised transition-colors"
            >
              ✕
            </button>
          </div>

          <p className="text-[11px] font-mono text-text-muted truncate mb-3">{selectedNode.path}</p>

          <div className="grid grid-cols-4 gap-2 text-[10px] text-text-muted border-t border-surface-border pt-2.5 mb-3.5">
            <div>
              <span className="text-text-subtle block">LOC</span>
              <span className="font-mono font-medium text-foreground">{selectedNode.loc}</span>
            </div>
            <div>
              <span className="text-text-subtle block">Depended By</span>
              <span className="font-mono font-medium text-emerald-500 font-semibold">{selectedNode.in_degree} files</span>
            </div>
            <div>
              <span className="text-text-subtle block">Imports</span>
              <span className="font-mono font-medium text-sky-400">{selectedNode.out_degree} files</span>
            </div>
            <div>
              <span className="text-text-subtle block">Subsystem</span>
              <span className="font-mono font-medium text-accent-cyan truncate block">
                {nodeSubsystemMap[selectedNode.id] || "root"}
              </span>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => onSelectNode && onSelectNode(selectedNode)}
              className="flex-1 flex items-center justify-center gap-1.5 bg-surface-raised hover:bg-surface-border text-foreground text-xs font-medium py-1.5 px-2.5 rounded-lg border border-surface-border transition-all"
            >
              <FileCode className="h-3.5 w-3.5 text-primary-500" />
              <span>Inspect Code</span>
            </button>

            {onAskAboutFile && (
              <button
                onClick={() => {
                  onAskAboutFile(selectedNode.path);
                  setSelectedNode(null);
                }}
                className="flex-1 flex items-center justify-center gap-1.5 bg-primary-600 hover:bg-primary-500 text-white text-xs font-semibold py-1.5 px-2.5 rounded-lg shadow-sm transition-all hover:scale-[1.01]"
              >
                <Compass className="h-3.5 w-3.5" />
                <span>Ask AI</span>
              </button>
            )}
          </div>
        </div>
      ) : hoveredNode ? (
        /* Hover Tooltip Card */
        <div className="absolute bottom-4 left-4 z-30 pointer-events-none surface-card p-3.5 rounded-xl shadow-elevation max-w-xs animate-in fade-in duration-150">
          <div className="flex items-center gap-2 mb-1.5">
            <FileCode className="h-4 w-4 text-primary-500" />
            <span className="font-semibold text-xs text-foreground truncate">{hoveredNode.label}</span>
          </div>
          <p className="text-[11px] font-mono text-text-muted truncate mb-2">{hoveredNode.path}</p>
          <div className="grid grid-cols-3 gap-2 text-[10px] text-text-muted border-t border-surface-border pt-2">
            <div>
              <span className="text-text-subtle block">LOC</span>
              <span className="font-mono font-medium text-foreground">{hoveredNode.loc}</span>
            </div>
            <div>
              <span className="text-text-subtle block">Depended By</span>
              <span className="font-mono font-medium text-emerald-500">{hoveredNode.in_degree}</span>
            </div>
            <div>
              <span className="text-text-subtle block">Subsystem</span>
              <span className="font-mono font-medium text-accent-cyan truncate block">
                {nodeSubsystemMap[hoveredNode.id] || "root"}
              </span>
            </div>
          </div>
        </div>
      ) : null}

      {/* Minimap HUD Overlay */}
      {showMinimap && (
        <div className="absolute bottom-12 right-4 z-20 w-44 h-28 rounded-xl bg-surface/90 border border-surface-border backdrop-blur-md shadow-card p-2 overflow-hidden flex flex-col justify-between animate-in fade-in duration-200">
          <div className="flex items-center justify-between text-[10px] font-mono text-text-subtle font-semibold">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              MINIMAP
            </span>
            <span className="text-text-muted">{Math.round(zoom * 100)}%</span>
          </div>

          <div
            className="relative flex-1 w-full h-full my-1 rounded-lg bg-background/60 overflow-hidden border border-surface-border/40 cursor-crosshair"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const clickX = e.clientX - rect.left;
              const clickY = e.clientY - rect.top;
              const normalizedX = ((clickX / rect.width) - 0.5) * 800;
              const normalizedY = ((clickY / rect.height) - 0.5) * 600;
              setPan({ x: -normalizedX, y: -normalizedY });
            }}
          >
            <svg className="w-full h-full" viewBox="-450 -350 900 700">
              {/* Cluster hulls thumbnail */}
              {Object.entries(clusterHulls).map(([mod, hull]) => (
                <rect
                  key={mod}
                  x={hull.x}
                  y={hull.y}
                  width={hull.width}
                  height={hull.height}
                  rx={10}
                  fill="rgba(59, 130, 246, 0.05)"
                  stroke="rgba(148, 163, 184, 0.3)"
                  strokeWidth={1.5}
                />
              ))}

              {/* Node dots */}
              {nodes.map((n) => {
                const p = nodePositions[n.id];
                if (!p) return null;
                const isHub = n.in_degree && n.in_degree > 3;
                return (
                  <circle
                    key={n.id}
                    cx={p.x}
                    cy={p.y}
                    r={isHub ? 8 : 4}
                    fill={isHub ? "#10b981" : "#64748b"}
                    opacity={0.85}
                  />
                );
              })}

              {/* Current Viewport Window */}
              <rect
                x={-pan.x - 300 / zoom}
                y={-pan.y - 200 / zoom}
                width={600 / zoom}
                height={400 / zoom}
                fill="rgba(59, 130, 246, 0.12)"
                stroke="#3b82f6"
                strokeWidth={2}
                rx={4}
              />
            </svg>
          </div>
        </div>
      )}

      {/* Graph Stats Footer */}
      <div className="absolute bottom-4 right-4 z-20 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface/90 border border-surface-border text-[11px] text-text-muted pointer-events-none backdrop-blur-md shadow-sm font-mono">
        <span className="font-medium text-foreground">{nodes.length}</span> files
        <span>•</span>
        <span className="font-medium text-foreground">{edges.length}</span> dependencies
        <span>•</span>
        <span className="font-medium text-foreground">{modules.length}</span> subsystems
      </div>
    </div>
  );
}
