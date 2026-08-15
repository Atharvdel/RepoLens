/**
 * Typed API client for RepoLens FastAPI REST backend (SDD §12).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000/api";

export interface RepositoryItem {
  id: number;
  url_or_path: string;
  name: string;
  default_branch?: string;
  status: "indexing" | "ready" | "failed";
  indexed_at?: string;
  github_owner?: string;
  github_repo?: string;
  file_count: number;
  symbol_count: number;
  total_loc: number;
}

export interface KeyFile {
  path: string;
  in_degree: number;
}

export interface RepositoryDetail extends RepositoryItem {
  language_breakdown: Record<string, number>;
  key_files: KeyFile[];
  docs_count: number;
  commits_count: number;
  issues_count: number;
  readme_preview?: string;
}

export interface GraphNode {
  id: string;
  file_id: number;
  path: string;
  label: string;
  language: string;
  loc: number;
  in_degree: number;
  out_degree: number;
  module: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  scope: string;
  target?: string;
}

export interface ArchitectureResponse {
  modules: string[];
  key_files: KeyFile[];
  edges: Array<{ source: string; target: string; weight: number }>;
}

export interface ChatResponse {
  session_id: number;
  message_id: number;
  answer: string;
  citations: string[];
  node_trace: string[];
  tool_trace: Record<string, any>;
  replans_used: number;
  created_at: string;
}

export interface ChatMessageItem {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  tool_trace?: Record<string, any>;
}

export interface ChatSessionItem {
  id: number;
  repository_id: number;
  created_at: string;
  message_count: number;
  last_message?: string;
}

export interface SymbolItem {
  id: number;
  name: string;
  kind: string;
  line_start: number;
  line_end: number;
  docstring?: string;
  parent_symbol_id?: number;
}

export interface FileDetailResponse {
  id: number;
  repository_id: number;
  path: string;
  language: string;
  loc: number;
  last_modified?: string;
  symbols: SymbolItem[];
  imports: string[];
  referenced_by: string[];
  top_contributors: Array<{ author: string; commits: number }>;
  recent_commits: Array<{ hash: string; message: string; date: string; author: string }>;
  content?: string;
}

export interface DocumentItem {
  id: number;
  path: string;
  title?: string;
  content: string;
}

export interface SearchResponse {
  query: string;
  search_type: string;
  total_hits: number;
  hits: any[];
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) errorDetail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(errorDetail);
  }

  if (res.status === 204) {
    return {} as T;
  }

  return res.json();
}

export const api = {
  // Repositories
  listRepositories: () => request<RepositoryItem[]>("/repositories"),
  getRepository: (id: number) => request<RepositoryDetail>(`/repositories/${id}`),
  createRepository: (data: { url_or_path: string; name?: string; github_token?: string }) =>
    request<RepositoryItem>("/repositories", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  reindexRepository: (id: number) =>
    request<RepositoryItem>(`/repositories/${id}/reindex`, {
      method: "POST",
    }),
  deleteRepository: (id: number) =>
    request<void>(`/repositories/${id}`, {
      method: "DELETE",
    }),

  // Graph & Architecture
  getGraph: (id: number, params?: { scope?: string; target?: string; depth?: number }) => {
    const query = new URLSearchParams();
    if (params?.scope) query.set("scope", params.scope);
    if (params?.target) query.set("target", params.target);
    if (params?.depth) query.set("depth", params.depth.toString());
    return request<GraphResponse>(`/repositories/${id}/graph?${query.toString()}`);
  },
  getArchitecture: (id: number, params?: { target?: string; top_k?: number; radius?: number }) => {
    const query = new URLSearchParams();
    if (params?.target) query.set("target", params.target);
    if (params?.top_k) query.set("top_k", params.top_k.toString());
    if (params?.radius) query.set("radius", params.radius.toString());
    return request<ArchitectureResponse>(`/repositories/${id}/architecture?${query.toString()}`);
  },

  // Chat
  askChat: (id: number, data: { message: string; session_id?: number; model?: string }) =>
    request<ChatResponse>(`/repositories/${id}/chat`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
  listChatSessions: (id: number) => request<ChatSessionItem[]>(`/repositories/${id}/chat/sessions`),
  getChatSessionMessages: (id: number, sessionId: number) =>
    request<ChatMessageItem[]>(`/repositories/${id}/chat/sessions/${sessionId}`),

  // Files & Docs
  getFileDetail: (id: number, path: string, includeContent: boolean = true) =>
    request<FileDetailResponse>(`/repositories/${id}/files/${path}?include_content=${includeContent}`),
  listDocuments: (id: number) => request<DocumentItem[]>(`/repositories/${id}/documents`),
  getIssues: (id: number, target?: string) => {
    const q = target ? `?target=${encodeURIComponent(target)}` : "";
    return request<{ issues: any[]; prs: any[] }>(`/repositories/${id}/issues${q}`);
  },

  // Search
  search: (id: number, params: { q: string; type?: string; kind?: string; regex?: boolean }) => {
    const query = new URLSearchParams();
    query.set("q", params.q);
    if (params.type) query.set("type", params.type);
    if (params.kind) query.set("kind", params.kind);
    if (params.regex !== undefined) query.set("regex", String(params.regex));
    return request<SearchResponse>(`/repositories/${id}/search?${query.toString()}`);
  },

  // Executive Architecture Report
  getRepositoryReport: (id: number) =>
    request<{ repository_id: string; name: string; filename: string; markdown: string }>(
      `/repositories/${id}/report`
    ),

  // Architectural Health & Tech Debt
  getRepositoryHealth: (id: number) => request<ArchitectureHealthResponse>(`/repositories/${id}/health`),
};

export interface HealthMetric {
  name: string;
  value: string;
  status: "healthy" | "warning" | "alert";
  description: string;
}

export interface ArchitectureHealthResponse {
  repository_id: number;
  health_score: number;
  grade: string;
  circular_dependencies: string[][];
  hub_risks: Array<{
    path: string;
    in_degree: number;
    risk_level: "High" | "Medium" | "Low";
    assessment: string;
  }>;
  orphaned_files: string[];
  modularity_ratio: number;
  metrics: Array<{
    name: string;
    value: string;
    status: "healthy" | "warning" | "alert";
    description: string;
  }>;
  recommendations: string[];
}
