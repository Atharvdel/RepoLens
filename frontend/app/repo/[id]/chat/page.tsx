"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bot,
  Check,
  ChevronRight,
  Compass,
  Copy,
  CornerDownLeft,
  FileCode,
  GitGraph,
  HelpCircle,
  History,
  Layers,
  Loader2,
  Maximize2,
  MessageSquare,
  MessageSquareCode,
  Plus,
  RotateCcw,
  Send,
  Sparkles,
  User,
} from "lucide-react";
import CodeViewerModal from "@/components/CodeViewerModal";
import GraphCanvas from "@/components/GraphCanvas";
import Navbar from "@/components/Navbar";
import ToolTraceViewer from "@/components/ToolTraceViewer";
import {
  api,
  ChatMessageItem,
  ChatSessionItem,
  GraphEdge,
  GraphNode,
  RepositoryDetail,
} from "@/lib/api";

export default function RepoChatAndGraphPage() {
  const params = useParams();
  const repoId = Number(params.id);

  const [repo, setRepo] = useState<RepositoryDetail | null>(null);
  const [graphNodes, setGraphNodes] = useState<GraphNode[]>([]);
  const [graphEdges, setGraphEdges] = useState<GraphEdge[]>([]);

  // Chat state
  const [sessions, setSessions] = useState<ChatSessionItem[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<number | null>(null);

  // Graph synchronization state
  const [highlightedPath, setHighlightedPath] = useState<string | null>(null);
  const [selectedFileForModal, setSelectedFileForModal] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initial load
  useEffect(() => {
    api.getRepository(repoId).then((r) => setRepo(r)).catch(console.error);

    api.getGraph(repoId, { scope: "whole" }).then((g) => {
      setGraphNodes(g.nodes);
      setGraphEdges(g.edges);
    }).catch(console.error);

    api.listChatSessions(repoId).then((sessList) => {
      setSessions(sessList);
      if (sessList.length > 0) {
        setCurrentSessionId(sessList[0].id);
        api.getChatSessionMessages(repoId, sessList[0].id).then(setMessages);
      }
    }).catch(console.error);
  }, [repoId]);

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  const handleStartNewChat = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setInputMessage("");
  };

  const handleCopyMessage = (msgId: number, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedMessageId(msgId);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const handleSendMessage = async (customPrompt?: string) => {
    const textToSend = customPrompt || inputMessage.trim();
    if (!textToSend || isSending) return;

    setInputMessage("");
    setIsSending(true);

    const tempUserMsg: ChatMessageItem = {
      id: Date.now(),
      role: "user",
      content: textToSend,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    try {
      const resp = await api.askChat(repoId, {
        message: textToSend,
        session_id: currentSessionId || undefined,
      });

      setCurrentSessionId(resp.session_id);

      const assistantMsg: ChatMessageItem = {
        id: resp.message_id,
        role: "assistant",
        content: resp.answer,
        created_at: resp.created_at,
        tool_trace: resp.tool_trace,
      };
      setMessages((prev) => [...prev, assistantMsg]);

      if (resp.citations && resp.citations.length > 0) {
        setHighlightedPath(resp.citations[0]);
      }

      api.listChatSessions(repoId).then(setSessions);
    } catch (err: any) {
      const errMsg: ChatMessageItem = {
        id: Date.now() + 1,
        role: "assistant",
        content: `Error: ${err.message || "Failed to analyze question"}`,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsSending(false);
    }
  };

  // Custom renderer for markdown links & citations
  const renderMessageContent = (content: string, citations?: string[]) => {
    return (
      <div className="markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, className, children, ...props }) {
              const str = String(children);
              const cleanStr = str.split(":")[0];
              const isPath = graphNodes.some(
                (n) =>
                  n.path === cleanStr ||
                  n.path.endsWith(cleanStr) ||
                  cleanStr.endsWith(n.path)
              );
              if (isPath) {
                return (
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      setHighlightedPath(cleanStr);
                      setSelectedFileForModal(str);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        setHighlightedPath(cleanStr);
                        setSelectedFileForModal(str);
                      }
                    }}
                    className="citation-pill inline-flex items-center gap-1 mx-0.5 cursor-pointer select-text"
                    title={`Click to inspect ${str}`}
                  >
                    <FileCode className="h-3 w-3 inline-block flex-shrink-0" />
                    <span className="font-mono">{str}</span>
                  </span>
                );
              }
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>

        {/* Citations Pill Bar */}
        {citations && citations.length > 0 && (
          <div className="mt-3.5 pt-3 border-t border-surface-border">
            <span className="text-[10px] uppercase font-mono text-text-subtle block mb-1.5 font-semibold">
              Grounding Citations:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {citations.map((cite, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setHighlightedPath(cite);
                    setSelectedFileForModal(cite);
                  }}
                  className="citation-pill"
                  title="Click to inspect file code and focus on dependency graph"
                >
                  <FileCode className="h-3 w-3 text-primary-500" />
                  <span>{cite}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const sampleCategories = [
    {
      category: "🏛️ Architecture & System Design",
      questions: [
        "Explain the high-level architecture and key modules of this codebase.",
        "Explain the entire data flow from UI to backend API and database step by step.",
      ],
    },
    {
      category: "🔐 Core Logic & Features",
      questions: [
        "How is authentication, session handling, and authorization implemented?",
        "Where are the main API endpoints defined and what do they handle?",
      ],
    },
    {
      category: "📊 Dependency & Team Insights",
      questions: [
        "What are the central key files with the highest in-degree dependencies?",
        "Who are the top contributors and what are the recent git commits?",
      ],
    },
  ];

  // Dynamic agent step simulation during synthesis
  const [agentStepIndex, setAgentStepIndex] = useState(0);
  const agentSteps = [
    "🧠 Planner: Formulating query strategy and selecting tools...",
    "🔍 Search Agent: Scanning AST symbols & cross-language references...",
    "🏛️ Context Agent: Inspecting dependency graph & degree centrality...",
    "✍️ Synthesizer: Grounding facts & generating inline citations...",
  ];

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isSending) {
      setAgentStepIndex(0);
      interval = setInterval(() => {
        setAgentStepIndex((prev) => (prev < agentSteps.length - 1 ? prev + 1 : prev));
      }, 1800);
    }
    return () => clearInterval(interval);
  }, [isSending]);

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden transition-colors">
      <Navbar currentRepo={repo} />

      {/* Main Split Workspace */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left Column: Chat Console */}
        <div className="w-full lg:w-1/2 flex flex-col border-b lg:border-b-0 lg:border-r border-surface-border bg-surface h-[50vh] lg:h-full">
          {/* Chat Header */}
          <div className="flex items-center justify-between px-5 py-3 border-b border-surface-border bg-surface-raised/40">
            <div className="flex items-center gap-2.5">
              <div className="p-1.5 rounded-lg bg-primary-500/10 text-primary-500 border border-primary-500/20">
                <Bot className="h-4 w-4" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xs font-semibold text-foreground">Codebase Intelligence Assistant</h2>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 bg-primary-500/10 text-primary-500 rounded border border-primary-500/20">
                    Local Agent
                  </span>
                </div>
                <p className="text-[10px] text-text-muted">Multi-agent AST parsing, dependency graph & git history</p>
              </div>
            </div>

            <button
              onClick={handleStartNewChat}
              className="flex items-center gap-1 text-xs text-text-muted hover:text-foreground font-medium px-2.5 py-1 rounded-lg bg-surface-raised border border-surface-border hover:border-text-subtle transition-colors shadow-subtle"
            >
              <Plus className="h-3.5 w-3.5" />
              <span>New Thread</span>
            </button>
          </div>

          {/* Messages Stream */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center p-6 text-center text-text-muted max-w-lg mx-auto">
                <div className="h-11 w-11 rounded-2xl bg-primary-500/10 text-primary-500 flex items-center justify-center border border-primary-500/20 mb-3 shadow-subtle">
                  <Sparkles className="h-5 w-5" />
                </div>
                <h3 className="font-bold text-sm text-foreground mb-1">
                  Ask RepoLens about this Codebase
                </h3>
                <p className="text-xs text-text-muted mb-5 leading-relaxed">
                  RepoLens combines AST symbol parsing, import graph traversal, and git history to provide 100% grounded explanations with verified file citations.
                </p>

                {/* Categorized Quick Question Chips */}
                <div className="w-full space-y-3 text-left">
                  {sampleCategories.map((cat, catIdx) => (
                    <div key={catIdx}>
                      <span className="text-[10px] font-semibold text-text-subtle uppercase tracking-wider block mb-1.5">
                        {cat.category}
                      </span>
                      <div className="space-y-1.5">
                        {cat.questions.map((q, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSendMessage(q)}
                            className="w-full text-left text-xs p-2.5 rounded-xl surface-card hover:bg-surface-raised text-text-muted hover:text-foreground border border-surface-border transition-all flex items-center justify-between group shadow-subtle"
                          >
                            <span className="font-medium">{q}</span>
                            <ChevronRight className="h-3.5 w-3.5 text-text-subtle group-hover:text-primary-500 group-hover:translate-x-0.5 transition-transform flex-shrink-0 ml-2" />
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => {
                const isUser = msg.role === "user";
                return (
                  <div
                    key={msg.id}
                    className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    {!isUser && (
                      <div className="flex-shrink-0 h-7 w-7 rounded-lg bg-primary-500/10 text-primary-500 border border-primary-500/20 flex items-center justify-center mt-0.5 shadow-subtle">
                        <Bot className="h-3.5 w-3.5" />
                      </div>
                    )}

                    <div
                      className={`max-w-[85%] rounded-2xl p-4 text-xs transition-colors ${
                        isUser
                          ? "bg-primary-600 text-white rounded-br-none shadow-subtle font-medium leading-relaxed"
                          : "surface-card text-foreground rounded-bl-none shadow-card"
                      }`}
                    >
                      {isUser ? (
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      ) : (
                        <div className="relative group">
                          {/* Copy button on top-right */}
                          <button
                            onClick={() => handleCopyMessage(msg.id, msg.content)}
                            title="Copy response"
                            className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 p-1 rounded text-text-muted hover:text-foreground hover:bg-surface-raised transition-all"
                          >
                            {copiedMessageId === msg.id ? (
                              <Check className="h-3.5 w-3.5 text-accent-emerald" />
                            ) : (
                              <Copy className="h-3.5 w-3.5" />
                            )}
                          </button>

                          {renderMessageContent(msg.content, msg.tool_trace?.citations)}
                          <ToolTraceViewer toolTrace={msg.tool_trace} />
                        </div>
                      )}
                    </div>

                    {isUser && (
                      <div className="flex-shrink-0 h-7 w-7 rounded-lg bg-surface-raised border border-surface-border text-text-muted flex items-center justify-center mt-0.5 shadow-subtle">
                        <User className="h-3.5 w-3.5" />
                      </div>
                    )}
                  </div>
                );
              })
            )}

            {/* Dynamic Multi-Agent Step Animation */}
            {isSending && (
              <div className="flex items-start gap-3 animate-in fade-in duration-200">
                <div className="h-7 w-7 rounded-lg bg-primary-500/10 text-primary-500 border border-primary-500/20 flex items-center justify-center mt-0.5 flex-shrink-0">
                  <Bot className="h-3.5 w-3.5" />
                </div>
                <div className="p-3.5 rounded-xl surface-card border border-primary-500/30 shadow-subtle space-y-2 max-w-md">
                  <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                    <Loader2 className="h-3.5 w-3.5 text-primary-500 animate-spin flex-shrink-0" />
                    <span>Executing LangGraph Multi-Agent Flow</span>
                  </div>
                  <div className="text-[11px] font-mono text-primary-500 bg-primary-500/5 px-2.5 py-1.5 rounded-lg border border-primary-500/15">
                    {agentSteps[agentStepIndex]}
                  </div>
                  <div className="flex items-center gap-1.5 pt-1">
                    {agentSteps.map((_, i) => (
                      <div
                        key={i}
                        className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                          i <= agentStepIndex ? "bg-primary-500" : "bg-surface-raised"
                        }`}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Starter Chips Bar (when in active chat) */}
          {messages.length > 0 && !isSending && (
            <div className="px-4 py-2 border-t border-surface-border bg-surface-raised/20 flex items-center gap-2 overflow-x-auto no-scrollbar text-[11px]">
              <span className="text-text-subtle font-medium flex-shrink-0">Quick:</span>
              <button
                onClick={() => handleSendMessage("Explain the high-level architecture and key modules of this codebase.")}
                className="px-2.5 py-1 rounded-full surface-card hover:bg-surface-raised border border-surface-border text-text-muted hover:text-foreground flex-shrink-0 transition-colors"
              >
                🏛️ Architecture
              </button>
              <button
                onClick={() => handleSendMessage("How is authentication and authorization handled in this repository?")}
                className="px-2.5 py-1 rounded-full surface-card hover:bg-surface-raised border border-surface-border text-text-muted hover:text-foreground flex-shrink-0 transition-colors"
              >
                🔐 Authentication
              </button>
              <button
                onClick={() => handleSendMessage("Explain the data flow from UI to database step by step.")}
                className="px-2.5 py-1 rounded-full surface-card hover:bg-surface-raised border border-surface-border text-text-muted hover:text-foreground flex-shrink-0 transition-colors"
              >
                🔄 Data Flow
              </button>
              <button
                onClick={() => handleSendMessage("Who are the top contributors and recent changes to this repository?")}
                className="px-2.5 py-1 rounded-full surface-card hover:bg-surface-raised border border-surface-border text-text-muted hover:text-foreground flex-shrink-0 transition-colors"
              >
                👥 Git History
              </button>
            </div>
          )}

          {/* Prompt Input Bar */}
          <div className="p-3 sm:p-4 border-t border-surface-border bg-surface-raised/30">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="relative flex items-center"
            >
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask about architecture, symbols, entry points, or file dependencies..."
                disabled={isSending}
                className="w-full pl-4 pr-12 py-2.5 rounded-xl bg-surface border border-surface-border text-foreground text-xs placeholder:text-text-subtle focus:outline-none focus:border-primary-500 focus:ring-1 focus:ring-primary-500 shadow-subtle transition-all"
              />
              <button
                type="submit"
                disabled={!inputMessage.trim() || isSending}
                className="absolute right-1.5 p-1.5 rounded-lg bg-primary-600 hover:bg-primary-500 disabled:opacity-40 text-white transition-all shadow-sm"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Interactive Dependency Graph View */}
        <div className="w-full lg:w-1/2 flex flex-col bg-background h-[50vh] lg:h-full p-4 overflow-hidden relative">
          <GraphCanvas
            nodes={graphNodes}
            edges={graphEdges}
            highlightedPath={highlightedPath}
            onSelectNode={(node) => {
              setHighlightedPath(node.path);
              setSelectedFileForModal(node.path);
            }}
            onAskAboutFile={(path) => {
              handleSendMessage(`Explain what '${path}' does, its main exports, and how it interacts with the rest of the codebase.`);
            }}
          />
        </div>
      </div>

      <CodeViewerModal
        repoId={repoId}
        filePath={selectedFileForModal}
        onClose={() => setSelectedFileForModal(null)}
      />
    </div>
  );
}
