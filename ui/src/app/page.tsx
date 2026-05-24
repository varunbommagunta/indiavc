"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Sparkles,
  Search,
  TrendingUp,
  Eye,
  Shield,
  Brain,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import { type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

// ── types ─────────────────────────────────────────────────────────────────────

type AgentStatus = "pending" | "running" | "completed" | "failed";

interface AgentState {
  status: AgentStatus;
  outputPreview?: string;
  toolCalls?: number;
  error?: string;
}

interface ResearchState {
  question: string;
  sessionId: string | null;
  agents: Record<string, AgentState>;
  criticOutput: string | null;
  finalBrief: string | null;
  status:
    | "idle"
    | "running"
    | "awaiting_approval"
    | "approved"
    | "completed"
    | "rejected"
    | "refused";
  refusalReason?: string;
}

// ── constants ─────────────────────────────────────────────────────────────────

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface AgentMeta {
  name: string;
  icon: LucideIcon;
  color: string;
}

const AGENT_META: Record<string, AgentMeta> = {
  web_researcher: {
    name: "Web Researcher",
    icon: Search,
    color: "from-blue-500 to-cyan-500",
  },
  news_analyzer: {
    name: "News Analyzer",
    icon: TrendingUp,
    color: "from-purple-500 to-pink-500",
  },
  competitor_analyzer: {
    name: "Competitor Analyzer",
    icon: Eye,
    color: "from-orange-500 to-amber-500",
  },
  critic: {
    name: "Critic",
    icon: Shield,
    color: "from-rose-500 to-red-500",
  },
};

const INITIAL_AGENTS: Record<string, AgentState> = {
  web_researcher: { status: "pending" },
  news_analyzer: { status: "pending" },
  competitor_analyzer: { status: "pending" },
  critic: { status: "pending" },
};

function freshState(): ResearchState {
  return {
    question: "",
    sessionId: null,
    agents: { ...INITIAL_AGENTS },
    criticOutput: null,
    finalBrief: null,
    status: "idle",
  };
}

// ── main component ────────────────────────────────────────────────────────────

export default function Home() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<ResearchState>(freshState());
  const abortRef = useRef<AbortController | null>(null);

  // ── SSE stream ──────────────────────────────────────────────────────────────

  const startResearch = async () => {
    if (!question.trim()) return;

    setState({ ...freshState(), question, status: "running" });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_URL}/research/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";

        for (const chunk of chunks) {
          if (!chunk.startsWith("data: ")) continue;
          try {
            handleEvent(JSON.parse(chunk.slice(6)));
          } catch {
            // malformed event — skip
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "AbortError") {
        console.error("Stream error:", err);
        setState((prev) => ({ ...prev, status: "idle" }));
      }
    }
  };

  // ── event handler ───────────────────────────────────────────────────────────

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleEvent = (data: Record<string, any>) => {
    switch (data.event) {
      case "started":
        setState((prev) => ({ ...prev, sessionId: data.session_id as string }));
        break;
      case "refused":
        setState((prev) => ({
          ...prev,
          status: "refused",
          refusalReason: data.reason as string,
        }));
        break;
      case "agent_started":
        setState((prev) => ({
          ...prev,
          agents: {
            ...prev.agents,
            [data.agent]: { status: "running" },
          },
        }));
        break;
      case "agent_completed":
        setState((prev) => ({
          ...prev,
          agents: {
            ...prev.agents,
            [data.agent]: {
              status: "completed",
              outputPreview: data.output_preview as string,
              toolCalls: data.tool_calls as number,
            },
          },
        }));
        break;
      case "agent_failed":
        setState((prev) => ({
          ...prev,
          agents: {
            ...prev.agents,
            [data.agent]: { status: "failed", error: data.error as string },
          },
        }));
        break;
      case "critic_completed":
        setState((prev) => ({
          ...prev,
          agents: { ...prev.agents, critic: { status: "completed" } },
          criticOutput: data.output as string,
        }));
        break;
      case "awaiting_approval":
        setState((prev) => ({ ...prev, status: "awaiting_approval" }));
        break;
    }
  };

  // ── HITL actions ────────────────────────────────────────────────────────────

  const approveAndContinue = async () => {
    setState((prev) => ({ ...prev, status: "approved" }));
    try {
      const res = await fetch(`${API_URL}/research/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setState((prev) => ({
        ...prev,
        status: "completed",
        finalBrief: data.brief as string,
      }));
    } catch (err) {
      console.error("Approval error:", err);
      setState((prev) => ({ ...prev, status: "idle" }));
    }
  };

  const rejectResearch = async () => {
    if (state.sessionId) {
      await fetch(`${API_URL}/research/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      }).catch(() => {});
    }
    setState((prev) => ({ ...prev, status: "rejected" }));
  };

  const reset = () => {
    abortRef.current?.abort();
    setQuestion("");
    setState(freshState());
  };

  const isActive =
    state.status === "running" ||
    state.status === "awaiting_approval" ||
    state.status === "approved";

  // ── render ──────────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen bg-animated-gradient relative overflow-hidden">
      {/* Floating blobs */}
      <div className="blob bg-blue-600 w-96 h-96 -top-20 -left-20 animate-float" />
      <div
        className="blob bg-purple-600 w-96 h-96 top-1/3 -right-20 animate-float"
        style={{ animationDelay: "2s" }}
      />
      <div
        className="blob bg-cyan-500 w-96 h-96 bottom-0 left-1/3 animate-float"
        style={{ animationDelay: "4s" }}
      />

      {/* Unsplash hero background — abstract tech network */}
      <div className="hero-image" />

      {/* Unsplash secondary background — neural network pattern */}
      <div
        className="absolute inset-0 opacity-5 pointer-events-none"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1920&q=80')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          mixBlendMode: "screen",
        }}
      />

      {/* Subtle grid overlay */}
      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-16 md:px-12 lg:px-16">
        {/* Hero header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-6">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <span className="text-sm text-slate-300">AI Research Team</span>
          </div>

          <h1 className="text-7xl md:text-8xl lg:text-9xl font-bold gradient-text mb-4">
            IndiaVC
          </h1>

          <p className="text-2xl text-slate-400 max-w-2xl mx-auto">
            Multi-agent due diligence for Indian startups.
            <br />
            <span className="text-slate-500 text-base">
              5 specialized AI agents collaborate to produce investor-grade
              research briefs.
            </span>
          </p>
        </motion.header>

        {/* Main content — animated state transitions */}
        <AnimatePresence>
          {/* Idle — query input */}
          {state.status === "idle" && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="glass-strong rounded-2xl p-8 mb-8"
            >
              <label className="block text-sm font-medium text-slate-300 mb-3">
                What company or topic would you like to research?
              </label>
              <div className="relative group">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && startResearch()}
                  placeholder="e.g. Razorpay, Yubi, PhonePe vs Paytm comparison"
                  className="w-full px-5 py-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 transition-all"
                />
                <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500/0 via-blue-500/20 to-purple-500/0 opacity-0 group-focus-within:opacity-100 transition-opacity pointer-events-none -z-10 blur-xl" />
              </div>

              <button
                onClick={startResearch}
                disabled={!question.trim()}
                className="mt-6 w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed text-white font-semibold py-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2 group"
              >
                <span>Start Research</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </button>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                {[
                  { icon: Search, text: "Web research" },
                  { icon: TrendingUp, text: "News & sentiment analysis" },
                  { icon: Eye, text: "Competitor landscape" },
                  { icon: Shield, text: "Critic review for red flags" },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-2 text-slate-400">
                    <item.icon className="w-4 h-4 text-cyan-400" />
                    <span>{item.text}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Active — agent progress cards */}
          {isActive && (
            <motion.div
              key="running"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
              className="glass-strong rounded-2xl p-8 mb-8"
            >
              <div className="mb-6">
                <h2 className="text-3xl font-bold text-white mb-2">
                  Researching:{" "}
                  <span className="gradient-text">{state.question}</span>
                </h2>
                <p className="text-slate-400 text-sm">
                  AI agents are gathering and analyzing information…
                </p>
              </div>

              <div className="space-y-4">
                {Object.entries(state.agents).map(([name, agent]) => (
                  <AgentCard key={name} name={name} agent={agent} />
                ))}
              </div>

              {state.status === "approved" && (
                <div className="mt-6 flex items-center gap-2 text-blue-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Generating final brief…</span>
                </div>
              )}
            </motion.div>
          )}

          {/* HITL approval */}
          {state.status === "awaiting_approval" && state.criticOutput && (
            <motion.div
              key="hitl"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ type: "spring", stiffness: 200, damping: 20 }}
              className="glass-strong rounded-2xl border-2 border-amber-500/30 p-8 mb-8 glow-blue"
            >
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex-shrink-0">
                  <AlertTriangle className="w-6 h-6 text-amber-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">
                    Critic Review Complete
                  </h3>
                  <p className="text-sm text-slate-400">
                    Review the findings before generating the final investor
                    brief.
                  </p>
                </div>
              </div>

              <div className="bg-black/30 border border-white/5 rounded-xl p-6 mb-6 max-h-[400px] overflow-y-auto">
                <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
                  {state.criticOutput}
                </pre>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={approveAndContinue}
                  className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-semibold py-4 rounded-xl transition-all flex items-center justify-center gap-2"
                >
                  <CheckCircle className="w-5 h-5" />
                  <span>Approve &amp; Generate Brief</span>
                </button>
                <button
                  onClick={rejectResearch}
                  className="px-6 bg-white/5 hover:bg-white/10 text-slate-300 font-medium py-4 rounded-xl transition-all border border-white/10"
                >
                  Reject
                </button>
              </div>
            </motion.div>
          )}

          {/* Final brief */}
          {state.status === "completed" && state.finalBrief && (
            <motion.div
              key="completed"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="glass-strong rounded-2xl p-8 mb-8"
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 rounded-xl bg-green-500/10 border border-green-500/30">
                  <CheckCircle className="w-6 h-6 text-green-400" />
                </div>
                <h2 className="text-3xl font-bold text-white">
                  Investor Brief
                </h2>
              </div>

              <div className="bg-black/30 border border-white/5 rounded-xl p-6 mb-6 max-h-[60vh] overflow-y-auto">
                <pre className="text-sm text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
                  {state.finalBrief}
                </pre>
              </div>

              <button
                onClick={reset}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-semibold px-6 py-3 rounded-xl transition-all flex items-center gap-2 group"
              >
                <RefreshCw className="w-5 h-5 group-hover:rotate-180 transition-transform duration-500" />
                <span>New Research</span>
              </button>
            </motion.div>
          )}

          {/* Refused */}
          {state.status === "refused" && (
            <motion.div
              key="refused"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="glass-strong rounded-2xl border-2 border-red-500/30 p-8 mb-8"
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 flex-shrink-0">
                  <XCircle className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white mb-1">
                    Query Refused
                  </h3>
                  <p className="text-sm text-slate-400">
                    {state.refusalReason}
                  </p>
                </div>
              </div>
              <button
                onClick={reset}
                className="bg-white/5 hover:bg-white/10 text-slate-300 px-6 py-3 rounded-xl transition-all"
              >
                Try Another Query
              </button>
            </motion.div>
          )}

          {/* Rejected */}
          {state.status === "rejected" && (
            <motion.div
              key="rejected"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="glass-strong rounded-2xl p-8 mb-8 text-center"
            >
              <p className="text-slate-300 mb-4">Research session rejected.</p>
              <button
                onClick={reset}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold px-6 py-3 rounded-xl"
              >
                New Research
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <footer className="text-center mt-16 text-slate-500 text-sm">
          <p>Built with FastAPI, Next.js, OpenAI, and MCP</p>
          <p className="mt-2 text-xs">
            Research is automated and informational. Not financial advice.
          </p>
        </footer>
      </div>
    </main>
  );
}

// ── AgentCard ──────────────────────────────────────────────────────────────────

function AgentCard({ name, agent }: { name: string; agent: AgentState }) {
  const meta: AgentMeta = AGENT_META[name] ?? {
    name,
    icon: Brain,
    color: "from-slate-500 to-slate-700",
  };
  const Icon = meta.icon;
  const isRunning = agent.status === "running";
  const isCompleted = agent.status === "completed";
  const isFailed = agent.status === "failed";

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className={cn(
        "relative overflow-hidden rounded-xl border p-5 transition-all",
        "border-white/10 bg-white/[0.02]",
        isRunning && "border-blue-500/40 bg-blue-500/5",
        isCompleted && "border-green-500/30 bg-green-500/5",
        isFailed && "border-red-500/30 bg-red-500/5"
      )}
    >
      {isRunning && (
        <div className="absolute inset-0 shimmer pointer-events-none" />
      )}

      <div className="relative flex items-start gap-4">
        <div
          className={cn(
            "p-3 rounded-xl flex-shrink-0 bg-gradient-to-br",
            meta.color,
            isRunning && "animate-pulse-glow"
          )}
        >
          <Icon className="w-5 h-5 text-white" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-1">
            <h3 className="font-semibold text-white">{meta.name}</h3>
            <StatusIndicator status={agent.status} />
          </div>

          {agent.outputPreview && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-sm text-slate-400 line-clamp-2 mt-2"
            >
              {agent.outputPreview}
            </motion.p>
          )}

          {agent.toolCalls !== undefined && (
            <p className="text-xs text-slate-500 mt-2">
              {agent.toolCalls} tool{" "}
              {agent.toolCalls === 1 ? "call" : "calls"} made
            </p>
          )}

          {agent.error && (
            <p className="text-xs text-red-400 mt-2">{agent.error}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
}

// ── StatusIndicator ───────────────────────────────────────────────────────────

function StatusIndicator({ status }: { status: AgentStatus }) {
  if (status === "pending") {
    return <span className="text-xs text-slate-500">Pending</span>;
  }
  if (status === "running") {
    return (
      <div className="flex items-center gap-1.5">
        <Loader2 className="w-3 h-3 animate-spin text-blue-400" />
        <span className="text-xs text-blue-400 font-medium">Working</span>
      </div>
    );
  }
  if (status === "completed") {
    return (
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 500, damping: 15 }}
        className="flex items-center gap-1.5"
      >
        <CheckCircle className="w-3 h-3 text-green-400" />
        <span className="text-xs text-green-400 font-medium">Done</span>
      </motion.div>
    );
  }
  if (status === "failed") {
    return (
      <div className="flex items-center gap-1.5">
        <XCircle className="w-3 h-3 text-red-400" />
        <span className="text-xs text-red-400 font-medium">Failed</span>
      </div>
    );
  }
  return null;
}
