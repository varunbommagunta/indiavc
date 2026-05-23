"use client";

import { useRef, useState } from "react";
import { AlertTriangle, CheckCircle, Loader2, XCircle } from "lucide-react";

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

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const AGENT_LABELS: Record<string, string> = {
  web_researcher: "🔍 Web Researcher",
  news_analyzer: "📰 News Analyzer",
  competitor_analyzer: "🏢 Competitor Analyzer",
  critic: "🎯 Critic",
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
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900">IndiaVC</h1>
          <p className="text-slate-600 mt-2">
            AI Research Team for Indian Startup Due Diligence
          </p>
        </header>

        {/* Input */}
        {state.status === "idle" && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Company or topic to research
            </label>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && startResearch()}
              placeholder="e.g. Razorpay, Yubi, PhonePe vs Paytm comparison"
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
            />
            <button
              onClick={startResearch}
              disabled={!question.trim()}
              className="mt-4 w-full bg-blue-600 text-white font-medium py-3 rounded-lg hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
            >
              Start Research
            </button>
          </div>
        )}

        {/* Live progress */}
        {isActive && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 text-slate-800">
              Researching:{" "}
              <span className="text-blue-600">{state.question}</span>
            </h2>
            <div className="space-y-3">
              {Object.entries(state.agents).map(([name, agent]) => (
                <AgentCard key={name} name={name} agent={agent} />
              ))}
            </div>
            {state.status === "approved" && (
              <div className="mt-4 flex items-center gap-2 text-blue-600">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Generating final brief…</span>
              </div>
            )}
          </div>
        )}

        {/* HITL approval */}
        {state.status === "awaiting_approval" && state.criticOutput && (
          <div className="bg-amber-50 border-2 border-amber-300 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-amber-900">
                  Critic&apos;s Assessment Ready
                </h3>
                <p className="text-sm text-amber-800 mt-1">
                  Review the critic&apos;s findings before generating the final brief.
                </p>
              </div>
            </div>

            <div className="bg-white rounded border border-amber-200 p-4 mb-4 max-h-96 overflow-y-auto">
              <pre className="text-sm text-slate-700 whitespace-pre-wrap font-sans leading-relaxed">
                {state.criticOutput}
              </pre>
            </div>

            <div className="flex gap-3">
              <button
                onClick={approveAndContinue}
                className="flex-1 bg-green-600 text-white font-medium py-3 rounded-lg hover:bg-green-700 transition-colors"
              >
                Approve &amp; Generate Final Brief
              </button>
              <button
                onClick={rejectResearch}
                className="flex-1 bg-slate-200 text-slate-700 font-medium py-3 rounded-lg hover:bg-slate-300 transition-colors"
              >
                Reject
              </button>
            </div>
          </div>
        )}

        {/* Final brief */}
        {state.status === "completed" && state.finalBrief && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 text-green-700 flex items-center gap-2">
              <CheckCircle className="w-6 h-6" /> Final Investor Brief
            </h2>
            <div className="bg-slate-50 rounded-lg p-4 max-h-[60vh] overflow-y-auto">
              <pre className="text-sm text-slate-800 whitespace-pre-wrap font-sans leading-relaxed">
                {state.finalBrief}
              </pre>
            </div>
            <button
              onClick={reset}
              className="mt-6 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              New Research
            </button>
          </div>
        )}

        {/* Refused */}
        {state.status === "refused" && (
          <div className="bg-red-50 border border-red-300 rounded-lg p-6 mb-6">
            <div className="flex items-start gap-3">
              <XCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-red-900">
                  Query Refused
                </h3>
                <p className="text-sm text-red-800 mt-1">
                  {state.refusalReason}
                </p>
              </div>
            </div>
            <button
              onClick={reset}
              className="mt-4 bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Rejected */}
        {state.status === "rejected" && (
          <div className="bg-slate-50 rounded-lg p-6 mb-6 text-center">
            <p className="text-slate-600">Research was rejected.</p>
            <button
              onClick={reset}
              className="mt-4 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              New Research
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

// ── AgentCard sub-component ───────────────────────────────────────────────────

function AgentCard({ name, agent }: { name: string; agent: AgentState }) {
  const label = AGENT_LABELS[name] ?? name;

  return (
    <div className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg border border-slate-100">
      <div className="flex-shrink-0 mt-0.5">
        {agent.status === "pending" && (
          <div className="w-5 h-5 rounded-full bg-slate-300" />
        )}
        {agent.status === "running" && (
          <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
        )}
        {agent.status === "completed" && (
          <CheckCircle className="w-5 h-5 text-green-600" />
        )}
        {agent.status === "failed" && (
          <XCircle className="w-5 h-5 text-red-600" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-medium text-slate-900 text-sm">{label}</div>
        {agent.outputPreview && (
          <p className="text-xs text-slate-600 mt-1 line-clamp-2 leading-relaxed">
            {agent.outputPreview}
          </p>
        )}
        {agent.toolCalls !== undefined && (
          <span className="text-xs text-slate-400 mt-1 block">
            {agent.toolCalls} search{agent.toolCalls !== 1 ? "es" : ""}
          </span>
        )}
        {agent.error && (
          <p className="text-xs text-red-600 mt-1">{agent.error}</p>
        )}
      </div>
    </div>
  );
}
