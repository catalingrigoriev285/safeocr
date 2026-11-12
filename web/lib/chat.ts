import { API_URL, type Recall, type Source } from "./api";

/** Opaque model-history messages (contain only user text and tool reports, never PII values). */
export type HistoryMessage = Record<string, unknown>;

export type ChatEvent =
  | { type: "tool_call"; id: string; name: string; arguments: Record<string, unknown>; auto: boolean }
  | { type: "tool_result"; id: string; name: string; ok: boolean; ms: number; report?: Record<string, unknown>; error?: string }
  | ({ type: "recall" } & Recall)
  | { type: "token"; content: string }
  | { type: "error"; message: string }
  | { type: "done"; history: HistoryMessage[]; language?: string };

export type ChatRequest = {
  message: string;
  history: HistoryMessage[];
  attachment?: Pick<Source, "source" | "name"> | null;
};

/** POST /api/chat and dispatch each NDJSON event as it arrives. */
export async function streamChat(body: ChatRequest, onEvent: (e: ChatEvent) => void, signal?: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error(`Backend unreachable at ${API_URL}. Is uvicorn running?`);
  }
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(typeof detail?.detail === "string" ? detail.detail : `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as ChatEvent);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as ChatEvent);
}
