/**
 * Chat sessions persisted in the browser (localStorage).
 * They hold the user's own messages and tool *reports* only, never document text or PII values.
 */
import type { Recall, Source } from "./api";
import type { HistoryMessage } from "./chat";

export type ToolRun = {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  auto: boolean;
  ok?: boolean;
  ms?: number;
  report?: Record<string, unknown>;
  error?: string;
};

export type Attachment = {
  source?: Source["source"];
  name?: string;
  matched_sample?: string | null;
  label: string;
  preview: string;
  uploading?: boolean;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  attachment?: Attachment;
  tools: ToolRun[];
  resultIds: string[];
  recall?: Recall;
  error?: string;
  streaming?: boolean;
};

export type ChatSession = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
  history: HistoryMessage[];
};

const KEY = "safeocr.chats.v1";
const MAX_CHATS = 50;

export const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export function loadChats(): ChatSession[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as ChatSession[]).sort((a, b) => b.updatedAt - a.updatedAt) : [];
  } catch {
    return [];
  }
}

function forStorage(chat: ChatSession): ChatSession {
  return {
    ...chat,
    messages: chat.messages.map((m) => ({
      ...m,
      streaming: false,
      // blob: previews die with the tab; uploads are re-served by the backend anyway
      attachment: m.attachment && !m.attachment.preview.startsWith("blob:") ? m.attachment : undefined,
    })),
  };
}

export function saveChats(chats: ChatSession[]): void {
  let list = [...chats].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CHATS).map(forStorage);
  // on quota errors drop the oldest chats and retry
  while (list.length) {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(list));
      return;
    } catch {
      list = list.slice(0, -1);
    }
  }
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable (private mode) — chats simply won't persist */
  }
}

export function titleFrom(text: string, attachmentLabel?: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean) return clean.length > 48 ? `${clean.slice(0, 47)}…` : clean;
  return attachmentLabel ? `Image: ${attachmentLabel}` : "New chat";
}

export function groupByDate(chats: ChatSession[]): { label: string; items: ChatSession[] }[] {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const day = 86_400_000;
  const buckets: [string, (t: number) => boolean][] = [
    ["Today", (t) => t >= startOfToday.getTime()],
    ["Yesterday", (t) => t >= startOfToday.getTime() - day],
    ["Previous 7 days", (t) => t >= startOfToday.getTime() - 7 * day],
    ["Previous 30 days", (t) => t >= startOfToday.getTime() - 30 * day],
    ["Older", () => true],
  ];
  const groups = new Map<string, ChatSession[]>();
  for (const c of [...chats].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const label = buckets.find(([, test]) => test(c.updatedAt))![0];
    groups.set(label, [...(groups.get(label) ?? []), c]);
  }
  return [...groups.entries()].map(([label, items]) => ({ label, items }));
}

/** Approximate bytes used by stored chats (UTF-16 → 2 bytes per char). */
export function storageBytes(): number {
  try {
    return (window.localStorage.getItem(KEY)?.length ?? 0) * 2;
  } catch {
    return 0;
  }
}

export function exportChats(chats: ChatSession[]): void {
  const blob = new Blob([JSON.stringify(chats.map(forStorage), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), {
    href: url,
    download: `safeocr-chats-${new Date().toISOString().slice(0, 10)}.json`,
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
