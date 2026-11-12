"use client";

/* eslint-disable @next/next/no-img-element -- images come from the local FastAPI backend / blob URLs */
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Health, type Sample } from "@/lib/api";
import { streamChat, type ChatEvent } from "@/lib/chat";
import {
  loadChats,
  newId,
  saveChats,
  titleFrom,
  type Attachment,
  type ChatSession,
  type Message,
  type ToolRun,
} from "@/lib/chats";
import { loadPrefs, savePrefs, type Prefs } from "@/lib/prefs";
import SettingsDialog from "./SettingsDialog";
import Sidebar, { SidebarIcon } from "./Sidebar";

const DESKTOP = "(min-width: 768px)";

const TYPE_LABEL: Record<string, string> = {
  CNP: "CNP",
  NAME: "Name",
  ADDRESS: "Address",
  ID_SERIES: "Series",
  ID_NUMBER: "Number",
  MRZ: "MRZ",
};

const SUGGESTIONS = [
  "What personal data can you detect?",
  "How do you protect my privacy?",
  "What happens to the original image?",
];

// ---------- icons ----------
const Icon = {
  clip: (
    <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  ),
  up: (
    <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  ),
  stop: (
    <svg viewBox="0 0 24 24" className="size-4" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  ),
  x: (
    <svg viewBox="0 0 24 24" className="size-3.5" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  ),
  chevron: (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
  image: (
    <svg viewBox="0 0 24 24" className="size-8" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </svg>
  ),
};

// ---------- small render helpers ----------
function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : part.startsWith("`") && part.endsWith("`") ? (
          <code key={i} className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[0.85em] dark:bg-zinc-800">
            {part.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

/** Tiny markdown subset: paragraphs, bullet/numbered lists, **bold**, `code`. */
function RichText({ text }: { text: string }) {
  const blocks: { list: boolean; lines: string[] }[] = [];
  for (const line of text.split("\n")) {
    const isItem = /^\s*([-*•]|\d+[.)])\s+/.test(line);
    const last = blocks[blocks.length - 1];
    if (!line.trim()) {
      blocks.push({ list: false, lines: [] });
    } else if (last && last.list === isItem && (isItem || last.lines.length)) {
      last.lines.push(isItem ? line.replace(/^\s*([-*•]|\d+[.)])\s+/, "") : line);
    } else {
      blocks.push({ list: isItem, lines: [isItem ? line.replace(/^\s*([-*•]|\d+[.)])\s+/, "") : line] });
    }
  }
  return (
    <div className="space-y-3 leading-7">
      {blocks
        .filter((b) => b.lines.length)
        .map((b, i) =>
          b.list ? (
            <ul key={i} className="list-disc space-y-1 pl-5">
              {b.lines.map((l, j) => (
                <li key={j}>
                  <Inline text={l} />
                </li>
              ))}
            </ul>
          ) : (
            <p key={i} className="whitespace-pre-wrap">
              <Inline text={b.lines.join("\n")} />
            </p>
          ),
        )}
    </div>
  );
}

function countsText(counts: unknown): string {
  if (!counts || typeof counts !== "object") return "";
  const entries = Object.entries(counts as Record<string, number>);
  return entries.length ? entries.map(([t, n]) => `${TYPE_LABEL[t] ?? t} ×${n}`).join(", ") : "none";
}

function toolSummary(t: ToolRun): string {
  if (t.ok === undefined) return "running…";
  if (!t.ok) return t.error ?? "failed";
  const r = t.report ?? {};
  switch (t.name) {
    case "ocr_document":
      return `${r.word_count} words read (${r.ocr_backend})`;
    case "detect_pii":
      return countsText(r.counts);
    case "redact_image":
      return `${r.boxes_drawn} boxes · ${countsText(r.redacted_counts)}`;
    case "verify_redaction":
      return r.passed ? "nothing detectable left" : `still visible: ${countsText(r.remaining_counts)}`;
    case "announce":
      return `“${r.text}”`;
    default:
      return "done";
  }
}

function argsText(args: Record<string, unknown>): string {
  return Object.entries(args)
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : String(v)}`)
    .join(", ");
}

function ToolRuns({ tools, streaming }: { tools: ToolRun[]; streaming?: boolean }) {
  const [open, setOpen] = useState(false);
  const failed = tools.some((t) => t.ok === false);
  const running = tools.some((t) => t.ok === undefined);
  const expanded = open || (streaming && running);
  return (
    <div className="rounded-xl border border-zinc-200 text-sm dark:border-zinc-800">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-zinc-600 dark:text-zinc-400"
      >
        <span
          className={`size-2 rounded-full ${failed ? "bg-red-500" : running ? "animate-pulse bg-amber-400" : "bg-emerald-500"}`}
          aria-hidden
        />
        <span className="flex-1">
          {running ? "Running MCP tools…" : `Used ${tools.length} MCP tool${tools.length === 1 ? "" : "s"}`}
          <span className="ml-2 text-xs text-zinc-400">only reports, no document text</span>
        </span>
        <span className={`transition ${expanded ? "rotate-180" : ""}`}>{Icon.chevron}</span>
      </button>
      {expanded && (
        <ol className="space-y-1.5 border-t border-zinc-200 px-3 py-2 dark:border-zinc-800">
          {tools.map((t) => (
            <li key={t.id} className="flex gap-2">
              <span className={`mt-0.5 font-mono text-xs ${t.ok === false ? "text-red-500" : t.ok ? "text-emerald-600" : "text-amber-500"}`}>
                {t.ok === false ? "✗" : t.ok ? "✓" : "•"}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <code className="font-mono text-[12.5px] font-medium">{t.name}</code>
                  <span className="truncate font-mono text-[11px] text-zinc-400">({argsText(t.arguments)})</span>
                  {!t.auto && <span className="rounded bg-indigo-100 px-1.5 text-[10px] font-medium text-indigo-700 dark:bg-indigo-900/50 dark:text-indigo-300">called by model</span>}
                  {t.ms !== undefined && <span className="ml-auto font-mono text-[11px] text-zinc-400">{t.ms} ms</span>}
                </div>
                <div className={`text-[13px] ${t.ok === false ? "text-red-600 dark:text-red-400" : "text-zinc-600 dark:text-zinc-400"}`}>
                  {toolSummary(t)}
                </div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function Typing() {
  return (
    <span className="inline-flex gap-1 py-2" aria-label="thinking">
      {[0, 150, 300].map((d) => (
        <span key={d} className="size-1.5 animate-bounce rounded-full bg-zinc-400" style={{ animationDelay: `${d}ms` }} />
      ))}
    </span>
  );
}

function PlayButton({ resultId }: { resultId: string }) {
  const [playing, setPlaying] = useState(false);
  const audio = useRef<HTMLAudioElement>(null);
  const toggle = () => {
    const el = audio.current;
    if (!el) return;
    if (playing) {
      el.pause();
      setPlaying(false);
    } else {
      el.play()
        .then(() => setPlaying(true))
        .catch(() => setPlaying(false));
    }
  };
  return (
    <button
      onClick={toggle}
      className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 px-3 py-1 text-xs font-medium text-zinc-600 transition hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
    >
      {playing ? (
        <svg viewBox="0 0 24 24" className="size-3.5" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className="size-3.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 5 6 9H2v6h4l5 4V5Z" />
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07M19.07 4.93a10 10 0 0 1 0 14.14" />
        </svg>
      )}
      {playing ? "Pause" : "Play spoken summary"}
      <audio ref={audio} src={api.audioUrl(resultId)} preload="none" onEnded={() => setPlaying(false)} hidden />
    </button>
  );
}

function AssistantMessage({ m }: { m: Message }) {
  const spoken = m.tools.find((t) => t.name === "announce" && t.ok && t.report?.audio);
  const showTyping = m.streaming && !m.text && (m.tools.length === 0 || m.tools.every((t) => t.ok !== undefined));
  return (
    <div className="flex gap-4">
      <div className="mt-1 flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-900 text-[11px] font-bold text-white dark:bg-zinc-100 dark:text-zinc-900">
        S
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {m.tools.length > 0 && <ToolRuns tools={m.tools} streaming={m.streaming} />}
        {m.resultIds.map((rid) => (
          <figure key={rid} className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
            <img src={api.imageUrl("result", rid)} alt="Redacted document" className="w-full" />
            <figcaption className="flex items-center justify-between gap-2 px-3 py-2 text-xs text-zinc-500">
              <span>
                Redacted image <code className="font-mono">{rid}</code>
              </span>
              <a href={api.imageUrl("result", rid)} download={`${rid}.png`} className="font-medium text-indigo-600 hover:underline">
                Download
              </a>
            </figcaption>
          </figure>
        ))}
        {spoken && <PlayButton resultId={String(spoken.report?.result_id)} />}
        {m.recall && (
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 font-medium text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300">
              Recall {m.recall.covered}/{m.recall.total} ({Math.round(m.recall.recall * 100)}%)
            </span>
            {Object.entries(m.recall.per_type).map(([t, s]) => (
              <span key={t} className="rounded-full bg-zinc-100 px-2 py-0.5 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
                {TYPE_LABEL[t] ?? t} {s.covered}/{s.total}
              </span>
            ))}
          </div>
        )}
        {m.text && <RichText text={m.text} />}
        {showTyping && <Typing />}
        {m.error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
            {m.error}
          </div>
        )}
      </div>
    </div>
  );
}

function UserMessage({ m }: { m: Message }) {
  return (
    <div className="flex flex-col items-end gap-2">
      {m.attachment && (
        <img src={m.attachment.preview} alt={m.attachment.label} className="max-h-56 max-w-[70%] rounded-2xl border border-zinc-200 object-contain dark:border-zinc-800" />
      )}
      {m.text && (
        <div className="max-w-[80%] whitespace-pre-wrap rounded-3xl bg-zinc-100 px-4 py-2.5 leading-7 dark:bg-zinc-800">{m.text}</div>
      )}
    </div>
  );
}


// ---------- main ----------
export default function Chat() {
  const [health, setHealth] = useState<Health | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [chats, setChats] = useState<ChatSession[]>(() => loadChats()); // client-only component
  const [activeId, setActiveId] = useState<string | null>(null);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.matchMedia(DESKTOP).matches);
  const [input, setInput] = useState("");
  const [attachment, setAttachment] = useState<Attachment | null>(null);
  const [prefs, setPrefs] = useState<Prefs>(() => loadPrefs());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const dragDepth = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const active = chats.find((c) => c.id === activeId) ?? null;
  const messages = useMemo(() => active?.messages ?? [], [active]);
  const streaming = streamingId !== null;

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.samples().then(setSamples).catch(() => setSamples([]));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // persist to localStorage (not on every streamed token)
  useEffect(() => {
    if (!streaming) saveChats(chats);
  }, [chats, streaming]);

  const isMobile = () => !window.matchMedia(DESKTOP).matches;
  const updateChat = (id: string, fn: (c: ChatSession) => ChatSession) =>
    setChats((cs) => cs.map((c) => (c.id === id ? fn(c) : c)));

  const resizeTextarea = () => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  // ----- chat management -----
  const stop = () => abortRef.current?.abort();

  const newChat = () => {
    stop();
    setActiveId(null);
    setAttachment(null);
    setNotice(null);
    setInput("");
    if (isMobile()) setSidebarOpen(false);
    requestAnimationFrame(() => textarea.current?.focus());
  };

  const selectChat = (id: string) => {
    if (id !== activeId) {
      stop();
      setActiveId(id);
      setAttachment(null);
      setNotice(null);
    }
    if (isMobile()) setSidebarOpen(false);
  };

  const deleteAllChats = () => {
    stop();
    setChats([]);
    setActiveId(null);
  };

  const changePrefs = (p: Prefs) => {
    setPrefs(p);
    savePrefs(p);
  };

  const deleteChat = (id: string) => {
    if (streamingId === id) stop();
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  // Ctrl/Cmd + Shift + O -> new chat (like ChatGPT)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        document.getElementById("new-chat-trigger")?.click();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ----- attachments -----
  const attachFile = async (file?: File | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setNotice("Only images are supported (PNG, JPG, …).");
      return;
    }
    setNotice(null);
    const blob = URL.createObjectURL(file);
    setAttachment({ label: file.name, preview: blob, uploading: true });
    try {
      const src = await api.upload(file);
      // switch to the backend URL so the preview survives reloads (chat history)
      setAttachment({ ...src, label: file.name, preview: api.imageUrl("upload", src.name) });
    } catch (e) {
      setAttachment(null);
      setNotice((e as Error).message);
    } finally {
      setTimeout(() => URL.revokeObjectURL(blob), 5000);
    }
    textarea.current?.focus();
  };

  const attachSample = (s: Sample) => {
    setNotice(null);
    setAttachment({ source: "sample", name: s.name, label: s.name, preview: api.imageUrl("sample", s.name) });
    textarea.current?.focus();
  };

  // ----- sending -----
  const send = async (override?: string) => {
    const text = (override ?? input).trim();
    const att = attachment;
    if ((!text && !att) || streaming || att?.uploading) return;

    const now = Date.now();
    const history = active?.history ?? [];
    const userMsg: Message = { id: newId(), role: "user", text, attachment: att ?? undefined, tools: [], resultIds: [] };
    const asstId = newId();
    const asstMsg: Message = { id: asstId, role: "assistant", text: "", tools: [], resultIds: [], streaming: true };

    let chatId = activeId;
    if (!chatId) {
      chatId = newId();
      const session: ChatSession = {
        id: chatId,
        title: titleFrom(text, att?.label),
        createdAt: now,
        updatedAt: now,
        messages: [userMsg, asstMsg],
        history: [],
      };
      setChats((cs) => [session, ...cs]);
      setActiveId(chatId);
    } else {
      updateChat(chatId, (c) => ({ ...c, updatedAt: now, messages: [...c.messages, userMsg, asstMsg] }));
    }
    const id: string = chatId;

    setInput("");
    setAttachment(null);
    setNotice(null);
    requestAnimationFrame(resizeTextarea);
    setStreamingId(id);

    const update = (fn: (m: Message) => Message) =>
      updateChat(id, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === asstId ? fn(m) : m)) }));

    const onEvent = (ev: ChatEvent) => {
      switch (ev.type) {
        case "tool_call":
          update((m) => ({ ...m, tools: [...m.tools, { id: ev.id, name: ev.name, arguments: ev.arguments, auto: ev.auto }] }));
          break;
        case "tool_result":
          update((m) => ({
            ...m,
            tools: m.tools.map((t) => (t.id === ev.id ? { ...t, ok: ev.ok, ms: ev.ms, report: ev.report, error: ev.error } : t)),
            resultIds:
              ev.ok && ev.name === "redact_image" && typeof ev.report?.result_id === "string"
                ? [...m.resultIds, ev.report.result_id]
                : m.resultIds,
          }));
          if (ev.ok && ev.name === "announce" && ev.report?.audio && prefs.autoplayVoice && health?.tts?.enabled) {
            new Audio(api.audioUrl(String(ev.report.result_id))).play().catch(() => {});
          }
          break;
        case "recall": {
          const { covered, total, recall, per_type } = ev;
          update((m) => ({ ...m, recall: { covered, total, recall, per_type } }));
          break;
        }
        case "token":
          update((m) => ({ ...m, text: m.text + ev.content }));
          break;
        case "error":
          update((m) => ({ ...m, error: ev.message }));
          break;
        case "done":
          updateChat(id, (c) => ({ ...c, history: ev.history, updatedAt: Date.now() }));
          break;
      }
    };

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        { message: text, history, attachment: att?.source && att.name ? { source: att.source, name: att.name } : null },
        onEvent,
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") update((m) => ({ ...m, error: (e as Error).message }));
    } finally {
      update((m) => ({ ...m, streaming: false }));
      setStreamingId(null);
      abortRef.current = null;
    }
  };

  // ----- drag & drop anywhere on the chat -----
  const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer.types).includes("Files");
  const dropHandlers = {
    onDragEnter: (e: React.DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    },
    onDragOver: (e: React.DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    },
    onDragLeave: () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      attachFile(e.dataTransfer.files?.[0]);
    },
  };

  const stubUploadWarning =
    health?.ocr_backend === "stub" && attachment?.source === "upload" && !attachment.uploading && !attachment.matched_sample;
  const canSend = (!!input.trim() || !!attachment) && !attachment?.uploading && !streaming;
  const iconButton =
    "rounded-lg p-2 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100";

  return (
    <div className="flex h-dvh overflow-hidden">
      <button id="new-chat-trigger" hidden onClick={newChat} />
      <Sidebar
        chats={chats}
        activeId={activeId}
        streamingId={streamingId}
        health={health}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={selectChat}
        onNew={newChat}
        onDelete={deleteChat}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        chats={chats}
        health={health}
        prefs={prefs}
        onPrefsChange={changePrefs}
        onDeleteAll={deleteAllChats}
      />

      <main className="relative flex min-w-0 flex-1 flex-col" {...dropHandlers}>
        {dragging && (
          <div className="pointer-events-none absolute inset-3 z-20 flex flex-col items-center justify-center gap-2 rounded-3xl border-2 border-dashed border-indigo-400 bg-indigo-50/90 text-indigo-700 dark:bg-indigo-950/80 dark:text-indigo-200">
            {Icon.image}
            <p className="text-lg font-medium">Drop the ID card image here</p>
            <p className="text-sm opacity-75">It is processed locally. The model only sees reports.</p>
          </div>
        )}

        {/* top bar */}
        <header className="flex h-14 shrink-0 items-center gap-1 px-3">
          {!sidebarOpen && (
            <>
              <button onClick={() => setSidebarOpen(true)} className={iconButton} aria-label="Open sidebar" title="Open sidebar">
                {SidebarIcon.panel}
              </button>
              <button onClick={newChat} className={iconButton} aria-label="New chat" title="New chat (⌘⇧O)">
                {SidebarIcon.compose}
              </button>
            </>
          )}
          <div className="ml-1 flex min-w-0 items-baseline gap-2">
            <span className="text-[17px] font-semibold">SafeOCR</span>
            <span className="truncate text-sm text-zinc-500">{health?.chat_model ?? ""}</span>
          </div>
          {active && <span className="ml-auto hidden max-w-[40%] truncate text-sm text-zinc-500 sm:block">{active.title}</span>}
        </header>

        {/* messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center px-4 pb-10 text-center">
              <h1 className="text-3xl font-semibold tracking-tight">What should we anonymize today?</h1>
              <p className="mt-2 max-w-md text-sm text-zinc-500">
                Attach a (fictive) ID card photo or ask a question, in any language. OCR, detection and redaction run
                locally. The model never sees the document text.
              </p>
              {samples.length > 0 && (
                <div className="mt-8 w-full">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-400">Try a generated sample</p>
                  <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                    {samples.slice(0, 6).map((s) => (
                      <button
                        key={s.name}
                        onClick={() => attachSample(s)}
                        className="overflow-hidden rounded-lg border border-zinc-200 transition hover:border-indigo-400 hover:shadow dark:border-zinc-800"
                        title={`Attach ${s.name}`}
                      >
                        <img src={api.imageUrl("sample", s.name)} alt={s.name} className="aspect-[3/2] w-full object-cover" />
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="rounded-full border border-zinc-200 px-3.5 py-1.5 text-sm text-zinc-600 transition hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-900"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-8 px-4 py-6">
              {messages.map((m) => (m.role === "user" ? <UserMessage key={m.id} m={m} /> : <AssistantMessage key={m.id} m={m} />))}
              <div ref={bottom} />
            </div>
          )}
        </div>

        {/* composer */}
        <div className="mx-auto w-full max-w-3xl px-4 pb-3">
          {notice && <p className="mb-2 text-center text-sm text-red-600 dark:text-red-400">{notice}</p>}
          <div className="rounded-3xl border border-zinc-200 bg-white p-2 shadow-sm focus-within:border-zinc-300 dark:border-zinc-700 dark:bg-zinc-900">
            {attachment && (
              <div className="px-2 pt-1 pb-2">
                <div className="relative inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-zinc-50 p-1.5 pr-3 dark:border-zinc-700 dark:bg-zinc-800">
                  <img src={attachment.preview} alt="" className={`size-12 rounded-lg object-cover ${attachment.uploading ? "opacity-50" : ""}`} />
                  <div className="text-left">
                    <p className="max-w-48 truncate text-sm font-medium">{attachment.label}</p>
                    <p className="text-xs text-zinc-500">
                      {attachment.uploading ? "Uploading…" : attachment.matched_sample ? `matches ${attachment.matched_sample}` : attachment.source}
                    </p>
                  </div>
                  <button
                    onClick={() => setAttachment(null)}
                    className="absolute -top-2 -right-2 rounded-full bg-zinc-900 p-1 text-white dark:bg-zinc-100 dark:text-zinc-900"
                    aria-label="Remove attachment"
                  >
                    {Icon.x}
                  </button>
                </div>
                {stubUploadWarning && (
                  <p className="mt-1.5 text-xs text-amber-600 dark:text-amber-400">
                    The stub OCR only reads generated samples. Install Tesseract to process real photos.
                  </p>
                )}
              </div>
            )}
            <div className="flex items-end gap-1">
              <button
                onClick={() => fileInput.current?.click()}
                className="rounded-full p-2 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
                aria-label="Attach image"
                title="Attach image"
              >
                {Icon.clip}
              </button>
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  attachFile(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
              <textarea
                ref={textarea}
                value={input}
                rows={1}
                autoFocus
                placeholder={attachment ? "Add a message (optional)…" : "Ask anything, or drop an ID card image…"}
                onChange={(e) => {
                  setInput(e.target.value);
                  resizeTextarea();
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    send();
                  }
                }}
                onPaste={(e) => {
                  const file = Array.from(e.clipboardData.files).find((f) => f.type.startsWith("image/"));
                  if (file) {
                    e.preventDefault();
                    attachFile(file);
                  }
                }}
                className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-2 leading-6 outline-none placeholder:text-zinc-400"
              />
              {streaming ? (
                <button onClick={stop} className="rounded-full bg-zinc-900 p-2.5 text-white dark:bg-zinc-100 dark:text-zinc-900" aria-label="Stop">
                  {Icon.stop}
                </button>
              ) : (
                <button
                  onClick={() => send()}
                  disabled={!canSend}
                  className="rounded-full bg-zinc-900 p-2 text-white transition disabled:bg-zinc-200 disabled:text-zinc-400 dark:bg-zinc-100 dark:text-zinc-900 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-600"
                  aria-label="Send"
                >
                  {Icon.up}
                </button>
              )}
            </div>
          </div>
          <p className="mt-2 text-center text-xs text-zinc-400">
            Runs locally. The model sees only reports, never your document&apos;s text.
          </p>
        </div>
      </main>
    </div>
  );
}
