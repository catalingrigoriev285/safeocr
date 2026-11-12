"use client";

import Link from "next/link";
import type { Health } from "@/lib/api";
import { groupByDate, type ChatSession } from "@/lib/chats";

export const SidebarIcon = {
  panel: (
    <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <path d="M9 4v16" />
    </svg>
  ),
  compose: (
    <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  ),
  trash: (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />
    </svg>
  ),
  gear: (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
    </svg>
  ),
  layers: (
    <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5" />
    </svg>
  ),
};

type Props = {
  chats: ChatSession[];
  activeId: string | null;
  streamingId: string | null;
  health: Health | null;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
};

export default function Sidebar({ chats, activeId, streamingId, health, open, onClose, onSelect, onNew, onDelete, onOpenSettings }: Props) {
  const groups = groupByDate(chats);
  const modelReady = health?.ollama?.reachable && health.ollama.model_installed;

  return (
    <>
      {/* mobile backdrop */}
      <div
        onClick={onClose}
        className={`fixed inset-0 z-30 bg-black/40 transition-opacity md:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        aria-hidden
      />
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[260px] flex-col bg-zinc-50 transition-transform duration-200 dark:bg-zinc-900 md:static md:z-auto md:transition-[margin] ${
          open ? "translate-x-0 md:ml-0" : "-translate-x-full md:-ml-[260px] md:translate-x-0"
        }`}
        aria-label="Chat history"
      >
        <div className="flex h-14 items-center justify-between px-3">
          <span className="flex items-center gap-2 px-1 text-[15px] font-semibold">
            <span className="flex size-6 items-center justify-center rounded-md bg-zinc-900 text-[11px] font-bold text-white dark:bg-zinc-100 dark:text-zinc-900">
              S
            </span>
            SafeOCR
          </span>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-500 transition hover:bg-zinc-200 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
            aria-label="Close sidebar"
            title="Close sidebar"
          >
            {SidebarIcon.panel}
          </button>
        </div>

        <div className="px-3 pb-2">
          <button
            onClick={onNew}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium transition hover:bg-zinc-200 dark:hover:bg-zinc-800"
          >
            {SidebarIcon.compose}
            New chat
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 pb-3">
          {chats.length === 0 ? (
            <p className="px-2.5 pt-4 text-sm text-zinc-500">
              Your conversations will appear here. They are saved only in this browser.
            </p>
          ) : (
            groups.map((g) => (
              <div key={g.label} className="pt-4">
                <p className="px-2.5 pb-1 text-xs font-medium text-zinc-500">{g.label}</p>
                <ul>
                  {g.items.map((c) => {
                    const active = c.id === activeId;
                    return (
                      <li key={c.id} className="group relative">
                        <button
                          onClick={() => onSelect(c.id)}
                          className={`w-full truncate rounded-lg py-2 pr-8 pl-2.5 text-left text-sm transition ${
                            active
                              ? "bg-zinc-200 font-medium dark:bg-zinc-800"
                              : "text-zinc-700 hover:bg-zinc-200/70 dark:text-zinc-300 dark:hover:bg-zinc-800/70"
                          }`}
                          title={c.title}
                        >
                          {streamingId === c.id && (
                            <span className="mr-1.5 inline-block size-1.5 animate-pulse rounded-full bg-amber-400 align-middle" />
                          )}
                          {c.title}
                        </button>
                        <button
                          onClick={() => {
                            if (window.confirm(`Delete “${c.title}”?`)) onDelete(c.id);
                          }}
                          className={`absolute top-1/2 right-1.5 -translate-y-1/2 rounded-md p-1 text-zinc-400 transition hover:text-red-600 ${
                            active ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus:opacity-100"
                          }`}
                          aria-label={`Delete ${c.title}`}
                          title="Delete chat"
                        >
                          {SidebarIcon.trash}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
          )}
        </nav>

        <div className="space-y-1 border-t border-zinc-200 p-3 text-xs dark:border-zinc-800">
          <button
            onClick={onOpenSettings}
            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-zinc-600 transition hover:bg-zinc-200 dark:text-zinc-400 dark:hover:bg-zinc-800"
          >
            {SidebarIcon.gear}
            Settings
          </button>
          <Link
            href="/pipeline"
            className="flex items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-zinc-600 transition hover:bg-zinc-200 dark:text-zinc-400 dark:hover:bg-zinc-800"
          >
            {SidebarIcon.layers}
            Pipeline view
          </Link>
          <div className="flex items-center gap-2 px-2.5 py-1 text-zinc-500">
            <span className={`size-1.5 rounded-full ${modelReady ? "bg-emerald-500" : "bg-red-500"}`} />
            <span className="truncate">
              {health?.chat_model ?? "model"} ·{" "}
              {!health ? "API offline" : !health.ollama?.reachable ? "Ollama not running" : modelReady ? "local" : "model not pulled"}
            </span>
          </div>
          <div className="px-2.5 text-zinc-400">OCR: {health?.ocr_backend ?? "?"} · stored in this browser</div>
        </div>
      </aside>
    </>
  );
}
