"use client";

import { useEffect, useState } from "react";
import { API_URL, type Health } from "@/lib/api";
import { exportChats, storageBytes, type ChatSession } from "@/lib/chats";
import type { Prefs } from "@/lib/prefs";

type Props = {
  open: boolean;
  onClose: () => void;
  chats: ChatSession[];
  health: Health | null;
  prefs: Prefs;
  onPrefsChange: (p: Prefs) => void;
  onDeleteAll: () => void;
};

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-zinc-200 py-4 first:border-t-0 first:pt-0 dark:border-zinc-800">
      <h3 className="mb-3 text-xs font-semibold tracking-wide text-zinc-500 uppercase">{title}</h3>
      {children}
    </section>
  );
}

function Toggle({ checked, disabled, onChange, label }: { checked: boolean; disabled?: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative h-6 w-11 shrink-0 rounded-full transition disabled:cursor-not-allowed disabled:opacity-40 ${
        checked ? "bg-emerald-500" : "bg-zinc-300 dark:bg-zinc-700"
      }`}
    >
      <span className={`absolute top-0.5 left-0.5 size-5 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-5" : ""}`} />
    </button>
  );
}

export default function SettingsDialog({ open, onClose, chats, health, prefs, onPrefsChange, onDeleteAll }: Props) {
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const close = () => {
    setConfirming(false);
    onClose();
  };
  const tts = health?.tts;
  const ttsReady = !!tts?.enabled && tts.engine !== "none";
  const row = "flex items-start justify-between gap-4";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={close} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        className="relative max-h-[90dvh] w-full max-w-lg overflow-y-auto rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-800 dark:bg-zinc-900"
      >
        <div className="mb-5 flex items-center justify-between">
          <h2 id="settings-title" className="text-lg font-semibold">
            Settings
          </h2>
          <button
            onClick={close}
            className="rounded-lg p-1.5 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
            aria-label="Close settings"
          >
            <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        <Section title="Voice">
          <div className={row}>
            <div>
              <p className="text-sm font-medium">Read results aloud</p>
              <p className="mt-0.5 text-sm text-zinc-500">
                {ttsReady
                  ? `Plays the spoken summary after each anonymization (${tts!.engine}${tts!.voice ? ` · ${tts!.voice}` : ""}, ${tts!.language}).`
                  : "No voice engine is enabled. Set tts.enabled: true in config/settings.yaml (see the README)."}
              </p>
            </div>
            <Toggle
              label="Read results aloud"
              checked={ttsReady && prefs.autoplayVoice}
              disabled={!ttsReady}
              onChange={(v) => onPrefsChange({ ...prefs, autoplayVoice: v })}
            />
          </div>

          <div className={`${row} mt-4`}>
            <div>
              <p className="text-sm font-medium">Voice input &amp; voice mode</p>
              <p className="mt-0.5 text-sm text-zinc-500">
                {health?.stt?.enabled
                  ? `Speech is transcribed locally (${health.stt.engine} · ${health.stt.model}). Use the mic in the composer to dictate, or the waveform button to talk.`
                  : "Speech recognition is off. Set stt.enabled: true in config/settings.yaml (see the README)."}
              </p>
            </div>
            <span
              className={`mt-1 shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                health?.stt?.enabled
                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                  : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800"
              }`}
            >
              {health?.stt?.enabled ? "On" : "Off"}
            </span>
          </div>
        </Section>

        <Section title="Data">
          <p className="text-sm">
            {chats.length} {chats.length === 1 ? "chat" : "chats"} stored in this browser ({formatBytes(storageBytes())}).
          </p>
          <p className="mt-0.5 text-sm text-zinc-500">
            Chats hold your messages and tool reports only, never document text. Processed images stay in the backend&apos;s
            data/store folder.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={() => exportChats(chats)}
              disabled={chats.length === 0}
              className="rounded-lg border border-zinc-200 px-3 py-1.5 text-sm font-medium transition hover:bg-zinc-50 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
            >
              Export chats
            </button>
            {!confirming ? (
              <button
                onClick={() => setConfirming(true)}
                disabled={chats.length === 0}
                className="rounded-lg border border-red-200 px-3 py-1.5 text-sm font-medium text-red-600 transition hover:bg-red-50 disabled:opacity-40 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950/40"
              >
                Delete all chats
              </button>
            ) : (
              <div className="flex w-full items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 dark:border-red-900 dark:bg-red-950/40">
                <span className="text-sm text-red-700 dark:text-red-300">
                  {chats.length === 1 ? "Delete this chat?" : `Delete all ${chats.length} chats?`} This cannot be undone.
                </span>
                <span className="flex shrink-0 gap-2">
                  <button onClick={() => setConfirming(false)} className="rounded-md px-2.5 py-1 text-sm hover:bg-white/60 dark:hover:bg-zinc-800">
                    Cancel
                  </button>
                  <button
                    onClick={() => {
                      onDeleteAll();
                      setConfirming(false);
                    }}
                    className="rounded-md bg-red-600 px-2.5 py-1 text-sm font-medium text-white hover:bg-red-500"
                  >
                    Delete all
                  </button>
                </span>
              </div>
            )}
          </div>
        </Section>

        <Section title="About">
          <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-sm">
            <dt className="text-zinc-500">Chat model</dt>
            <dd>
              {health?.chat_model ?? "?"}{" "}
              <span className="text-zinc-500">
                ({!health ? "API offline" : !health.ollama?.reachable ? "Ollama not running" : health.ollama.model_installed ? "Ollama, local" : "not pulled"})
              </span>
            </dd>
            <dt className="text-zinc-500">OCR</dt>
            <dd>{health?.ocr_backend ?? "?"}</dd>
            <dt className="text-zinc-500">Voice output</dt>
            <dd>{ttsReady ? `${tts!.engine}${tts!.voice ? ` · ${tts!.voice}` : ""}` : "off"}</dd>
            <dt className="text-zinc-500">Voice input</dt>
            <dd>{health?.stt?.enabled ? `${health.stt.engine} · ${health.stt.model}` : "off"}</dd>
            <dt className="text-zinc-500">Backend</dt>
            <dd className="truncate font-mono text-xs leading-5">{API_URL}</dd>
          </dl>
        </Section>
      </div>
    </div>
  );
}
