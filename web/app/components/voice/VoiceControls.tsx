"use client";

import { useEffect, useRef, useState } from "react";
import { MicRecorder, SpeechPlayer, synthesize, transcribe } from "@/lib/voice";

/** Tiny live level meter (4 bars) for dictation. */
function LevelBars({ getLevel }: { getLevel: () => number }) {
  const bars = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let raf = 0;
    const t0 = performance.now();
    const loop = () => {
      const level = getLevel();
      const t = (performance.now() - t0) / 1000;
      const el = bars.current;
      if (el) {
        Array.from(el.children).forEach((child, i) => {
          const h = 0.2 + Math.min(1, level * (1.2 + 0.5 * Math.sin(t * 9 + i * 1.7)));
          (child as HTMLElement).style.setProperty("transform", `scaleY(${h})`);
        });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [getLevel]);
  return (
    <span ref={bars} className="flex h-4 items-center gap-[3px]" aria-hidden>
      {[0, 1, 2, 3].map((i) => (
        <span key={i} className="h-4 w-[3px] origin-center rounded-full bg-current transition-transform duration-75" />
      ))}
    </span>
  );
}

/** Composer mic: record → local transcription → text goes into the input (like ChatGPT dictation). */
export function DictationButton({
  enabled,
  onText,
  onError,
}: {
  enabled: boolean;
  onText: (text: string) => void;
  onError: (message: string) => void;
}) {
  const [state, setState] = useState<"idle" | "recording" | "transcribing">("idle");
  const mic = useRef<MicRecorder | null>(null);
  const [getLevel] = useState(() => () => mic.current?.level() ?? 0);

  useEffect(() => () => mic.current?.close(), []);

  const click = async () => {
    if (!enabled) {
      onError("Voice input is off. Set stt.enabled: true in config/settings.yaml (see README).");
      return;
    }
    if (state === "recording") {
      mic.current?.stop();
      return;
    }
    if (state !== "idle") return;
    if (!MicRecorder.supported()) {
      onError("This browser can't record audio.");
      return;
    }
    const m = new MicRecorder();
    mic.current = m;
    try {
      await m.open();
    } catch {
      onError("Microphone access was denied.");
      return;
    }
    setState("recording");
    const audio = await m.record({ maxMs: 120_000 });
    m.close();
    if (!audio) {
      setState("idle");
      return;
    }
    setState("transcribing");
    try {
      const { text } = await transcribe(audio);
      if (text.trim()) onText(text.trim());
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setState("idle");
    }
  };

  return (
    <button
      type="button"
      onClick={click}
      className={`flex h-9 min-w-9 items-center justify-center rounded-full px-2 transition ${
        state === "recording"
          ? "bg-red-500 text-white"
          : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
      }`}
      aria-label={state === "recording" ? "Stop dictation" : "Dictate"}
      title={state === "recording" ? "Stop and transcribe" : "Dictate (local speech recognition)"}
    >
      {state === "recording" ? (
        <LevelBars getLevel={getLevel} />
      ) : state === "transcribing" ? (
        <span className="size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
      ) : (
        <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="2" width="6" height="12" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0M12 17v5" />
        </svg>
      )}
    </button>
  );
}

/** Speaker button under an assistant reply. */
export function ReadAloudButton({ text, language }: { text: string; language?: string | null }) {
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const player = useRef<SpeechPlayer | null>(null);
  useEffect(() => () => player.current?.close(), []);

  const click = async () => {
    if (state === "playing") {
      player.current?.stop();
      return;
    }
    if (state === "loading") return;
    setState("loading");
    try {
      const audio = await synthesize(text, language);
      player.current ??= new SpeechPlayer();
      setState("playing");
      await player.current.play(audio);
    } catch {
      /* TTS unavailable: just reset */
    } finally {
      setState("idle");
    }
  };

  return (
    <button
      type="button"
      onClick={click}
      className="rounded-md p-1.5 text-zinc-400 transition hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
      aria-label={state === "playing" ? "Stop reading" : "Read aloud"}
      title={state === "playing" ? "Stop" : "Read aloud"}
    >
      {state === "loading" ? (
        <span className="block size-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
      ) : state === "playing" ? (
        <svg viewBox="0 0 24 24" className="size-4" fill="currentColor">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 5 6 9H2v6h4l5 4V5Z" />
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07M19.07 4.93a10 10 0 0 1 0 14.14" />
        </svg>
      )}
    </button>
  );
}
