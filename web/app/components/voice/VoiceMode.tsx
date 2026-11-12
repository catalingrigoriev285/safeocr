"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MicRecorder, SpeechPlayer, synthesize, transcribe } from "@/lib/voice";
import VoiceOrb, { type OrbMode } from "./VoiceOrb";

export type Reply = { text: string; language: string | null } | null;

type Props = {
  /** Send the transcribed text as a chat message and resolve with the assistant's reply. */
  ask: (text: string) => Promise<Reply>;
  onClose: () => void;
  sttEnabled: boolean;
  ttsEnabled: boolean;
  hasAttachment: boolean;
};

const STATUS: Record<OrbMode, string> = {
  connecting: "Starting microphone…",
  listening: "Listening",
  transcribing: "Transcribing…",
  thinking: "Thinking…",
  speaking: "Speaking. Tap the bubble to interrupt",
  paused: "Microphone muted",
  error: "Something went wrong",
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Hands-free conversation: listen → transcribe (local Whisper) → chat → speak (local TTS) → listen…
 * Mounted only while voice mode is open.
 */
export default function VoiceMode({ ask, onClose, sttEnabled, ttsEnabled, hasAttachment }: Props) {
  const [mode, setMode] = useState<OrbMode>("connecting");
  const [you, setYou] = useState("");
  const [reply, setReply] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);

  const mic = useRef<MicRecorder | null>(null);
  const player = useRef<SpeechPlayer | null>(null);
  const active = useRef(true);
  const mutedRef = useRef(false);
  const modeRef = useRef<OrbMode>("connecting");
  // the loop runs once per mount; always call the latest `ask` (it closes over the current chat)
  const askRef = useRef(ask);
  const ttsRef = useRef(ttsEnabled);
  useEffect(() => {
    askRef.current = ask;
    ttsRef.current = ttsEnabled;
  });

  const go = (m: OrbMode) => {
    modeRef.current = m;
    setMode(m);
  };

  const getLevel = useCallback(() => {
    if (modeRef.current === "speaking") return player.current?.level() ?? 0;
    if (modeRef.current === "listening") return mic.current?.level() ?? 0;
    return 0;
  }, []);

  useEffect(() => {
    active.current = true;
    const m = new MicRecorder();
    const p = new SpeechPlayer();
    mic.current = m;
    player.current = p;

    const loop = async () => {
      await Promise.resolve(); // leave the effect before the first state update
      if (!sttEnabled) {
        setError("Voice input is disabled on the backend. Set stt.enabled: true in config/settings.yaml (see README).");
        go("error");
        return;
      }
      if (!MicRecorder.supported()) {
        setError("This browser can't record audio.");
        go("error");
        return;
      }
      try {
        await m.open();
      } catch {
        setError("Microphone access was denied. Allow it in the browser's site settings and try again.");
        go("error");
        return;
      }
      while (active.current) {
        if (mutedRef.current) {
          go("paused");
          await sleep(200);
          continue;
        }
        go("listening");
        const audio = await m.record({ autoStop: true, silenceMs: 1000, maxMs: 30_000 });
        if (!active.current) break;
        if (!audio) continue; // nothing said, or muted
        go("transcribing");
        let text = "";
        try {
          text = (await transcribe(audio)).text.trim();
        } catch (e) {
          setError((e as Error).message);
          go("error");
          await sleep(2500);
          setError(null);
          continue;
        }
        if (!active.current) break;
        if (!text) continue;
        setYou(text);
        setReply("");
        go("thinking");
        const answer = await askRef.current(text);
        if (!active.current) break;
        if (!answer?.text) continue;
        setReply(answer.text);
        if (!ttsRef.current) continue;
        go("speaking");
        try {
          await p.play(await synthesize(answer.text, answer.language));
        } catch (e) {
          setError((e as Error).message);
          await sleep(2000);
          setError(null);
        }
      }
    };
    loop();

    return () => {
      active.current = false;
      m.close();
      p.close();
    };
    // runs once per mount; `ask`/`ttsEnabled` are read through refs, sttEnabled only matters at start
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const toggleMute = () => {
    const next = !mutedRef.current;
    mutedRef.current = next;
    setMuted(next);
    if (next) mic.current?.cancel();
  };

  const onOrb = () => {
    if (modeRef.current === "speaking") player.current?.stop(); // interrupt
    else if (modeRef.current === "listening") mic.current?.stop(); // "I'm done talking"
  };

  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center bg-[var(--background)]/95 backdrop-blur-sm">
      <div className="flex w-full items-center justify-between px-4 py-3 text-sm text-zinc-500">
        <span className="font-medium text-zinc-700 dark:text-zinc-300">Voice mode</span>
        <span>Local speech recognition &amp; voice</span>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center gap-8 px-6 text-center">
        <VoiceOrb mode={mode} getLevel={getLevel} onClick={onOrb} label={STATUS[mode]} />
        <div className="min-h-[5.5rem] max-w-xl space-y-2">
          <p className={`text-sm font-medium ${mode === "error" ? "text-red-500" : "text-zinc-500"}`}>
            {error ?? STATUS[mode]}
          </p>
          {you && <p className="text-base text-zinc-700 dark:text-zinc-300">“{you}”</p>}
          {reply && mode !== "thinking" && (
            <p className="line-clamp-3 text-sm text-zinc-500">{reply}</p>
          )}
          {hasAttachment && !you && (
            <p className="text-xs text-zinc-400">The attached image will be sent with your first sentence.</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-6 pb-10">
        <button
          onClick={toggleMute}
          className={`flex size-14 items-center justify-center rounded-full transition ${
            muted ? "bg-red-500 text-white" : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
          }`}
          aria-label={muted ? "Unmute microphone" : "Mute microphone"}
          title={muted ? "Unmute" : "Mute"}
        >
          <svg viewBox="0 0 24 24" className="size-6" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0M12 17v5" />
            {muted && <path d="M3 3l18 18" />}
          </svg>
        </button>
        <button
          onClick={onClose}
          className="flex size-14 items-center justify-center rounded-full bg-zinc-900 text-white transition hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
          aria-label="End voice mode"
          title="End voice mode (Esc)"
        >
          <svg viewBox="0 0 24 24" className="size-6" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
