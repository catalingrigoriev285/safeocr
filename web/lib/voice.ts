/**
 * Voice I/O for the chat. Speech-to-text and text-to-speech run on the local backend
 * (/api/transcribe, /api/speak). We deliberately do NOT use the browser's Web Speech API,
 * because Chrome sends that audio to Google's servers.
 */
import { API_URL } from "./api";

export type Transcription = { text: string; language: string | null; duration: number | null; ms: number };

async function errorMessage(res: Response): Promise<string> {
  const body = await res.json().catch(() => ({}));
  return typeof body?.detail === "string" ? body.detail : `HTTP ${res.status}`;
}

export async function transcribe(audio: Blob, language?: string): Promise<Transcription> {
  const ext = audio.type.includes("mp4") ? "mp4" : audio.type.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("file", audio, `speech.${ext}`);
  if (language) form.append("language", language);
  const res = await fetch(`${API_URL}/api/transcribe`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function synthesize(text: string, language?: string | null, signal?: AbortSignal): Promise<ArrayBuffer> {
  const res = await fetch(`${API_URL}/api/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language: language ?? null }),
    signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.arrayBuffer();
}

function rms(analyser: AnalyserNode, buf: Float32Array<ArrayBuffer>): number {
  analyser.getFloatTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  return Math.sqrt(sum / buf.length);
}

function pickMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return candidates.find((t) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) ?? "";
}

type RecordOptions = {
  /** stop automatically after this much silence once speech was heard (voice mode) */
  autoStop?: boolean;
  silenceMs?: number;
  maxMs?: number;
};

/** Microphone capture with a live level meter and simple voice-activity detection. */
export class MicRecorder {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private buf = new Float32Array(1024);
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private raf = 0;
  private smoothed = 0;

  static supported(): boolean {
    return typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined";
  }

  async open(): Promise<void> {
    if (this.stream) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    source.connect(this.analyser);
  }

  /** 0..1, smoothed, for the UI. */
  level(): number {
    if (!this.analyser) return 0;
    const v = Math.min(1, rms(this.analyser, this.buf) * 6);
    this.smoothed = this.smoothed * 0.7 + v * 0.3;
    return this.smoothed;
  }

  /** Record one utterance. Resolves with the audio, or null if cancelled / nothing was said. */
  record(opts: RecordOptions = {}): Promise<Blob | null> {
    const { autoStop = false, silenceMs = 1300, maxMs = 60_000 } = opts;
    if (!this.stream || !this.analyser) return Promise.reject(new Error("microphone not open"));
    const mimeType = pickMimeType();
    const rec = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.recorder = rec;
    this.chunks = [];
    rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);

    return new Promise((resolve) => {
      let heardSpeech = false;
      let speechSince = 0;
      let quietSince = 0;
      let noiseFloor = 0.01;
      const started = performance.now();
      let cancelled = false;

      rec.onstop = () => {
        cancelAnimationFrame(this.raf);
        if (cancelled || (autoStop && !heardSpeech) || !this.chunks.length) return resolve(null);
        resolve(new Blob(this.chunks, { type: rec.mimeType || mimeType || "audio/webm" }));
      };
      (rec as MediaRecorder & { cancel?: () => void }).cancel = () => {
        cancelled = true;
        if (rec.state !== "inactive") rec.stop();
      };

      const tick = () => {
        const now = performance.now();
        const level = rms(this.analyser!, this.buf);
        // learn the background noise during the first half second
        if (now - started < 500) noiseFloor = Math.max(noiseFloor, level);
        const threshold = Math.max(0.02, noiseFloor * 2.2);
        if (level > threshold) {
          speechSince ||= now;
          if (now - speechSince > 150) heardSpeech = true;
          quietSince = 0;
        } else {
          speechSince = 0;
          quietSince ||= now;
        }
        const tooLong = now - started > maxMs;
        const doneTalking = autoStop && heardSpeech && quietSince && now - quietSince > silenceMs;
        if ((tooLong || doneTalking) && rec.state === "recording") {
          rec.stop();
          return;
        }
        this.raf = requestAnimationFrame(tick);
      };
      rec.start(250);
      this.raf = requestAnimationFrame(tick);
    });
  }

  /** Finish the current recording (manual stop, e.g. dictation button). */
  stop(): void {
    if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
  }

  /** Abort the current recording without returning audio. */
  cancel(): void {
    (this.recorder as (MediaRecorder & { cancel?: () => void }) | null)?.cancel?.();
  }

  close(): void {
    this.cancel();
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close().catch(() => {});
    this.stream = null;
    this.ctx = null;
    this.analyser = null;
  }
}

/** Plays synthesized speech through an analyser so the UI can react to the voice. */
export class SpeechPlayer {
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private source: AudioBufferSourceNode | null = null;
  private buf = new Float32Array(1024);
  private smoothed = 0;

  level(): number {
    if (!this.analyser || !this.source) return 0;
    const v = Math.min(1, rms(this.analyser, this.buf) * 5);
    this.smoothed = this.smoothed * 0.6 + v * 0.4;
    return this.smoothed;
  }

  /** Resolves when playback ends or is stopped. */
  async play(audio: ArrayBuffer): Promise<void> {
    this.stop();
    this.ctx ??= new AudioContext();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const decoded = await this.ctx.decodeAudioData(audio.slice(0));
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    const src = this.ctx.createBufferSource();
    src.buffer = decoded;
    src.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    this.source = src;
    return new Promise((resolve) => {
      src.onended = () => {
        if (this.source === src) this.source = null;
        resolve();
      };
      src.start();
    });
  }

  stop(): void {
    try {
      this.source?.stop();
    } catch {
      /* already stopped */
    }
    this.source = null;
  }

  close(): void {
    this.stop();
    this.ctx?.close().catch(() => {});
    this.ctx = null;
  }
}
