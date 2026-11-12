"use client";

import { useEffect, useRef } from "react";

export type OrbMode = "connecting" | "listening" | "transcribing" | "thinking" | "speaking" | "paused" | "error";

const GRADIENTS: Record<OrbMode, [string, string, string]> = {
  connecting: ["#cbd5e1", "#94a3b8", "#64748b"],
  listening: ["#bae6fd", "#38bdf8", "#4f46e5"],
  transcribing: ["#e0e7ff", "#818cf8", "#4338ca"],
  thinking: ["#e4e4e7", "#a1a1aa", "#52525b"],
  speaking: ["#ffffff", "#7dd3fc", "#2563eb"],
  paused: ["#52525b", "#3f3f46", "#27272a"],
  error: ["#fecaca", "#f87171", "#b91c1c"],
};

/**
 * ChatGPT-style voice bubble: three soft blobs whose size, shape and jitter follow the
 * audio level (microphone while listening, reply audio while speaking).
 */
export default function VoiceOrb({
  mode,
  getLevel,
  size = 220,
  onClick,
  label,
}: {
  mode: OrbMode;
  getLevel: () => number;
  size?: number;
  onClick?: () => void;
  label?: string;
}) {
  const glow = useRef<HTMLDivElement>(null);
  const mid = useRef<HTMLDivElement>(null);
  const core = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const layers = [glow.current, mid.current, core.current];
    const reactive = mode === "listening" || mode === "speaking";
    const t0 = performance.now();
    let level = 0;
    let raf = 0;
    const loop = () => {
      const t = (performance.now() - t0) / 1000;
      let target = 0;
      if (reactive) target = getLevel();
      else if (mode === "thinking" || mode === "transcribing") target = 0.1 + 0.07 * Math.sin(t * 3.2);
      else if (mode === "connecting") target = 0.05 + 0.04 * Math.sin(t * 2);
      level += (target - level) * (target > level ? 0.35 : 0.12); // fast attack, slow release

      layers.forEach((el, i) => {
        if (!el) return;
        const amp = 3 + level * (18 + i * 6); // shape wobble, in %
        const r = (phase: number) => 50 + amp * Math.sin(t * (1.3 + i * 0.6) + phase);
        const radius = `${r(0)}% ${100 - r(0)}% ${r(2)}% ${100 - r(2)}% / ${r(1)}% ${r(3)}% ${100 - r(3)}% ${100 - r(1)}%`;
        const scale = (i === 0 ? 1.08 : 1) + level * (0.28 + i * 0.1);
        const shake = level * level * 9; // jitter grows quickly with loud audio
        const dx = Math.sin(t * 31 + i * 2) * shake;
        const dy = Math.cos(t * 27 + i * 3) * shake;
        const spin = t * [12, -18, 24][i];
        el.style.setProperty("border-radius", radius);
        el.style.setProperty("transform", `translate(${dx}px, ${dy}px) rotate(${spin}deg) scale(${scale})`);
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [mode, getLevel]);

  const [c1, c2, c3] = GRADIENTS[mode];
  const layer = "absolute inset-0 transition-[background] duration-700";
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label ?? mode}
      className="relative outline-none focus-visible:ring-4 focus-visible:ring-sky-400/40"
      style={{ width: size, height: size, borderRadius: "50%" }}
    >
      <div
        ref={glow}
        className={`${layer} opacity-50 blur-2xl`}
        style={{ background: `radial-gradient(circle at 50% 50%, ${c2}, transparent 70%)` }}
      />
      <div
        ref={mid}
        className={`${layer} opacity-80 blur-[2px]`}
        style={{ background: `radial-gradient(circle at 35% 30%, ${c1}, ${c2} 45%, ${c3})` }}
      />
      <div
        ref={core}
        className={`${layer} inset-[12%] opacity-90`}
        style={{ background: `radial-gradient(circle at 30% 25%, #ffffffcc, ${c2} 40%, ${c3} 95%)` }}
      />
    </button>
  );
}
