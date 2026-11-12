"use client";

/* eslint-disable @next/next/no-img-element -- images come from the local FastAPI backend */
import { useEffect, useRef, useState } from "react";
import { api, ApiError, type Health, type RunResult, type Sample, type Source, type Step } from "@/lib/api";

const TOOLS = [
  { tool: "ocr_document", label: "OCR", hint: "text + word boxes stored locally" },
  { tool: "detect_pii", label: "Detect PII", hint: "CNP regex + checksum, rules" },
  { tool: "redact_image", label: "Redact", hint: "black boxes over detections" },
  { tool: "verify_redaction", label: "Verify", hint: "re-OCR the output" },
  { tool: "announce", label: "Announce", hint: "local TTS (coming soon)" },
];

const TYPE_LABEL: Record<string, string> = {
  CNP: "CNP",
  NAME: "Name",
  ADDRESS: "Address",
  ID_SERIES: "ID series",
  ID_NUMBER: "ID number",
  MRZ: "MRZ",
};

function Badge({ on, children }: { on: boolean; children: React.ReactNode }) {
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
        on ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300" : "bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400"
      }`}
    >
      {children}
    </span>
  );
}

function Counts({ counts }: { counts?: Record<string, number> }) {
  if (!counts || Object.keys(counts).length === 0) return <span className="text-zinc-500">none</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {Object.entries(counts).map(([t, n]) => (
        <span key={t} className="rounded-md bg-zinc-900 px-2 py-0.5 font-mono text-xs text-white dark:bg-zinc-100 dark:text-zinc-900">
          {TYPE_LABEL[t] ?? t} × {n}
        </span>
      ))}
    </div>
  );
}

function StepSummary({ step }: { step: Step }) {
  const r = step.report ?? {};
  switch (step.tool) {
    case "ocr_document":
      return <span>{String(r.word_count)} words · {String(r.line_count)} lines · backend <code>{String(r.ocr_backend)}</code></span>;
    case "detect_pii":
      return <Counts counts={r.counts as Record<string, number>} />;
    case "redact_image":
      return <span>{String(r.boxes_drawn)} boxes drawn over {String(r.detections_redacted)} detections</span>;
    case "verify_redaction":
      return r.passed ? (
        <span className="text-emerald-600 dark:text-emerald-400">Nothing detectable left</span>
      ) : (
        <span className="text-red-600 dark:text-red-400">
          Still detectable: <Counts counts={r.remaining_counts as Record<string, number>} />
        </span>
      );
    case "announce":
      return <span className="italic">“{String(r.text)}”</span>;
    default:
      return null;
  }
}

export default function Anonymizer() {
  const [health, setHealth] = useState<Health | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<Source | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    Promise.all([api.health(), api.samples()])
      .then(([h, s]) => {
        setHealth(h);
        setSamples(s);
        if (s.length) setSelected({ source: "sample", name: s[0].name });
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const pick = (src: Source) => {
    setSelected(src);
    setResult(null);
    setSteps([]);
    setError(null);
  };

  const onUpload = async (file?: File) => {
    if (!file) return;
    try {
      pick(await api.upload(file));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const run = async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    setResult(null);
    setSteps([]);
    try {
      const r = await api.run(selected);
      setResult(r);
      setSteps(r.steps);
    } catch (e) {
      setError((e as Error).message);
      if (e instanceof ApiError) setSteps(e.steps);
    } finally {
      setBusy(false);
    }
  };

  const doneTools = new Map(steps.map((s) => [s.tool, s]));

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">SafeOCR</h1>
          <p className="text-sm text-zinc-500">
            Anonymizes fictive ID documents. The model only sees ids and reports, never the text.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge on={!!health}>{health ? "API online" : "API offline"}</Badge>
          <Badge on={health?.ocr_backend === "tesseract"}>OCR: {health?.ocr_backend ?? "?"}</Badge>
          <Badge on={!!health?.llm_enabled}>Local SLM {health?.llm_enabled ? "on" : "off"}</Badge>
          <Badge on={!!health?.tts_enabled}>Local TTS {health?.tts_enabled ? "on" : "off"}</Badge>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        {/* --- source picker --- */}
        <aside className="space-y-4">
          <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
            <h2 className="mb-3 text-sm font-semibold">Fictive samples</h2>
            {samples.length === 0 ? (
              <p className="text-sm text-zinc-500">
                No samples yet. Run <code className="text-xs">python scripts/generate_samples.py</code>
              </p>
            ) : (
              <div className="grid grid-cols-3 gap-2">
                {samples.map((s) => {
                  const active = selected?.source === "sample" && selected.name === s.name;
                  return (
                    <button
                      key={s.name}
                      onClick={() => pick({ source: "sample", name: s.name })}
                      title={s.variation ? `${s.variation.font}, ${s.variation.rotation_deg}°, noise ${s.variation.noise_sigma}` : s.name}
                      className={`overflow-hidden rounded-md border-2 text-left transition ${
                        active ? "border-indigo-500" : "border-transparent hover:border-zinc-300 dark:hover:border-zinc-700"
                      }`}
                    >
                      <img src={api.imageUrl("sample", s.name)} alt={s.name} className="aspect-[3/2] w-full object-cover" />
                      <span className="block truncate px-1 py-0.5 font-mono text-[10px] text-zinc-500">{s.name}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </section>

          <section
            className="rounded-xl border border-dashed border-zinc-300 p-4 text-center dark:border-zinc-700"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              onUpload(e.dataTransfer.files[0]);
            }}
          >
            <p className="text-sm text-zinc-500">Drop your own fictive image here</p>
            <button onClick={() => fileInput.current?.click()} className="mt-2 text-sm font-medium text-indigo-600 hover:underline">
              or browse…
            </button>
            <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => onUpload(e.target.files?.[0])} />
            <p className="mt-2 text-xs text-zinc-400">Uploads need the Tesseract backend.</p>
          </section>

          <button
            onClick={run}
            disabled={!selected || busy}
            className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? "Running pipeline…" : "Run anonymization"}
          </button>
        </aside>

        {/* --- results --- */}
        <main className="space-y-6">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          <section className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">MCP tool calls</h2>
              {steps.length > 0 && (
                <button onClick={() => setShowJson((v) => !v)} className="text-xs text-indigo-600 hover:underline">
                  {showJson ? "hide" : "show"} raw reports the agent receives
                </button>
              )}
            </div>
            <ol className="space-y-2">
              {TOOLS.map(({ tool, label, hint }, i) => {
                const s = doneTools.get(tool);
                const state = s ? (s.ok ? "ok" : "err") : busy ? "wait" : "idle";
                return (
                  <li key={tool} className="flex items-start gap-3 rounded-lg px-2 py-1.5">
                    <span
                      className={`mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                        state === "ok"
                          ? "bg-emerald-500 text-white"
                          : state === "err"
                            ? "bg-red-500 text-white"
                            : state === "wait"
                              ? "animate-pulse bg-indigo-200 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200"
                              : "bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
                      }`}
                    >
                      {state === "ok" ? "✓" : state === "err" ? "!" : i + 1}
                    </span>
                    <div className="min-w-0 flex-1 text-sm">
                      <div className="flex flex-wrap items-baseline gap-x-2">
                        <code className="font-mono text-[13px] font-medium">{tool}()</code>
                        <span className="text-xs text-zinc-500">{label}: {hint}</span>
                        {s && <span className="ml-auto font-mono text-xs text-zinc-400">{s.ms} ms</span>}
                      </div>
                      {s && (
                        <div className="mt-1 text-zinc-700 dark:text-zinc-300">
                          {s.ok ? <StepSummary step={s} /> : <span className="text-red-600">{s.error}</span>}
                        </div>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
            {showJson && (
              <pre className="mt-3 max-h-80 overflow-auto rounded-lg bg-zinc-950 p-3 text-xs text-zinc-200">
                {JSON.stringify(steps.map(({ tool, report, error }) => ({ tool, report, error })), null, 2)}
              </pre>
            )}
          </section>

          {selected && (
            <section className="grid gap-4 md:grid-cols-2">
              <figure className="rounded-xl border border-zinc-200 p-3 dark:border-zinc-800">
                <figcaption className="mb-2 text-xs font-medium text-zinc-500">Original (stays local)</figcaption>
                <img src={api.imageUrl(selected.source, selected.name)} alt="original" className="w-full rounded-md" />
              </figure>
              <figure className="rounded-xl border border-zinc-200 p-3 dark:border-zinc-800">
                <figcaption className="mb-2 flex items-center justify-between text-xs font-medium text-zinc-500">
                  <span>Redacted</span>
                  {result && (
                    <a href={api.imageUrl("result", result.result_id)} download={`${result.result_id}.png`} className="text-indigo-600 hover:underline">
                      download
                    </a>
                  )}
                </figcaption>
                {result ? (
                  <img src={api.imageUrl("result", result.result_id)} alt="redacted" className="w-full rounded-md" />
                ) : (
                  <div className="flex aspect-[3/2] items-center justify-center rounded-md bg-zinc-100 text-sm text-zinc-400 dark:bg-zinc-900">
                    {busy ? "processing…" : "run the pipeline"}
                  </div>
                )}
              </figure>
            </section>
          )}

          {result && (
            <section className="grid gap-4 md:grid-cols-2">
              <div className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
                <h2 className="mb-3 text-sm font-semibold">Recall (covered / existing)</h2>
                {result.recall ? (
                  <>
                    <p className="mb-3 text-3xl font-semibold tabular-nums">
                      {(result.recall.recall * 100).toFixed(1)}%
                      <span className="ml-2 text-sm font-normal text-zinc-500">
                        {result.recall.covered}/{result.recall.total} entities
                      </span>
                    </p>
                    <table className="w-full text-sm">
                      <tbody>
                        {Object.entries(result.recall.per_type).map(([t, s]) => (
                          <tr key={t} className="border-t border-zinc-100 dark:border-zinc-800">
                            <td className="py-1">{TYPE_LABEL[t] ?? t}</td>
                            <td className="py-1 text-right font-mono tabular-nums">{s.covered}/{s.total}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                ) : (
                  <p className="text-sm text-zinc-500">No ground truth for this image (uploaded), so recall is not available.</p>
                )}
              </div>
              <div className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800">
                <h2 className="mb-3 text-sm font-semibold">Voice announcement</h2>
                <p className="text-sm leading-relaxed">{result.announcement.text}</p>
                <p className="mt-3 text-xs text-zinc-500">
                  {result.announcement.spoken ? "Spoken by the local TTS." : "Text only. A local CPU TTS (Piper, Romanian voice) will be added later."}
                </p>
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
