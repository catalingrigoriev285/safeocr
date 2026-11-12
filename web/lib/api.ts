export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type PiiType = "CNP" | "NAME" | "ADDRESS" | "ID_SERIES" | "ID_NUMBER" | "MRZ";

export type Sample = {
  name: string;
  has_ground_truth: boolean;
  variation: { font: string; rotation_deg: number; noise_sigma: number; blur_radius: number } | null;
};

export type Health = { status: string; ocr_backend: string; llm_enabled: boolean; tts_enabled: boolean };

export type Step = {
  tool: string;
  ok: boolean;
  ms: number;
  report?: Record<string, unknown>;
  error?: string;
};

export type Recall = {
  covered: number;
  total: number;
  recall: number;
  per_type: Record<string, { covered: number; total: number; recall: number }>;
};

export type RunResult = {
  document_id: string;
  result_id: string;
  passed: boolean;
  steps: Step[];
  announcement: { text: string; spoken: boolean };
  recall: Recall | null;
};

export type Source = { source: "sample" | "upload"; name: string };

export class ApiError extends Error {
  constructor(message: string, public steps: Step[] = []) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(`Backend unreachable at ${API_URL}. Is uvicorn running?`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body?.detail;
    if (detail && typeof detail === "object") throw new ApiError(detail.message, detail.steps ?? []);
    throw new ApiError(typeof detail === "string" ? detail : `HTTP ${res.status}`);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  samples: () => request<Sample[]>("/api/samples"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Source>("/api/documents", { method: "POST", body: form });
  },
  run: (src: Source) =>
    request<RunResult>("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(src),
    }),
  imageUrl: (kind: "sample" | "upload" | "result", name: string) =>
    `${API_URL}/api/images/${kind}/${encodeURIComponent(name)}`,
};
