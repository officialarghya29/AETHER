import type {
  BenchmarkReport,
  DriftEvent,
  DriftStatus,
  Experiment,
  Health,
  MetricsSummary,
  ModelInfo,
  Paginated,
  Prediction,
  ReviewCase,
  RoutingMetrics,
  SystemStatus,
} from '../types';

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '';
const API = `${BASE_URL}/api/v1`;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, {
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
      ...init,
    });
  } catch {
    throw new ApiError(0, 'network_error', 'The API is unreachable. Is the backend running?');
  }
  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = `Request failed with status ${response.status}.`;
    try {
      const body = await response.json();
      code = body?.error?.code ?? body?.detail?.code ?? code;
      message = body?.error?.message ?? (typeof body?.detail === 'string' ? body.detail : message);
    } catch {
      /* non-JSON error body: keep the status-based message */
    }
    throw new ApiError(response.status, code, message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: async (): Promise<Health> => {
    try {
      const response = await fetch(`${BASE_URL}/health`, { headers: { Accept: 'application/json' } });
      if (!response.ok) throw new ApiError(response.status, 'health_failed', 'Health check failed');
      return (await response.json()) as Health;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(0, 'network_error', 'The API is unreachable.');
    }
  },
  systemStatus: () => request<SystemStatus>('/system/status'),
  models: () => request<ModelInfo[]>('/models'),
  metricsSummary: () => request<MetricsSummary>('/metrics/summary'),
  routingMetrics: () => request<RoutingMetrics>('/routing/metrics'),
  routingDecisions: (limit = 25) =>
    request<Paginated<Record<string, unknown>>>(`/routing/decisions?limit=${limit}`),
  driftStatus: () => request<DriftStatus>('/drift/status'),
  driftEvents: (limit = 25) => request<DriftEvent[]>(`/drift/events?limit=${limit}`),
  predictions: (limit = 25) =>
    request<Paginated<Record<string, any>>>(`/predictions?limit=${limit}`),
  diagnostics: (predictionId: string) =>
    request<{ prediction_id: string; action: string; predicted_label: string; diagnostics: any[] }>(
      `/diagnostics/${predictionId}`,
    ),
  reviews: (status = 'pending', limit = 50) =>
    request<Paginated<ReviewCase> & { pending: number }>(
      `/reviews?status=${encodeURIComponent(status)}&limit=${limit}`,
    ),
  submitReview: (reviewId: string, payload: { label: number; reviewer: string; notes?: string }) =>
    request<{ feedback_id: string; review_id: string; label: number; verified: boolean }>(
      `/reviews/${reviewId}/submit`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
    ),
  experiments: (limit = 50) => request<Paginated<Experiment>>(`/experiments?limit=${limit}`),
  report: (name: string) => request<BenchmarkReport | Record<string, unknown>>(`/reports/${name}`),
  predict: async (file: File, errorBudget?: number | null): Promise<Prediction> => {
    const form = new FormData();
    form.append('file', file);
    const query = errorBudget ? `?error_budget=${errorBudget}` : '';
    return request<Prediction>(`/predict${query}`, { method: 'POST', body: form });
  },
  predictBatch: async (files: File[]): Promise<Prediction[]> => {
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    return request<Prediction[]>('/predict/batch', { method: 'POST', body: form });
  },
};
