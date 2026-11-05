export type Action = 'ACCEPT' | 'ESCALATE' | 'HUMAN_REVIEW' | 'ABSTAIN' | 'MONITOR' | 'REJECT';

export type Severity = 'none' | 'low' | 'moderate' | 'severe';

export interface Health {
  status: 'ok' | 'degraded' | 'error';
  version: string;
  environment: string;
  components: Record<string, string>;
  timestamp: string;
}

export interface ModelInfo {
  role: string;
  architecture: string;
  version: string;
  num_classes: number;
  checkpoint_present: boolean;
  loaded: boolean;
  model_id?: string | null;
  parameters?: number | null;
  device?: string | null;
  val_accuracy?: number | null;
}

export interface SystemStatus {
  application: string;
  version: string;
  environment: string;
  device: string;
  torch_version: string;
  artifacts: Record<string, boolean>;
  routing_policy: string;
  drift_window_size: number;
  uptime_seconds: number;
  models: ModelInfo[];
}

export interface Diagnostic {
  code: string;
  severity: string;
  title?: string;
  evidence: Record<string, unknown>;
  explanation: string;
  recommended_action: string;
}

export interface Decision {
  action: string;
  policy: string;
  risk: number;
  reason_codes: string[];
  explanation: string;
  test_time_augmentation?: boolean;
  utilities?: Array<{
    action: string;
    utility: number;
    expected_risk: number;
    added_cost: number;
    feasible: boolean;
  }>;
}

export interface RiskBudget {
  requested_alpha: number;
  alpha: number;
  threshold: number;
  tier: string;
  confidence: number;
  meets_threshold: boolean;
  generated_at: string;
}

export interface Prediction {
  prediction_id: string;
  action: Action;
  predicted_index: number;
  predicted_label: string;
  probabilities: number[];
  model_id: string;
  model_version: string;
  uncertainty: Record<string, number>;
  calibrated_uncertainty: Record<string, number>;
  temperature: number;
  risk: { value: number; source: string; lightweight: number; advanced: number | null };
  drift: {
    severity: Severity;
    score: number;
    window_size: number;
    status: string;
    signals?: Array<Record<string, unknown>>;
    warnings?: string[];
  };
  decision: Decision;
  diagnostics: Diagnostic[];
  quality: Record<string, number | boolean>;
  lightweight: Record<string, unknown>;
  advanced: Record<string, unknown> | null;
  timings: Record<string, number>;
  input_size: number;
  created_at: string;
  latency_ms: number;
  review_id?: string | null;
  risk_budget?: RiskBudget | null;
}

export interface DriftStatus {
  severity: Severity;
  score: number;
  window_size: number;
  status: string;
  signals: Array<Record<string, unknown>>;
  warnings: string[];
  input_shift: Record<string, unknown>;
  prediction_shift: Record<string, unknown>;
  interpretation: string;
}

export interface DriftEvent {
  id: string;
  severity: Severity;
  score: number;
  window_size: number;
  status: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface RoutingMetrics {
  total: number;
  action_distribution: Record<string, number>;
  mean_risk: number;
  mean_latency_ms: number;
  escalation_rate: number;
  abstention_rate: number;
  review_rate: number;
  budgeted_total: number;
  budget_misses: number;
  budget_miss_rate: number;
  tta_calls: number;
}

export interface MetricsSummary {
  uptime_seconds: number;
  counters: Record<string, number>;
  gauges: Record<string, number>;
  latency: Record<string, { count: number; mean_ms: number; p50_ms: number; p95_ms: number; max_ms: number }>;
  review_backlog: number;
  drift_events_24h: number;
  predictions_total: number;
}

export interface ReviewCase {
  review_id: string;
  prediction_id: string;
  status: string;
  predicted_index: number;
  predicted_label: string;
  confidence: number;
  uncertainty: Record<string, number>;
  drift_severity: Severity;
  diagnostics: string[];
  created_at: string;
  resolved_at?: string | null;
  verified_label?: number | null;
  reviewer?: string | null;
  notes?: string | null;
  history?: Array<Record<string, unknown>>;
}

export interface Paginated<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Experiment {
  experiment_id: string;
  name: string;
  status: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  dataset_id?: string | null;
  seed?: number | null;
  created_at: string;
}

export type BenchmarkReport = {
  generated_at: string;
  environment: Record<string, string | number>;
  dataset: Record<string, unknown>;
  lightweight: Record<string, any>;
  advanced: Record<string, any>;
  drift: Record<string, any>;
  routing: Record<string, any>;
  robustness?: Record<string, any>;
};
