import { useEffect, useState } from 'react';
import { api } from '../services/api';
import { useApi } from '../hooks/useApi';
import {
  ActionPill,
  EmptyState,
  ErrorState,
  Loading,
  Panel,
  SeverityPill,
  formatNumber,
} from '../components/primitives';

interface DiagnosticRow {
  code: string;
  severity: string;
  evidence: Record<string, unknown>;
  explanation: string;
  recommended_action: string;
}

export default function FailureDiagnosis() {
  const predictions = useApi(() => api.predictions(30));
  const [selected, setSelected] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<DiagnosticRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    api
      .diagnostics(selected)
      .then((payload) => setDiagnostics(payload.diagnostics as DiagnosticRow[]))
      .catch((caught: Error) => setError(caught.message))
      .finally(() => setLoading(false));
  }, [selected]);

  return (
    <div className="space-y-6">
      <header>
        <p className="label">05 / Failure diagnosis</p>
        <h1 className="mt-1 text-2xl font-medium tracking-tight">Evidence-backed findings</h1>
        <p className="mt-1 text-sm text-muted">
          Diagnostics describe observed evidence, not proven causes. Select a prediction to inspect it.
        </p>
      </header>

      <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Predictions" subtitle="Choose a case">
          {predictions.loading && !predictions.data ? (
            <Loading />
          ) : !predictions.data || predictions.data.items.length === 0 ? (
            <EmptyState message="No predictions recorded yet." />
          ) : (
            <ul className="max-h-[520px] space-y-1.5 overflow-y-auto pr-1">
              {predictions.data.items.map((row) => (
                <li key={String(row.prediction_id)}>
                  <button
                    type="button"
                    onClick={() => setSelected(String(row.prediction_id))}
                    className={`flex w-full items-center justify-between gap-3 rounded-lg border px-3 py-2 text-left text-xs transition ${
                      selected === row.prediction_id
                        ? 'border-accent/50 bg-accent/10'
                        : 'border-line bg-raised/30 hover:border-accent/30'
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <ActionPill action={String(row.action)} />
                      <span className="font-mono">{String(row.predicted_label)}</span>
                    </span>
                    <span className="font-mono text-[10px] text-muted">
                      {formatNumber(Number(row.latency_ms), 0)} ms
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Diagnostics" subtitle={selected ? `prediction ${selected}` : 'no prediction selected'}>
          {!selected ? (
            <EmptyState message="Select a prediction from the list." />
          ) : loading ? (
            <Loading label="Loading diagnostics" />
          ) : error ? (
            <ErrorState message={error} />
          ) : !diagnostics || diagnostics.length === 0 ? (
            <EmptyState message="No diagnostic rule fired for this prediction." />
          ) : (
            <ul className="space-y-3">
              {diagnostics.map((diagnostic, index) => (
                <li key={`${diagnostic.code}-${index}`} className="rounded-lg border border-line bg-raised/40 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="font-mono text-xs text-accent">{diagnostic.code}</p>
                      <p className="mt-1 text-sm leading-relaxed text-ink">{diagnostic.explanation}</p>
                    </div>
                    <SeverityPill severity={diagnostic.severity} />
                  </div>
                  <div className="mt-3 rounded-md border border-line/70 bg-void/40 p-2.5">
                    <p className="label mb-1">evidence</p>
                    <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted">
                      {JSON.stringify(diagnostic.evidence, null, 2)}
                    </pre>
                  </div>
                  <p className="mt-2 text-xs text-ink">
                    <span className="text-muted">recommended: </span>
                    {diagnostic.recommended_action}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
