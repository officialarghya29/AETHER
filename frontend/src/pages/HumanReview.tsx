import { useState } from 'react';
import { ApiError, api } from '../services/api';
import { useApi } from '../hooks/useApi';
import {
  EmptyState,
  ErrorState,
  Loading,
  Panel,
  SeverityPill,
  StatCard,
  formatNumber,
  formatPercent,
} from '../components/primitives';

const CLASS_LABELS = [
  'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
];

export default function HumanReview() {
  const queue = useApi(() => api.reviews('pending', 50));
  const [reviewer, setReviewer] = useState('');
  const [notes, setNotes] = useState('');
  const [labels, setLabels] = useState<Record<string, number>>({});
  const [feedback, setFeedback] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const submit = async (reviewId: string) => {
    const label = labels[reviewId];
    if (label === undefined) {
      setError('Select the verified class before submitting.');
      return;
    }
    if (!reviewer.trim()) {
      setError('Reviewer identity is required — feedback is attributable.');
      return;
    }
    setBusy(reviewId);
    setError(null);
    try {
      await api.submitReview(reviewId, { label, reviewer: reviewer.trim(), notes: notes || undefined });
      setFeedback((previous) => ({ ...previous, [reviewId]: 'verified label recorded' }));
      await queue.refresh();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Submission failed.');
    } finally {
      setBusy(null);
    }
  };

  const cases = queue.data?.items ?? [];

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label">06 / Human review</p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight">Verified feedback queue</h1>
          <p className="mt-1 text-sm text-muted">
            Unverified feedback never enters training data. A verified label is required, and the
            reviewer identity is recorded.
          </p>
        </div>
        <button type="button" className="btn" onClick={() => void queue.refresh()}>
          Refresh
        </button>
      </header>

      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard label="Pending cases" value={queue.data?.pending ?? 0} tone={cases.length > 0 ? 'warn' : 'ok'} />
        <StatCard label="Loaded" value={cases.length} hint="page size 50" />
        <StatCard label="Queue health" value={cases.length === 0 ? 'clear' : 'backlog'} tone={cases.length === 0 ? 'ok' : 'warn'} />
      </div>

      <Panel title="Reviewer identity" subtitle="Required for every verified label; stored in the audit trail">
        <label className="block text-xs text-muted" htmlFor="reviewer">
          reviewer
        </label>
        <input
          id="reviewer"
          value={reviewer}
          onChange={(event) => setReviewer(event.target.value)}
          placeholder="e.g. a.bose"
          className="mt-1 w-full rounded-lg px-3 py-2 text-sm"
          maxLength={128}
        />
        <label className="mt-3 block text-xs text-muted" htmlFor="notes">
          notes (optional)
        </label>
        <textarea
          id="notes"
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          rows={2}
          className="mt-1 w-full rounded-lg px-3 py-2 text-sm"
          maxLength={2000}
        />
      </Panel>

      {error && <ErrorState message={error} />}
      {queue.error && <ErrorState message={queue.error.message} onRetry={() => void queue.refresh()} />}

      <Panel title="Pending cases" subtitle="Predictions the router could not answer reliably">
        {queue.loading && !queue.data ? (
          <Loading label="Loading queue" />
        ) : cases.length === 0 ? (
          <EmptyState message="No pending review cases." />
        ) : (
          <ul className="space-y-3">
            {cases.map((item) => (
              <li key={item.review_id} className="rounded-lg border border-line bg-raised/40 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-accent">{item.review_id.slice(0, 8)}</span>
                      <SeverityPill severity={item.drift_severity} />
                      {item.diagnostics.map((code) => (
                        <span key={code} className="tag border-line text-muted">{code}</span>
                      ))}
                    </div>
                    <p className="mt-2 text-sm">
                      predicted <span className="font-medium text-ink">{item.predicted_label}</span>{' '}
                      <span className="text-muted">({formatPercent(item.confidence)} confidence)</span>
                    </p>
                    <p className="mt-1 font-mono text-[11px] text-muted">
                      entropy {formatNumber(item.uncertainty.predictive_entropy, 4)} · margin{' '}
                      {formatNumber(item.uncertainty.prediction_margin, 4)}
                    </p>
                  </div>
                  <span className="font-mono text-[10px] text-muted">
                    {new Date(item.created_at).toLocaleString()}
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <select
                    aria-label={`verified class for ${item.review_id}`}
                    className="rounded-lg px-2 py-1.5 text-xs"
                    value={labels[item.review_id] ?? ''}
                    onChange={(event) =>
                      setLabels((previous) => ({ ...previous, [item.review_id]: Number(event.target.value) }))
                    }
                  >
                    <option value="">verified class…</option>
                    {CLASS_LABELS.map((label, index) => (
                      <option key={label} value={index}>
                        {label}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    className="btn btn-primary text-xs"
                    disabled={busy === item.review_id}
                    onClick={() => void submit(item.review_id)}
                  >
                    {busy === item.review_id ? 'submitting…' : 'submit verified label'}
                  </button>
                  {feedback[item.review_id] && (
                    <span className="text-[11px] text-ok">{feedback[item.review_id]}</span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
