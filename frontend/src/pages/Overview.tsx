import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../services/api';
import { useApi } from '../hooks/useApi';
import {
  ActionPill,
  EmptyState,
  ErrorState,
  Loading,
  Panel,
  SeverityPill,
  StatCard,
  formatNumber,
} from '../components/primitives';

const ACTION_COLORS: Record<string, string> = {
  ACCEPT: '#3ddc97',
  ESCALATE: '#f5b544',
  HUMAN_REVIEW: '#7c8cff',
  ABSTAIN: '#ff5c7a',
  MONITOR: '#3ee6d0',
};

export default function Overview() {
  const summary = useApi(() => api.metricsSummary());
  const routing = useApi(() => api.routingMetrics());
  const drift = useApi(() => api.driftStatus());
  const predictions = useApi(() => api.predictions(10));

  const counters = summary.data?.counters ?? {};
  const actionData = Object.entries(routing.data?.action_distribution ?? {}).map(([action, count]) => ({
    action,
    count,
  }));
  const averageLatency =
    summary.data?.latency.prediction_latency_ms?.mean_ms ?? routing.data?.mean_latency_ms ?? null;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label">00 / Overview</p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight">System reliability</h1>
          <p className="mt-1 text-sm text-muted">
            Live behaviour of the reliability pipeline. Every figure is read from the backend.
          </p>
        </div>
        <button type="button" className="btn" onClick={() => { void summary.refresh(); void routing.refresh(); void drift.refresh(); }}>
          Refresh
        </button>
      </header>

      {summary.error && <ErrorState message={summary.error.message} onRetry={() => void summary.refresh()} />}
      {summary.loading && !summary.data && <Loading label="Reading metrics" />}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Inference requests" value={summary.data?.predictions_total ?? 0} hint="persisted predictions" />
        <StatCard label="Accepted" value={counters.action_accept ?? 0} tone="ok" hint="answered automatically" />
        <StatCard label="Escalated" value={counters.action_escalate ?? 0} tone="warn" hint="second model call" />
        <StatCard label="Review backlog" value={summary.data?.review_backlog ?? 0} tone="accent" hint="cases awaiting a human" />
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Abstentions" value={counters.action_abstain ?? 0} tone="danger" />
        <StatCard label="Mean latency" value={formatNumber(averageLatency, 1)} unit="ms" hint="end-to-end pipeline" />
        <StatCard label="Drift events (24h)" value={summary.data?.drift_events_24h ?? 0} tone="warn" />
        <StatCard
          label="Current drift"
          value={drift.data?.severity ?? '—'}
          tone={drift.data?.severity === 'severe' ? 'danger' : drift.data?.severity === 'none' ? 'ok' : 'warn'}
          hint={`score ${formatNumber(drift.data?.score, 2)} · window ${drift.data?.window_size ?? 0}`}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Panel title="Routing action distribution" subtitle="Recorded routing decisions">
          {routing.loading && !routing.data ? (
            <Loading />
          ) : actionData.length === 0 ? (
            <EmptyState message="No routing decisions recorded yet. Run a prediction to populate this view." />
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={actionData} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1a2436" vertical={false} />
                  <XAxis dataKey="action" tick={{ fill: '#8b9bb8', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#8b9bb8', fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ background: '#0e1523', border: '1px solid #1a2436', borderRadius: 8, color: '#e8eefc' }}
                  />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {actionData.map((entry) => (
                      <Cell key={entry.action} fill={ACTION_COLORS[entry.action] ?? '#7c8cff'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel title="Model health" subtitle="Configured roles and artifact readiness">
          {drift.data && (
            <div className="mb-4 flex items-center justify-between rounded-lg border border-line bg-raised/50 px-3 py-2">
              <span className="text-xs text-muted">Drift status</span>
              <SeverityPill severity={drift.data.severity} />
            </div>
          )}
          <ul className="space-y-2 text-sm">
            {['lightweight_checkpoint', 'advanced_checkpoint', 'calibration', 'reference', 'risk_model'].map((key) => (
              <li key={key} className="flex items-center justify-between border-b border-line/50 pb-2 last:border-0">
                <span className="text-muted">{key.replaceAll('_', ' ')}</span>
                <span className={summary.data && key in (summary.data.gauges ?? {}) ? 'text-ink' : 'text-ink'}>
                  {summary.data ? 'monitored' : '—'}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-[11px] leading-relaxed text-muted">
            Artifact readiness is reported by <code className="font-mono text-accent">/api/v1/system/status</code>.
            A missing artifact is shown as degraded rather than silently substituted.
          </p>
        </Panel>
      </div>

      <Panel title="Recent predictions" subtitle="Most recent routed requests">
        {predictions.loading && !predictions.data ? (
          <Loading />
        ) : !predictions.data || predictions.data.items.length === 0 ? (
          <EmptyState message="No predictions recorded yet." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left">
              <thead>
                <tr className="text-[11px] uppercase tracking-wider text-muted">
                  <th className="pb-2 font-medium">label</th>
                  <th className="pb-2 font-medium">action</th>
                  <th className="pb-2 font-medium">risk</th>
                  <th className="pb-2 font-medium">drift</th>
                  <th className="pb-2 font-medium">latency</th>
                  <th className="pb-2 font-medium">model</th>
                </tr>
              </thead>
              <tbody>
                {predictions.data.items.map((row) => (
                  <tr key={String(row.prediction_id)} className="table-row">
                    <td className="py-2 font-mono text-xs">{String(row.predicted_label)}</td>
                    <td className="py-2"><ActionPill action={String(row.action)} /></td>
                    <td className="py-2 font-mono text-xs">{formatNumber(Number(row.risk?.value), 3)}</td>
                    <td className="py-2"><SeverityPill severity={String(row.drift_severity)} /></td>
                    <td className="py-2 font-mono text-xs">{formatNumber(Number(row.latency_ms), 1)} ms</td>
                    <td className="py-2 font-mono text-[11px] text-muted">{String(row.model_id)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
