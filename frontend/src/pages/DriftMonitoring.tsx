import { api } from '../services/api';
import { useApi } from '../hooks/useApi';
import {
  EmptyState,
  ErrorState,
  Loading,
  Panel,
  SeverityPill,
  StatCard,
  formatNumber,
} from '../components/primitives';

interface DriftWindowRow {
  condition: string;
  family: string;
  severity: number;
  windows: number;
  mean_score: number;
  max_score: number;
  detection_rate: number | null;
  severity_counts: Record<string, number>;
}

export default function DriftMonitoring() {
  const status = useApi(() => api.driftStatus());
  const events = useApi(() => api.driftEvents(20));
  const report = useApi(() => api.report('benchmarks'));

  const drift = (report.data as Record<string, any> | null)?.drift as
    | { windows: DriftWindowRow[]; clean_false_positive_rate: number | null; mean_detection_rate_by_severity: Record<string, number> }
    | undefined;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label">03 / Drift monitoring</p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight">Distribution shift</h1>
          <p className="mt-1 text-sm text-muted">
            Input and prediction-behaviour shift are reported separately. Shift is not accuracy loss.
          </p>
        </div>
        <button type="button" className="btn" onClick={() => { void status.refresh(); void events.refresh(); }}>
          Refresh
        </button>
      </header>

      {status.error && <ErrorState message={status.error.message} onRetry={() => void status.refresh()} />}
      {status.loading && !status.data && <Loading label="Reading drift status" />}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Current severity"
          value={status.data?.severity ?? '—'}
          tone={status.data?.severity === 'severe' ? 'danger' : status.data?.severity === 'none' ? 'ok' : 'warn'}
        />
        <StatCard label="Composite score" value={formatNumber(status.data?.score, 2)} hint="max adjusted −log10 p" />
        <StatCard label="Window" value={status.data?.window_size ?? 0} hint="observations in the live window" />
        <StatCard
          label="Clean false-positive rate"
          value={drift?.clean_false_positive_rate !== undefined && drift?.clean_false_positive_rate !== null
            ? `${(drift.clean_false_positive_rate * 100).toFixed(1)}%`
            : '—'}
          tone="accent"
          hint="development windows, no corruption"
        />
      </div>

      <Panel
        title="Active assessment"
        subtitle={status.data ? `status: ${status.data.status}` : 'no assessment'}
        actions={status.data ? <SeverityPill severity={status.data.severity} /> : undefined}
      >
        {!status.data ? (
          <EmptyState message="No drift assessment has been produced. Send predictions to fill the window." />
        ) : (
          <div className="space-y-4">
            <p className="text-xs leading-relaxed text-muted">
              {status.data.warnings.length > 0
                ? status.data.warnings.join(' · ')
                : 'No warnings. Detectors ran on the current window.'}
            </p>
            {status.data.signals.length === 0 ? (
              <EmptyState message="No detector exceeded its reporting threshold." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] text-left">
                  <thead>
                    <tr className="text-[11px] uppercase tracking-wider text-muted">
                      <th className="pb-2 font-medium">signal</th>
                      <th className="pb-2 font-medium">statistic</th>
                      <th className="pb-2 font-medium">p-value</th>
                      <th className="pb-2 font-medium">score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status.data.signals.map((signal, index) => (
                      <tr key={`${String(signal.test ?? signal.feature)}-${index}`} className="table-row">
                        <td className="py-2 font-mono text-xs text-accent">
                          {String(signal.test ?? signal.feature)}
                        </td>
                        <td className="py-2 font-mono text-xs">{formatNumber(Number(signal.statistic), 4)}</td>
                        <td className="py-2 font-mono text-xs">
                          {signal.p_value === undefined || signal.p_value === null
                            ? '—'
                            : Number(signal.p_value).toExponential(2)}
                        </td>
                        <td className="py-2 font-mono text-xs">{formatNumber(Number(signal.score), 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Panel>

      <Panel title="Corruption study" subtitle="Detection behaviour per corruption and severity">
        {report.error ? (
          <ErrorState message={`${report.error.message} — run experiments/run_benchmarks.py.`} />
        ) : !drift ? (
          <Loading />
        ) : (
          <>
            <div className="mb-4 grid gap-2 sm:grid-cols-3">
              {Object.entries(drift.mean_detection_rate_by_severity).map(([severity, rate]) => (
                <div key={severity} className="rounded-lg border border-line bg-raised/40 p-3">
                  <p className="label">severity {severity}</p>
                  <p className="mt-1 font-mono text-lg">{(rate * 100).toFixed(1)}%</p>
                  <p className="text-[10px] text-muted">detected as moderate or severe</p>
                </div>
              ))}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-left">
                <thead>
                  <tr className="text-[11px] uppercase tracking-wider text-muted">
                    <th className="pb-2 font-medium">condition</th>
                    <th className="pb-2 font-medium">windows</th>
                    <th className="pb-2 font-medium">mean score</th>
                    <th className="pb-2 font-medium">max score</th>
                    <th className="pb-2 font-medium">detection</th>
                    <th className="pb-2 font-medium">severity mix</th>
                  </tr>
                </thead>
                <tbody>
                  {drift.windows.map((row) => (
                    <tr key={row.condition} className="table-row">
                      <td className="py-2 font-mono text-xs">{row.condition}</td>
                      <td className="py-2 font-mono text-xs">{row.windows}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.mean_score, 2)}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.max_score, 2)}</td>
                      <td className="py-2 font-mono text-xs">
                        {row.detection_rate === null ? '—' : `${(row.detection_rate * 100).toFixed(0)}%`}
                      </td>
                      <td className="py-2 font-mono text-[11px] text-muted">
                        {Object.entries(row.severity_counts)
                          .filter(([, count]) => count > 0)
                          .map(([level, count]) => `${level}:${count}`)
                          .join(' ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Panel>

      <Panel title="Drift events" subtitle="Persisted moderate and severe assessments">
        {events.loading && !events.data ? (
          <Loading />
        ) : !events.data || events.data.length === 0 ? (
          <EmptyState message="No drift events recorded." />
        ) : (
          <ul className="space-y-2">
            {events.data.map((event) => (
              <li key={event.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-line bg-raised/40 px-3 py-2">
                <div className="flex items-center gap-3">
                  <SeverityPill severity={event.severity} />
                  <span className="font-mono text-xs text-muted">score {formatNumber(event.score, 2)}</span>
                  <span className="font-mono text-xs text-muted">window {event.window_size}</span>
                </div>
                <span className="font-mono text-[11px] text-muted">{new Date(event.created_at).toLocaleString()}</span>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
