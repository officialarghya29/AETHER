import { api } from '../services/api';
import { useApi } from '../hooks/useApi';
import {
  ActionPill,
  EmptyState,
  ErrorState,
  Loading,
  Panel,
  StatCard,
  formatNumber,
  formatPercent,
} from '../components/primitives';

interface PolicyRow {
  policy: string;
  coverage: number;
  accuracy_on_answered: number;
  escalation_rate: number;
  review_rate: number;
  abstention_rate: number;
  mean_latency_ms: number;
  expected_cost_per_sample: number;
}

export default function RoutingIntelligence() {
  const metrics = useApi(() => api.routingMetrics());
  const decisions = useApi(() => api.routingDecisions(20));
  const report = useApi(() => api.report('benchmarks'));

  const routing = (report.data as Record<string, any> | null)?.routing as
    | { policies: PolicyRow[]; ablations: Record<string, PolicyRow>; records: number; interpretation: string }
    | undefined;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label">04 / Routing intelligence</p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight">Cost-aware decision making</h1>
          <p className="mt-1 text-sm text-muted">
            Every action is a priced decision over uncertainty, shift, disagreement and latency budget.
          </p>
        </div>
        <button type="button" className="btn" onClick={() => { void metrics.refresh(); void decisions.refresh(); }}>
          Refresh
        </button>
      </header>

      {metrics.error && <ErrorState message={metrics.error.message} onRetry={() => void metrics.refresh()} />}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Decisions" value={metrics.data?.total ?? 0} />
        <StatCard label="Escalation rate" value={formatPercent(metrics.data?.escalation_rate)} tone="warn" />
        <StatCard label="Review rate" value={formatPercent(metrics.data?.review_rate)} tone="accent" />
        <StatCard label="Mean risk" value={formatNumber(metrics.data?.mean_risk, 3)} hint={`latency ${formatNumber(metrics.data?.mean_latency_ms, 1)} ms`} />
      </div>

      <Panel title="Offline policy comparison" subtitle="Replayed over routing-development evidence (clean + corrupted)">
        {report.error ? (
          <ErrorState message={`${report.error.message} — run experiments/run_benchmarks.py.`} />
        ) : !routing ? (
          <Loading />
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[820px] text-left">
                <thead>
                  <tr className="text-[11px] uppercase tracking-wider text-muted">
                    <th className="pb-2 font-medium">policy</th>
                    <th className="pb-2 font-medium">coverage</th>
                    <th className="pb-2 font-medium">acc on answered</th>
                    <th className="pb-2 font-medium">escalation</th>
                    <th className="pb-2 font-medium">review</th>
                    <th className="pb-2 font-medium">abstain</th>
                    <th className="pb-2 font-medium">latency</th>
                    <th className="pb-2 font-medium">cost</th>
                  </tr>
                </thead>
                <tbody>
                  {routing.policies.map((row) => (
                    <tr key={row.policy} className={`table-row ${row.policy === 'aether' ? 'bg-accent/5' : ''}`}>
                      <td className="py-2 font-mono text-xs text-accent">{row.policy}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.coverage, 3)}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.accuracy_on_answered, 4)}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.escalation_rate, 3)}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.review_rate, 3)}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.abstention_rate, 3)}</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.mean_latency_ms, 1)} ms</td>
                      <td className="py-2 font-mono text-xs">{formatNumber(row.expected_cost_per_sample, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-muted">{routing.interpretation}</p>
            <div className="mt-6">
              <p className="label mb-2">ablations</p>
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {Object.entries(routing.ablations).map(([name, row]) => (
                  <div key={name} className="rounded-lg border border-line bg-raised/40 p-3">
                    <p className="font-mono text-[11px] text-accent">{name}</p>
                    <p className="mt-1 text-xs text-muted">
                      coverage {formatNumber(row.coverage, 3)} · acc {formatNumber(row.accuracy_on_answered, 4)}
                    </p>
                    <p className="text-xs text-muted">cost {formatNumber(row.expected_cost_per_sample, 3)}</p>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </Panel>

      <Panel title="Recent decisions" subtitle="Explanation and reason codes per request">
        {decisions.loading && !decisions.data ? (
          <Loading />
        ) : !decisions.data || decisions.data.items.length === 0 ? (
          <EmptyState message="No decisions recorded yet." />
        ) : (
          <ul className="space-y-2">
            {decisions.data.items.map((item, index) => (
              <li key={`${String(item.id)}-${index}`} className="rounded-lg border border-line bg-raised/40 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-3">
                    <ActionPill action={String(item.action)} />
                    <span className="font-mono text-[11px] text-muted">{String(item.policy)}</span>
                    <span className="font-mono text-[11px] text-muted">risk {formatNumber(Number(item.risk), 3)}</span>
                  </div>
                  <span className="font-mono text-[10px] text-muted">{String(item.created_at)}</span>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-muted">{String(item.explanation)}</p>
                <p className="mt-1 font-mono text-[10px] text-accent">
                  {(item.reason_codes as string[] | undefined)?.join(' · ') || '—'}
                </p>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
