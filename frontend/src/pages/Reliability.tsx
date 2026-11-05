import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../services/api';
import { useApi } from '../hooks/useApi';
import { EmptyState, ErrorState, Loading, Panel, StatCard, formatNumber } from '../components/primitives';

interface ReliabilityRow {
  bin_lower: number;
  bin_upper: number;
  count: number;
  avg_confidence: number | null;
  accuracy: number | null;
}

export default function Reliability() {
  const report = useApi(() => api.report('benchmarks'), []);

  if (report.loading && !report.data) return <Loading label="Loading benchmark report" />;
  if (report.error) {
    return (
      <div className="space-y-4">
        <header>
          <p className="label">02 / Model reliability</p>
          <h1 className="mt-1 text-2xl font-medium tracking-tight">Calibration and selective prediction</h1>
        </header>
        <ErrorState message={`${report.error.message} — run experiments/run_benchmarks.py to generate it.`} onRetry={() => void report.refresh()} />
      </div>
    );
  }

  const data = report.data as Record<string, any>;
  const rows = (['lightweight', 'advanced'] as const).map((role) => ({
    role,
    clean: data?.[role]?.clean_test,
    selective: data?.[role]?.selective,
    latency: data?.[role]?.latency_ms_per_sample,
    raw: data?.[role]?.raw_uncalibrated,
  }));

  const curve = (data?.lightweight?.clean_test?.reliability_curve ?? []) as ReliabilityRow[];
  const chart = curve
    .filter((row) => row.count > 0 && row.accuracy !== null)
    .map((row) => ({
      confidence: Number(row.avg_confidence),
      accuracy: Number(row.accuracy),
      perfect: Number(row.avg_confidence),
    }));

  return (
    <div className="space-y-6">
      <header>
        <p className="label">02 / Model reliability</p>
        <h1 className="mt-1 text-2xl font-medium tracking-tight">Calibration and selective prediction</h1>
        <p className="mt-1 text-sm text-muted">
          Measured on the frozen CIFAR-10 test split. Report generated {String(data.generated_at)}.
        </p>
      </header>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {rows.map((row) => (
          <StatCard
            key={`${row.role}-acc`}
            label={`${row.role} accuracy`}
            value={row.clean ? formatNumber(row.clean.accuracy, 4) : '—'}
            tone={row.role === 'advanced' ? 'accent' : 'default'}
            hint={row.clean ? `macro F1 ${formatNumber(row.clean.macro_f1, 4)}` : undefined}
          />
        ))}
        {rows.map((row) => (
          <StatCard
            key={`${row.role}-ece`}
            label={`${row.role} ECE`}
            value={row.clean ? formatNumber(row.clean.ece, 4) : '—'}
            tone="warn"
            hint={row.raw ? `uncalibrated ${formatNumber(row.raw.ece, 4)}` : undefined}
          />
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel title="Reliability diagram" subtitle="Lightweight model, calibrated probabilities">
          {chart.length === 0 ? (
            <EmptyState message="No reliability bins available." />
          ) : (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chart} margin={{ top: 8, right: 12, bottom: 6, left: -16 }}>
                  <CartesianGrid stroke="#1a2436" />
                  <XAxis dataKey="confidence" type="number" domain={[0, 1]} tick={{ fill: '#8b9bb8', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: '#8b9bb8', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: '#0e1523', border: '1px solid #1a2436', borderRadius: 8, color: '#e8eefc' }}
                  />
                  <Line type="monotone" dataKey="perfect" stroke="#1a2436" strokeDasharray="4 4" dot={false} name="perfect calibration" />
                  <Line type="monotone" dataKey="accuracy" stroke="#3ee6d0" strokeWidth={2} dot={{ r: 3 }} name="observed accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Panel>

        <Panel title="Selective prediction" subtitle="Risk at fixed coverage">
          <div className="space-y-4">
            {rows.map((row) => (
              <div key={row.role} className="rounded-lg border border-line bg-raised/40 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-ink">{row.role}</span>
                  <span className="font-mono text-xs text-muted">AURC {formatNumber(row.selective?.aurc, 4)}</span>
                </div>
                <ul className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px] text-muted">
                  {Object.entries(row.selective?.risk_at_coverage ?? {}).map(([coverage, risk]) => (
                    <li key={coverage} className="flex flex-col">
                      <span>cov {coverage}</span>
                      <span className="text-ink">{formatNumber(Number(risk), 4)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <p className="text-[11px] leading-relaxed text-muted">
              AURC is the area under the risk–coverage curve when samples are accepted from most to
              least confident. Lower is better.
            </p>
          </div>
        </Panel>
      </div>

      <Panel title="Metric table" subtitle="Calibration, discrimination and cost">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left">
            <thead>
              <tr className="text-[11px] uppercase tracking-wider text-muted">
                <th className="pb-2 font-medium">model</th>
                <th className="pb-2 font-medium">accuracy</th>
                <th className="pb-2 font-medium">macro F1</th>
                <th className="pb-2 font-medium">ECE</th>
                <th className="pb-2 font-medium">Brier</th>
                <th className="pb-2 font-medium">NLL</th>
                <th className="pb-2 font-medium">AURC</th>
                <th className="pb-2 font-medium">ms / sample</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.role} className="table-row">
                  <td className="py-2 text-sm">{row.role}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.clean?.accuracy, 4)}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.clean?.macro_f1, 4)}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.clean?.ece, 4)}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.clean?.brier, 4)}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.clean?.nll, 4)}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.clean?.aurc, 4)}</td>
                  <td className="py-2 font-mono text-xs">{formatNumber(row.latency, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-[11px] leading-relaxed text-muted">
          Calibrated probabilities come from a temperature fitted on a dedicated calibration split.
          Calibration improves the honesty of uncertainty estimates; it does not guarantee correctness.
        </p>
      </Panel>
    </div>
  );
}
