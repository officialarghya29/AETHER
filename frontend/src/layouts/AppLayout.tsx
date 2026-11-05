import { NavLink, Outlet } from 'react-router-dom';
import { api } from '../services/api';
import { useApi } from '../hooks/useApi';

const NAV = [
  { to: '/', label: 'Overview', code: '00' },
  { to: '/inference', label: 'Live inference', code: '01' },
  { to: '/reliability', label: 'Model reliability', code: '02' },
  { to: '/drift', label: 'Drift monitoring', code: '03' },
  { to: '/routing', label: 'Routing intelligence', code: '04' },
  { to: '/diagnosis', label: 'Failure diagnosis', code: '05' },
  { to: '/reviews', label: 'Human review', code: '06' },
  { to: '/experiments', label: 'Experiment centre', code: '07' },
];

export default function AppLayout() {
  const status = useApi(() => api.systemStatus(), []);

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="border-b border-line/80 bg-panel/60 lg:h-screen lg:border-b-0 lg:border-r">
        <div className="flex h-full flex-col">
          <div className="px-6 py-6">
            <p className="font-mono text-[11px] tracking-[0.3em] text-accent">AETHER-ML</p>
            <h1 className="mt-1 text-lg font-medium tracking-tight">Reliability Console</h1>
            <p className="mt-1 text-xs text-muted">
              Trust, evaluation &amp; recovery for ML systems
            </p>
          </div>

          <nav className="flex-1 space-y-1 px-3" aria-label="Primary">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                    isActive
                      ? 'bg-accent/10 text-accent'
                      : 'text-muted hover:bg-raised/60 hover:text-ink'
                  }`
                }
              >
                <span className="font-mono text-[10px] text-muted/70">{item.code}</span>
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="border-t border-line/70 px-6 py-4 text-[11px] text-muted">
            {status.data ? (
              <ul className="space-y-1.5">
                <li className="flex items-center justify-between">
                  <span>status</span>
                  <span className="flex items-center gap-2 text-ink">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        status.data.artifacts.lightweight_checkpoint ? 'bg-ok' : 'bg-danger'
                      }`}
                    />
                    {status.data.artifacts.lightweight_checkpoint ? 'operational' : 'degraded'}
                  </span>
                </li>
                <li className="flex items-center justify-between">
                  <span>device</span>
                  <span className="font-mono text-ink">{status.data.device}</span>
                </li>
                <li className="flex items-center justify-between">
                  <span>policy</span>
                  <span className="font-mono text-ink">{status.data.routing_policy}</span>
                </li>
                <li className="flex items-center justify-between">
                  <span>version</span>
                  <span className="font-mono text-ink">{status.data.version}</span>
                </li>
              </ul>
            ) : (
              <span className="text-muted">{status.error ? 'API offline' : 'connecting…'}</span>
            )}
            <p className="mt-3 font-mono text-[10px] text-muted/60">
              calibrated · shift-aware · cost-routed
            </p>
          </div>
        </div>
      </aside>

      <main className="min-w-0 px-5 py-6 sm:px-8 lg:px-10">
        <Outlet />
      </main>
    </div>
  );
}
