import type { ReactNode } from 'react';
import type { Action, Severity } from '../types';

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-head">
          <div>
            {title && <h2 className="text-sm font-medium tracking-wide text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function StatCard({
  label,
  value,
  unit,
  hint,
  tone = 'default',
}: {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
  tone?: 'default' | 'accent' | 'warn' | 'danger' | 'ok';
}) {
  const toneClass = {
    default: 'text-ink',
    accent: 'text-accent',
    warn: 'text-warn',
    danger: 'text-danger',
    ok: 'text-ok',
  }[tone];
  return (
    <div className="panel px-4 py-4">
      <p className="label">{label}</p>
      <p className={`metric mt-2 ${toneClass}`}>
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-muted">{unit}</span>}
      </p>
      {hint && <p className="mt-1 text-[11px] text-muted">{hint}</p>}
    </div>
  );
}

const SEVERITY_STYLES: Record<string, string> = {
  none: 'border-line text-muted',
  low: 'border-plasma/40 bg-plasma/10 text-plasma',
  moderate: 'border-warn/40 bg-warn/10 text-warn',
  severe: 'border-danger/40 bg-danger/10 text-danger',
  info: 'border-line text-muted',
  medium: 'border-warn/40 bg-warn/10 text-warn',
  high: 'border-danger/40 bg-danger/10 text-danger',
  critical: 'border-danger/60 bg-danger/20 text-danger',
};

export function SeverityPill({ severity }: { severity: Severity | string }) {
  return <span className={`tag ${SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.none}`}>{severity}</span>;
}

const ACTION_STYLES: Record<string, string> = {
  ACCEPT: 'border-ok/40 bg-ok/10 text-ok',
  MONITOR: 'border-plasma/40 bg-plasma/10 text-plasma',
  ESCALATE: 'border-warn/40 bg-warn/10 text-warn',
  HUMAN_REVIEW: 'border-warn/50 bg-warn/15 text-warn',
  ABSTAIN: 'border-danger/40 bg-danger/10 text-danger',
  REJECT: 'border-danger/60 bg-danger/20 text-danger',
};

export function ActionPill({ action }: { action: Action | string }) {
  return <span className={`tag ${ACTION_STYLES[action] ?? 'border-line text-muted'}`}>{action}</span>;
}

export function ProbabilityBars({ probabilities }: { probabilities: number[] }) {
  const labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
  ];
  const max = Math.max(...probabilities, 0.0001);
  return (
    <div className="space-y-1.5">
      {probabilities.map((value, index) => (
        <div key={index} className="flex items-center gap-3">
          <span className="w-20 shrink-0 font-mono text-[11px] text-muted">{labels[index] ?? index}</span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-raised">
            <div
              className="h-full rounded-full bg-gradient-to-r from-plasma/70 to-accent transition-[width] duration-300"
              style={{ width: `${Math.max((value / max) * 100, 1.5)}%` }}
            />
          </div>
          <span className="w-14 shrink-0 text-right font-mono text-[11px] text-ink">
            {(value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-8 text-sm text-muted">
      <span className="h-3 w-3 animate-pulse rounded-full bg-accent" />
      {label}…
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-lg border border-danger/30 bg-danger/5 px-4 py-3">
      <p className="text-sm text-danger">{message}</p>
      {onRetry && (
        <button type="button" className="btn mt-3" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return <p className="py-8 text-center text-sm text-muted">{message}</p>;
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}
