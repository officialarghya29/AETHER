import { useRef, useState } from 'react';
import { ApiError, api } from '../services/api';
import type { Prediction } from '../types';
import {
  ActionPill,
  ErrorState,
  Panel,
  ProbabilityBars,
  SeverityPill,
  formatNumber,
  formatPercent,
} from '../components/primitives';

export default function LiveInference() {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onSelect = (file: File) => {
    setFileName(file.name);
    setPrediction(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
  };

  const run = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      setPrediction(await api.predict(file));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Prediction failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <header>
        <p className="label">01 / Live inference</p>
        <h1 className="mt-1 text-2xl font-medium tracking-tight">Run the reliability pipeline</h1>
        <p className="mt-1 text-sm text-muted">
          Upload an image. The pipeline returns calibrated uncertainty, shift evidence, a routing
          decision with its justification, and diagnostics.
        </p>
      </header>

      <div className="grid gap-6 xl:grid-cols-[0.85fr_1.15fr]">
        <Panel title="Input" subtitle="PNG, JPEG, WebP or BMP · max 5 MB">
          <div
            className="flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-line bg-raised/40 p-6 text-center transition hover:border-accent/40"
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => event.key === 'Enter' && inputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            {preview ? (
              <img src={preview} alt="Uploaded input" className="max-h-52 rounded-md object-contain" />
            ) : (
              <>
                <p className="text-sm text-ink">Select an image</p>
                <p className="mt-1 text-xs text-muted">CIFAR-10 classes · 32×32 native, upscaled to 64×64</p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  onSelect(file);
                  void run(file);
                }
              }}
            />
          </div>
          <div className="mt-4 flex items-center justify-between text-xs text-muted">
            <span className="truncate font-mono">{fileName || 'no file selected'}</span>
            {busy && <span className="text-accent">inferring…</span>}
          </div>
          {error && <div className="mt-4"><ErrorState message={error} /></div>}
        </Panel>

        <Panel
          title="Result"
          actions={prediction ? <ActionPill action={prediction.action} /> : undefined}
          subtitle={prediction ? `${prediction.model_id} · ${prediction.model_version}` : 'Awaiting an input'}
        >
          {!prediction ? (
            <p className="py-10 text-center text-sm text-muted">No prediction yet.</p>
          ) : (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center gap-4">
                <div>
                  <p className="label">predicted class</p>
                  <p className="mt-1 text-xl font-medium text-accent">{prediction.predicted_label}</p>
                </div>
                <div>
                  <p className="label">confidence</p>
                  <p className="mt-1 font-mono text-xl">{formatPercent(prediction.uncertainty.max_probability)}</p>
                </div>
                <div>
                  <p className="label">estimated risk</p>
                  <p className="mt-1 font-mono text-xl">{formatNumber(prediction.risk.value, 3)}</p>
                  <p className="text-[10px] text-muted">{prediction.risk.source} model</p>
                </div>
                <div>
                  <p className="label">drift</p>
                  <p className="mt-1"><SeverityPill severity={prediction.drift.severity} /></p>
                </div>
                <div>
                  <p className="label">latency</p>
                  <p className="mt-1 font-mono text-xl">{formatNumber(prediction.latency_ms, 1)}<span className="text-sm text-muted"> ms</span></p>
                </div>
              </div>

              <div>
                <p className="label mb-2">class probabilities</p>
                <ProbabilityBars probabilities={prediction.probabilities} />
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border border-line bg-raised/40 p-3">
                  <p className="label">uncertainty</p>
                  <ul className="mt-2 space-y-1 font-mono text-xs text-ink">
                    <li className="flex justify-between"><span className="text-muted">entropy</span>{formatNumber(prediction.uncertainty.predictive_entropy, 4)}</li>
                    <li className="flex justify-between"><span className="text-muted">margin</span>{formatNumber(prediction.uncertainty.prediction_margin, 4)}</li>
                    <li className="flex justify-between"><span className="text-muted">temperature</span>{formatNumber(prediction.temperature, 3)}</li>
                  </ul>
                </div>
                <div className="rounded-lg border border-line bg-raised/40 p-3">
                  <p className="label">decision evidence</p>
                  <p className="mt-2 text-xs leading-relaxed text-muted">{prediction.decision.explanation}</p>
                  <p className="mt-2 font-mono text-[10px] text-accent">
                    {prediction.decision.reason_codes.join(' · ') || '—'}
                  </p>
                </div>
              </div>

              {prediction.advanced && (
                <div className="rounded-lg border border-warn/30 bg-warn/5 p-3">
                  <p className="label">advanced model comparison</p>
                  <div className="mt-2 flex flex-wrap items-center gap-4 text-xs">
                    <span className="font-mono">{String(prediction.advanced.model_id)}</span>
                    <span className="text-muted">
                      class <span className="text-ink">{String(prediction.advanced.class_label)}</span>
                    </span>
                    <span className="text-muted">
                      disagreement <span className="text-ink">{formatNumber(Number(prediction.advanced.disagreement), 3)}</span>
                    </span>
                  </div>
                </div>
              )}

              {prediction.diagnostics.length > 0 && (
                <div>
                  <p className="label mb-2">diagnostics</p>
                  <ul className="space-y-2">
                    {prediction.diagnostics.map((diagnostic) => (
                      <li key={diagnostic.code} className="rounded-lg border border-line bg-raised/40 p-3">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono text-xs text-accent">{diagnostic.code}</span>
                          <SeverityPill severity={diagnostic.severity} />
                        </div>
                        <p className="mt-1.5 text-xs leading-relaxed text-muted">{diagnostic.explanation}</p>
                        <p className="mt-1 text-[11px] text-ink">{diagnostic.recommended_action}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {prediction.review_id && (
                <p className="rounded-lg border border-plasma/30 bg-plasma/5 px-3 py-2 text-xs text-plasma">
                  Queued for human review · case {prediction.review_id}
                </p>
              )}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
