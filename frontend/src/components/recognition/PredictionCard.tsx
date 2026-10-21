import type { Prediction } from "../../../../Model/ml/models/staticClassifier";

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.75 ? "bg-signal" : value >= 0.5 ? "bg-amber" : "bg-alert";
  return (
    <div
      role="meter"
      aria-label="Confidence"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className="h-4 w-full overflow-hidden rounded-full bg-line"
    >
      <div className={`h-full ${color} transition-[width]`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function PredictionCard({ prediction }: { prediction: Prediction | null }) {
  return (
    <section aria-live="off" className="rounded-lg bg-white p-5">
      <h2 className="text-lg font-bold text-muted">Current sign</h2>
      <div className="flex items-end justify-between gap-4">
        <div className="min-h-24 text-8xl font-bold leading-none">
          {prediction ? prediction.label : <span className="text-line">–</span>}
        </div>
        <div className="pb-2 text-2xl font-bold">
          {prediction ? `${Math.round(prediction.confidence * 100)}%` : "No hand"}
        </div>
      </div>
      <div className="mt-3">
        <ConfidenceBar value={prediction?.confidence ?? 0} />
      </div>
    </section>
  );
}
