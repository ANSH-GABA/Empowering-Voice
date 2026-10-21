/** Format written by Model/scripts/train_static.py (static_mlp.json). */
export interface MlpModel {
  version: number;
  featureDim: number;
  classes: string[];
  mean: number[];
  scale: number[];
  layers: { W: number[][]; b: number[] }[]; // W is [inputs][outputs]
}

export interface Prediction {
  label: string;
  confidence: number;
  top3?: { label: string; score: number }[];
  cropPreview?: string;
}

export class StaticRemoteClassifier {
  private busy = false;
  private lastRequestAt = 0;

  /** Sends a landmark vector to the backend, limiting requests to five per second. */
  async classify(features: number[]): Promise<Prediction | null> {
    const now = Date.now();
    if (this.busy || now - this.lastRequestAt < 200) return null;
    this.busy = true;
    this.lastRequestAt = now;
    try {
      const res = await fetch("/api/landmark/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      if (!res.ok) throw new Error(`Backend landmark model error ${res.status}`);
      const result = (await res.json()) as Prediction;
      if (
        typeof result.label !== "string" ||
        !Number.isFinite(result.confidence) ||
        result.confidence < 0 ||
        result.confidence > 1
      ) {
        throw new Error("Backend returned an invalid landmark prediction.");
      }
      return result;
    } finally {
      this.busy = false;
    }
  }
}

export async function loadStaticModel(
  url = "/models/static_mlp.json",
): Promise<MlpModel> {
  const res = await fetch(url);
  const type = res.headers.get("content-type") ?? "";
  if (!res.ok || type.includes("text/html")) {
    throw new Error("static_mlp.json not found");
  }
  const model = (await res.json()) as MlpModel;
  if (
    !model ||
    typeof model !== "object" ||
    model.version !== 1 ||
    model.featureDim !== 129 ||
    !Array.isArray(model.classes) ||
    !model.classes.length ||
    model.classes.some((label) => typeof label !== "string" || !label) ||
    !Array.isArray(model.mean) ||
    model.mean.length !== model.featureDim ||
    !Array.isArray(model.scale) ||
    model.scale.length !== model.featureDim ||
    model.mean.some((value) => !Number.isFinite(value)) ||
    model.scale.some((value) => !Number.isFinite(value) || value === 0) ||
    !Array.isArray(model.layers) ||
    !model.layers.length
  ) {
    throw new Error("static_mlp.json has an invalid model format");
  }

  let inputDim = model.featureDim;
  for (const layer of model.layers) {
    const outputDim = layer.b.length;
    if (
      outputDim === 0 ||
      layer.W.length !== inputDim ||
      layer.W.some((row) => row.length !== outputDim || row.some((value) => !Number.isFinite(value))) ||
      layer.b.some((value) => !Number.isFinite(value))
    ) {
      throw new Error("static_mlp.json has invalid layer dimensions or values");
    }
    inputDim = outputDim;
  }
  if (inputDim !== model.classes.length) {
    throw new Error("static_mlp.json output dimension does not match its classes");
  }
  return model;
}

/** Plain forward pass: scaler -> Dense(ReLU) -> Dense(ReLU) -> softmax. */
export function predictStatic(model: MlpModel, features: number[]): Prediction {
  let x = features.map((v, i) => (v - model.mean[i]) / (model.scale[i] || 1));
  model.layers.forEach((layer, li) => {
    const outDim = layer.b.length;
    const y = layer.b.slice();
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = layer.W[i];
      for (let j = 0; j < outDim; j++) y[j] += xi * row[j];
    }
    const isLast = li === model.layers.length - 1;
    x = isLast ? y : y.map((v) => (v > 0 ? v : 0));
  });

  const max = Math.max(...x);
  const exps = x.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  let best = 0;
  for (let i = 1; i < exps.length; i++) if (exps[i] > exps[best]) best = i;
  return { label: model.classes[best], confidence: exps[best] / sum };
}
