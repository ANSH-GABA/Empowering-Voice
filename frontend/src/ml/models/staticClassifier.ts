/** Format written by ml/scripts/train_static.py (static_mlp.json). */
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
}

export async function loadStaticModel(
  url = "/models/static_mlp.json",
): Promise<MlpModel> {
  const res = await fetch(url);
  const type = res.headers.get("content-type") ?? "";
  if (!res.ok || type.includes("text/html")) {
    throw new Error("static_mlp.json not found");
  }
  return (await res.json()) as MlpModel;
}

/** Plain forward pass: scaler -> Dense(ReLU) -> Dense(ReLU) -> softmax. */
export function predictStatic(model: MlpModel, features: number[]): Prediction {
  let x = features.map((v, i) => (v - model.mean[i]) / model.scale[i]);
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
