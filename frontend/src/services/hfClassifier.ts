import type { Point } from "../ml/features/normalize";
import type { Prediction } from "../ml/models/staticClassifier";

const API = "/api"; // Vite proxies this to FastAPI on :8000 (see vite.config.ts)
const INPUT_SIZE = 224; // the ViT model's input size

export interface HfHealth {
  ok: boolean;
  model?: string;
  loaded?: boolean;
}

export async function checkHfHealth(): Promise<HfHealth> {
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) return { ok: false };
    return (await res.json()) as HfHealth;
  } catch {
    return { ok: false };
  }
}

/**
 * Square box around every visible hand (both hands for two-handed signs),
 * with a margin, clamped inside the video frame.
 */
export function handCropRect(hands: Point[][], vw: number, vh: number, margin = 0.25) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const hand of hands) {
    for (const p of hand) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
  }
  const cx = ((minX + maxX) / 2) * vw;
  const cy = ((minY + maxY) / 2) * vh;
  const side = Math.max((maxX - minX) * vw, (maxY - minY) * vh) * (1 + 2 * margin);
  const s = Math.min(Math.max(side, 96), Math.min(vw, vh));
  const x = Math.min(Math.max(cx - s / 2, 0), vw - s);
  const y = Math.min(Math.max(cy - s / 2, 0), vh - s);
  return { x, y, s };
}

/**
 * Sends one hand crop at a time to the local backend. While a request is
 * in flight, new frames are skipped, so the UI never queues up stale work.
 */
export class HfRemoteClassifier {
  private busy = false;
  private canvas = document.createElement("canvas");

  constructor() {
    this.canvas.width = INPUT_SIZE;
    this.canvas.height = INPUT_SIZE;
  }

  /** Resolves to null when skipped because a request is already running. */
  async classify(video: HTMLVideoElement, hands: Point[][], flip: boolean): Promise<Prediction | null> {
    if (this.busy) return null;
    this.busy = true;
    try {
      const { x, y, s } = handCropRect(hands, video.videoWidth, video.videoHeight);
      const ctx = this.canvas.getContext("2d")!;
      ctx.save();
      if (flip) {
        ctx.translate(INPUT_SIZE, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, x, y, s, s, 0, 0, INPUT_SIZE, INPUT_SIZE);
      ctx.restore();

      const blob = await new Promise<Blob>((resolve, reject) =>
        this.canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode frame"))), "image/jpeg", 0.85),
      );
      const form = new FormData();
      form.append("image", blob, "hand.jpg");
      const res = await fetch(`${API}/predict`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Backend error ${res.status}`);
      const data = (await res.json()) as { predictions: { label: string; score: number }[] };
      const top = data.predictions[0];
      return top ? { label: top.label, confidence: top.score } : null;
    } finally {
      this.busy = false;
    }
  }
}
