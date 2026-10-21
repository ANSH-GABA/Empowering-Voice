export interface StabilizerOptions {
  windowSize: number; // frames looked at
  minVotes: number; // frames in the window that must agree
  minConfidence: number; // frames below this count as "no vote"
  releaseFrames: number; // frames without a hand before the same sign can repeat
}

export const DEFAULT_STABILIZER: StabilizerOptions = {
  windowSize: 12,
  minVotes: 10,
  minConfidence: 0.75,
  releaseFrames: 8,
};

/**
 * Stops a held gesture from typing the same letter 30 times a second.
 * push() returns a label only once per "hold". To repeat a letter
 * (e.g. L-L), the signer lowers their hand briefly.
 */
export class PredictionStabilizer {
  private window: (string | null)[] = [];
  private last: string | null = null;
  private missing = 0;

  constructor(private opts: StabilizerOptions = DEFAULT_STABILIZER) {}

  push(label: string | null, confidence: number): string | null {
    if (label === null) {
      this.missing++;
      if (this.missing >= this.opts.releaseFrames) {
        this.last = null;
        this.window = [];
      }
      return null;
    }
    this.missing = 0;

    this.window.push(confidence >= this.opts.minConfidence ? label : null);
    if (this.window.length > this.opts.windowSize) this.window.shift();
    if (this.window.length < this.opts.windowSize) return null;

    const counts = new Map<string, number>();
    for (const l of this.window) if (l) counts.set(l, (counts.get(l) ?? 0) + 1);
    let best: string | null = null;
    let bestCount = 0;
    for (const [l, c] of counts) {
      if (c > bestCount) {
        best = l;
        bestCount = c;
      }
    }
    if (best && bestCount >= this.opts.minVotes && best !== this.last) {
      this.last = best;
      return best;
    }
    return null;
  }

  reset() {
    this.window = [];
    this.last = null;
    this.missing = 0;
  }
}

/** Applies a recognised token to the sentence. Works for letters and words. */
export function applyToken(sentence: string, token: string): string {
  switch (token) {
    case "SPACE":
      return sentence + " ";
    case "DELETE":
      return sentence.slice(0, -1);
    case "CLEAR":
      return "";
    default:
      return token.length > 1 ? `${sentence}${token} ` : sentence + token;
  }
}
