import { HAND_FEATURES, normalizeHand, type Point } from "./normalize";

/**
 * 2 hands x 63 values + 3 values for the offset between the two wrists.
 * IMPORTANT: ml/scripts/train_static.py expects exactly this layout.
 */
export const FEATURE_DIM = HAND_FEATURES * 2 + 3; // 129

/**
 * Turns MediaPipe output into one fixed-length feature vector.
 * Returns null when no hand is visible.
 *
 * - Hands are ordered left-to-right in the image (not by MediaPipe's
 *   "handedness" label, which flips on mirrored video).
 * - A missing second hand is filled with zeros.
 * - `aspect` (video width / height) removes the 16:9 stretch so that
 *   different cameras give the same numbers.
 */
export function extractFeatures(
  hands: Point[][],
  aspect: number,
): number[] | null {
  if (hands.length === 0) return null;

  const corrected = hands.map((hand) =>
    hand.map((p) => ({ x: p.x * aspect, y: p.y, z: p.z * aspect })),
  );
  const ordered = [...corrected]
    .sort((a, b) => a[0].x - b[0].x)
    .slice(0, 2)
    .map(normalizeHand);

  const out = new Array<number>(FEATURE_DIM).fill(0);
  ordered.forEach((h, i) => {
    for (let k = 0; k < HAND_FEATURES; k++) out[i * HAND_FEATURES + k] = h.values[k];
  });

  if (ordered.length === 2) {
    const [a, b] = ordered;
    const s = (a.scale + b.scale) / 2;
    out[HAND_FEATURES * 2] = (b.wrist.x - a.wrist.x) / s;
    out[HAND_FEATURES * 2 + 1] = (b.wrist.y - a.wrist.y) / s;
    out[HAND_FEATURES * 2 + 2] = (b.wrist.z - a.wrist.z) / s;
  }
  return out;
}
