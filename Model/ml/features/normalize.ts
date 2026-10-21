export interface Point {
  x: number;
  y: number;
  z: number;
}

export const LANDMARKS_PER_HAND = 21;
export const HAND_FEATURES = LANDMARKS_PER_HAND * 3; // 63

export interface NormalizedHand {
  values: number[]; // 63 numbers
  scale: number; // hand size used for scaling
  wrist: Point;
}

/**
 * Makes the hand shape independent of where the hand is and how big it
 * looks: subtract the wrist (landmark 0), divide by the largest distance.
 */
export function normalizeHand(hand: Point[]): NormalizedHand {
  const wrist = hand[0];
  const rel = hand.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));
  let scale = 0;
  for (const p of rel) scale = Math.max(scale, Math.hypot(p.x, p.y, p.z));
  if (scale < 1e-6) scale = 1;
  const values: number[] = [];
  for (const p of rel) values.push(p.x / scale, p.y / scale, p.z / scale);
  return { values, scale, wrist };
}
