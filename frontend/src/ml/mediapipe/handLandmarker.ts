import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// The wasm version MUST match the version in package.json.
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";

// For an offline demo: download hand_landmarker.task into public/models/
// and change this to "/models/hand_landmarker.task".
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let instance: Promise<HandLandmarker> | null = null;

/** Creates the landmarker once and reuses it (safe with React StrictMode). */
export function getHandLandmarker(): Promise<HandLandmarker> {
  if (!instance) {
    instance = create().catch((err) => {
      instance = null; // allow retry after a failure
      throw err;
    });
  }
  return instance;
}

async function create(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const make = (delegate: "GPU" | "CPU") =>
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numHands: 2, // ISL fingerspelling uses both hands
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  try {
    return await make("GPU");
  } catch {
    return await make("CPU");
  }
}
