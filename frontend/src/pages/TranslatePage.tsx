import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import CameraView from "../components/camera/CameraView";
import PredictionCard from "../components/recognition/PredictionCard";
import SentencePanel from "../components/sentence/SentencePanel";
import { extractFeatures } from "../ml/features/featureExtractor";
import type { Point } from "../ml/features/normalize";
import {
  loadStaticModel,
  predictStatic,
  StaticRemoteClassifier,
  type MlpModel,
  type Prediction,
} from "../ml/models/staticClassifier";
import { applyToken, PredictionStabilizer } from "../ml/prediction/smoother";
import { checkHfHealth, HfRemoteClassifier, type HfHealth } from "../services/hfClassifier";
import { speak } from "../services/speechService";

type Engine = "hf" | "landmark" | "backend";
type SigningHand = "right" | "left";

// The model responds ~10-15 times per second, so a responsive 5-frame window works well.
const STABILIZER_CONFIG = { windowSize: 5, minVotes: 4, minConfidence: 0.65, releaseFrames: 8 };

export default function TranslatePage() {
  const [engine, setEngine] = useState<Engine>("hf");
  const [signingHand, setSigningHand] = useState<SigningHand>("right");
  const [model, setModel] = useState<MlpModel | null>(null);
  const [modelMissing, setModelMissing] = useState(false);
  const [hfHealth, setHfHealth] = useState<HfHealth | null>(null);
  const [hfError, setHfError] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [cropPreview, setCropPreview] = useState<string | null>(null);
  const [sentence, setSentence] = useState("");

  const engineRef = useRef(engine);
  const signingHandRef = useRef(signingHand);
  const modelRef = useRef<MlpModel | null>(null);
  const handsVisible = useRef(false);
  const landmarkStab = useRef(new PredictionStabilizer());
  const backendStab = useRef(new PredictionStabilizer(STABILIZER_CONFIG));
  const hfStab = useRef(new PredictionStabilizer(STABILIZER_CONFIG));
  const hf = useRef(new HfRemoteClassifier());
  const backendClassifier = useRef(new StaticRemoteClassifier());

  engineRef.current = engine;
  signingHandRef.current = signingHand;

  useEffect(() => {
    loadStaticModel()
      .then((m) => {
        modelRef.current = m;
        setModel(m);
        setModelMissing(false);
      })
      .catch(() => setModelMissing(true));
  }, [engine]);

  useEffect(() => {
    landmarkStab.current.reset();
    backendStab.current.reset();
    hfStab.current.reset();
    setBackendError(null);
    setPrediction(null);
    if (engine !== "landmark") checkHfHealth().then(setHfHealth);
  }, [engine]);

  const commit = (label: string | null) => {
    if (label) setSentence((s) => applyToken(s, label));
  };

  const handleFrame = useCallback((hands: Point[][], aspect: number, video: HTMLVideoElement) => {
    handsVisible.current = hands.length > 0;

    if (engineRef.current === "hf") {
      if (hands.length === 0) {
        setPrediction(null);
        setCropPreview(null);
        hfStab.current.push(null, 0);
        return;
      }
      // For left-handed signers, flip the crop horizontally so it matches standard right-handed ASL training data
      const flipForModel = signingHandRef.current === "left";
      hf.current
        .classify(video, hands, flipForModel)
        .then((p) => {
          if (!p || !handsVisible.current || engineRef.current !== "hf") return;
          setHfError(null);
          setPrediction(p);
          if (p.cropPreview) setCropPreview(p.cropPreview);
          commit(hfStab.current.push(p.label, p.confidence));
        })
        .catch(() => setHfError("Can't reach the backend. Ensure 'uvicorn app.main:app' is running on port 8000."));
      return;
    }

    const features = extractFeatures(hands, aspect);
    if (!features) {
      setPrediction(null);
      setCropPreview(null);
      landmarkStab.current.push(null, 0);
      backendStab.current.push(null, 0);
      return;
    }

    if (engineRef.current === "backend") {
      backendClassifier.current
        .classify(features)
        .then((p) => {
          if (!p || !handsVisible.current || engineRef.current !== "backend") return;
          setBackendError(null);
          setPrediction(p);
          commit(backendStab.current.push(p.label, p.confidence));
        })
        .catch((error: unknown) => {
          setBackendError(
            error instanceof Error ? error.message : "Backend landmark prediction failed.",
          );
        });
      return;
    }

    const m = modelRef.current;
    if (!m) return;
    const p = predictStatic(m, features);
    setPrediction(p);
    commit(landmarkStab.current.push(p.label, p.confidence));
  }, []);

  const tab = (active: boolean) =>
    `rounded-lg border-2 px-4 py-2 text-base sm:text-lg font-bold transition-all ${
      active ? "border-ink bg-ink text-white shadow-sm" : "border-line bg-white text-ink hover:bg-line/40"
    }`;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
      <div className="space-y-4">
        {/* Recognition Engine Tabs */}
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Recognition engine">
          <button className={tab(engine === "hf")} aria-pressed={engine === "hf"} onClick={() => setEngine("hf")}>
            ASL Vision Model
          </button>
          <button
            className={tab(engine === "landmark")}
            aria-pressed={engine === "landmark"}
            onClick={() => setEngine("landmark")}
          >
            Browser Landmark Model
          </button>
          <button
            className={tab(engine === "backend")}
            aria-pressed={engine === "backend"}
            onClick={() => setEngine("backend")}
          >
            Backend Landmark Model
          </button>
        </div>

        {/* Main Camera Feed */}
        <CameraView onFrame={handleFrame} />

        {/* Simplified Hand Selection for Normal Users */}
        <div className="rounded-xl border border-line bg-white p-4 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-muted">Signing Hand</span>
              <p className="text-sm font-semibold text-ink">
                {signingHand === "right"
                  ? "Standard Right Hand Mode"
                  : "Left Hand Mode (Auto-Mirrored for AI)"}
              </p>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setSigningHand("right")}
                className={`flex items-center gap-2 rounded-lg border-2 px-4 py-2 font-bold transition-all ${
                  signingHand === "right"
                    ? "border-signal bg-signal/10 text-signal shadow-sm"
                    : "border-line bg-white text-muted hover:border-ink hover:text-ink"
                }`}
              >
                <span className="text-xl">✋</span> Right Hand
              </button>
              <button
                type="button"
                onClick={() => setSigningHand("left")}
                className={`flex items-center gap-2 rounded-lg border-2 px-4 py-2 font-bold transition-all ${
                  signingHand === "left"
                    ? "border-signal bg-signal/10 text-signal shadow-sm"
                    : "border-line bg-white text-muted hover:border-ink hover:text-ink"
                }`}
              >
                <span className="text-xl">🤚</span> Left Hand
              </button>
            </div>
          </div>
          <p className="mt-2 text-xs text-muted">
            {signingHand === "right"
              ? "Hold up your right hand to sign letters A–Z. Keep your palm facing the camera."
              : "Left-handed signers can sign naturally. The system automatically mirrors the image so the AI recognizes it."}
          </p>
        </div>

        {/* Engine Status & Live Vision Thumbnail */}
        {engine === "hf" && (
          <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-line bg-white p-4 text-base shadow-sm">
            <div className="space-y-1">
              {hfHealth?.ok ? (
                <div className="flex items-center gap-2">
                  <span className="size-2.5 rounded-full bg-signal animate-pulse" />
                  <p className="font-semibold text-ink">
                    AI Connected · <span className="text-muted">{hfHealth.model}</span>
                  </p>
                </div>
              ) : (
                <p className="font-bold text-alert">
                  Backend offline. Start with <code>uvicorn app.main:app --port 8000</code>
                </p>
              )}
              {hfError && <p className="font-bold text-alert">{hfError}</p>}
              <p className="text-xs text-muted">
                Recognizes single-handed ASL alphabet letters (A–Z).
              </p>
            </div>

            {/* AI Hand Vision Thumbnail */}
            <div className="flex items-center gap-3 rounded-lg border border-line bg-surface p-2">
              <div className="relative size-20 overflow-hidden rounded-md border-2 border-signal bg-ink flex items-center justify-center">
                {cropPreview ? (
                  <img
                    src={cropPreview}
                    alt="AI Hand Vision"
                    className="size-full object-cover"
                  />
                ) : (
                  <span className="text-center text-[10px] text-muted px-1">
                    No hand in frame
                  </span>
                )}
              </div>
              <div className="text-xs">
                <span className="font-bold uppercase tracking-wider text-muted block">AI Vision</span>
                <span className="font-semibold text-ink">
                  {cropPreview ? "Hand Tracked" : "Searching..."}
                </span>
                <span className="block text-[11px] text-muted">
                  {signingHand === "right" ? "Right hand crop" : "Left (mirrored)"}
                </span>
              </div>
            </div>
          </div>
        )}

        {engine === "backend" && (
          <div className="space-y-2 rounded-xl border border-line bg-white p-4 shadow-sm" role="status">
            {hfHealth?.ok && hfHealth.landmark_model_available ? (
              <p className="font-semibold text-signal">Backend connected · trained landmark model active</p>
            ) : (
              <p className="font-bold text-alert">
                Backend landmark model unavailable. Train it with{" "}
                <code>python Model/scripts/train_static.py</code>.
              </p>
            )}
            {backendError && <p className="font-bold text-alert">{backendError}</p>}
          </div>
        )}

        {engine === "landmark" && modelMissing && (
          <div className="rounded-xl border-2 border-amber bg-white p-4" role="status">
            No trained landmark model yet. Record signs on the{" "}
            <Link to="/collect" className="font-bold underline">
              Collect data
            </Link>{" "}
            page, then run <code className="font-bold">python Model/scripts/train_static.py</code>.
          </div>
        )}

        {engine === "landmark" && model && (
          <p className="text-muted">
            Model loaded: {model.classes.length} signs ({model.classes.slice(0, 8).join(" ")}
            {model.classes.length > 8 ? " …" : ""})
          </p>
        )}
      </div>

      {/* Recognition Panel */}
      <div className="space-y-4">
        <PredictionCard prediction={prediction} />
        <SentencePanel
          sentence={sentence}
          onSpace={() => setSentence((s) => s + " ")}
          onDelete={() => setSentence((s) => s.slice(0, -1))}
          onClear={() => setSentence("")}
          onSpeak={() => speak(sentence, "en-US")}
        />
      </div>
    </div>
  );
}
