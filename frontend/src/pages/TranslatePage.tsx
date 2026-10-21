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

type Engine = "landmark" | "backend" | "hf";

// The HF model answers ~5 times per second, so it needs a shorter voting window.
const HF_STABILIZER = { windowSize: 5, minVotes: 4, minConfidence: 0.7, releaseFrames: 10 };

export default function TranslatePage() {
  const [engine, setEngine] = useState<Engine>("hf");
  const [flip, setFlip] = useState(false);
  const [model, setModel] = useState<MlpModel | null>(null);
  const [modelMissing, setModelMissing] = useState(false);
  const [hfHealth, setHfHealth] = useState<HfHealth | null>(null);
  const [hfError, setHfError] = useState<string | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [sentence, setSentence] = useState("");

  const engineRef = useRef(engine);
  const flipRef = useRef(flip);
  const modelRef = useRef<MlpModel | null>(null);
  const handsVisible = useRef(false);
  const landmarkStab = useRef(new PredictionStabilizer());
  const backendStab = useRef(new PredictionStabilizer(HF_STABILIZER));
  const hfStab = useRef(new PredictionStabilizer(HF_STABILIZER));
  const hf = useRef(new HfRemoteClassifier());
  const backendClassifier = useRef(new StaticRemoteClassifier());
  engineRef.current = engine;
  flipRef.current = flip;

  useEffect(() => {
    loadStaticModel()
      .then((m) => {
        modelRef.current = m;
        setModel(m);
      })
      .catch(() => setModelMissing(true));
  }, []);

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
        hfStab.current.push(null, 0);
        return;
      }
      hf.current
        .classify(video, hands, flipRef.current)
        .then((p) => {
          if (!p || !handsVisible.current || engineRef.current !== "hf") return;
          setHfError(null);
          setPrediction(p);
          commit(hfStab.current.push(p.label, p.confidence));
        })
        .catch(() => setHfError("Can't reach the backend. Start it (see below) and try again."));
      return;
    }

    const features = extractFeatures(hands, aspect);
    if (!features) {
      setPrediction(null);
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
    `rounded-lg border-2 px-4 py-2 text-lg font-bold ${
      active ? "border-ink bg-ink text-white" : "border-line bg-white hover:bg-line"
    }`;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Recognition engine">
          <button className={tab(engine === "hf")} aria-pressed={engine === "hf"} onClick={() => setEngine("hf")}>
            Hugging Face model
          </button>
          <button
            className={tab(engine === "landmark")}
            aria-pressed={engine === "landmark"}
            onClick={() => setEngine("landmark")}
          >
            My trained model
          </button>
          <button
            className={tab(engine === "backend")}
            aria-pressed={engine === "backend"}
            onClick={() => setEngine("backend")}
          >
            Backend trained model
          </button>
        </div>

        <CameraView onFrame={handleFrame} />

        {engine === "hf" && (
          <div className="space-y-2 rounded-lg bg-white p-4 text-lg" role="status">
            {hfHealth?.ok ? (
              <p>
                Backend connected · <span className="font-bold">{hfHealth.model}</span>
                {hfHealth.loaded === false && " (still loading…)"}
              </p>
            ) : (
              <p className="font-bold text-alert">
                Backend not running. In a terminal: <code>cd backend</code> then{" "}
                <code>uvicorn app.main:app --port 8000</code>
              </p>
            )}
            {hfError && <p className="font-bold text-alert">{hfError}</p>}
            <label className="flex items-center gap-2">
              <input type="checkbox" className="size-5" checked={flip} onChange={(e) => setFlip(e.target.checked)} />
              Flip the hand image before sending (try this if left/right look swapped)
            </label>
            <p className="text-muted">
              Reads one sign at a time: letters A–Z and digits 1–9. It does not cover words or moving signs.
            </p>
          </div>
        )}

        {engine === "backend" && (
          <div className="space-y-2 rounded-lg bg-white p-4 text-lg" role="status">
            {hfHealth?.ok && hfHealth.landmark_model_available ? (
              <p>Backend connected · trained landmark model available</p>
            ) : (
              <p className="font-bold text-alert">
                Backend model unavailable. Start the backend and run{" "}
                <code>python Model/scripts/train_static.py</code> to train/export it.
              </p>
            )}
            {backendError && <p className="font-bold text-alert">{backendError}</p>}
            <p className="text-muted">
              Hand landmarks are extracted in the browser and sent to the backend model for prediction.
            </p>
          </div>
        )}

        {engine === "landmark" && modelMissing && (
          <div className="rounded-lg border-2 border-amber bg-white p-4 text-lg" role="status">
            No trained model yet. Record signs on the{" "}
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

      <div className="space-y-4">
        <PredictionCard prediction={prediction} />
        <SentencePanel
          sentence={sentence}
          onSpace={() => setSentence((s) => applyToken(s, "SPACE"))}
          onDelete={() => setSentence((s) => applyToken(s, "DELETE"))}
          onClear={() => setSentence("")}
          onSpeak={() => speak(sentence)}
        />
      </div>
    </div>
  );
}
