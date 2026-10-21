import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Play, Square, Trash2 } from "lucide-react";
import CameraView from "../components/camera/CameraView";
import { extractFeatures, FEATURE_DIM } from "../../../Model/ml/features/featureExtractor";
import type { Point } from "../../../Model/ml/features/normalize";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const GOAL_PER_CLASS = 500; // README target: 500-1000 samples per class
const SAMPLE_INTERVAL_MS = 100; // ~10 samples per second
const COUNTDOWN_MS = 2000;

interface Sample {
  participant: string;
  label: string;
  features: number[];
}

type Phase = "idle" | "countdown" | "recording";

interface Recording {
  mode: Phase;
  startAt: number;
  lastAt: number;
  taken: number;
  target: number;
  label: string;
  participant: string;
}

const IDLE: Recording = {
  mode: "idle",
  startAt: 0,
  lastAt: 0,
  taken: 0,
  target: 0,
  label: "",
  participant: "",
};

export default function CollectPage() {
  const [participant, setParticipant] = useState("P01");
  const [classes, setClasses] = useState<string[]>(ALPHABET);
  const [label, setLabel] = useState("A");
  const [takeSize, setTakeSize] = useState(50);
  const [autoAdvance, setAutoAdvance] = useState(true);
  const [newClass, setNewClass] = useState("");

  const [phase, setPhase] = useState<Phase>("idle");
  const [taken, setTaken] = useState(0);
  const [handsSeen, setHandsSeen] = useState(0);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [total, setTotal] = useState(0);
  const [savedUpTo, setSavedUpTo] = useState(0);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainStatus, setTrainStatus] = useState<string | null>(null);

  const samples = useRef<Sample[]>([]);
  const rec = useRef<Recording>({ ...IDLE });
  const classesRef = useRef(classes);
  classesRef.current = classes;

  const stop = useCallback((advance: boolean) => {
    const finishedLabel = rec.current.label;
    rec.current = { ...IDLE };
    setPhase("idle");
    if (advance) {
      const list = classesRef.current;
      const next = list[list.indexOf(finishedLabel) + 1];
      if (next) setLabel(next);
    }
  }, []);

  const handleFrame = useCallback(
    (hands: Point[][], aspect: number) => {
      setHandsSeen(hands.length);
      const r = rec.current;
      if (r.mode === "idle") return;
      const now = performance.now();
      if (now < r.startAt) return;
      if (r.mode === "countdown") {
        r.mode = "recording";
        setPhase("recording");
      }
      const features = extractFeatures(hands, aspect);
      if (!features || now - r.lastAt < SAMPLE_INTERVAL_MS) return;
      r.lastAt = now;
      samples.current.push({ participant: r.participant, label: r.label, features });
      r.taken++;
      setTaken(r.taken);
      setTotal(samples.current.length);
      setCounts((c) => ({ ...c, [r.label]: (c[r.label] ?? 0) + 1 }));
      if (r.taken >= r.target) stop(autoAdvanceRef.current);
    },
    [stop],
  );

  const autoAdvanceRef = useRef(autoAdvance);
  autoAdvanceRef.current = autoAdvance;

  const toggle = useCallback(() => {
    if (rec.current.mode !== "idle") {
      stop(false);
      return;
    }
    if (!participant.trim() || !label) return;
    rec.current = {
      mode: "countdown",
      startAt: performance.now() + COUNTDOWN_MS,
      lastAt: 0,
      taken: 0,
      target: takeSize || 200,
      label,
      participant: participant.trim(),
    };
    setTaken(0);
    setPhase("countdown");
  }, [label, participant, stop, takeSize]);

  // Space bar starts/stops recording (ignored while typing or on a focused button)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (e.code !== "Space" || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)) return;
      e.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  // Warn before closing the tab with samples that were never downloaded
  const unsaved = total - savedUpTo;
  useEffect(() => {
    if (unsaved <= 0) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unsaved]);

  const getCsvString = () => {
    const header = ["participant_id", "label", ...Array.from({ length: FEATURE_DIM }, (_, i) => `f${i}`)];
    const rows = samples.current.map((s) =>
      [s.participant, s.label, ...s.features.map((v) => v.toFixed(5))].join(","),
    );
    return [header.join(","), ...rows].join("\n");
  };

  const saveToBackend = async () => {
    if (samples.current.length === 0) return;
    try {
      setSaveStatus("Saving directly to project training folder…");
      const res = await fetch("/api/dataset/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          participant: participant.trim() || "P01",
          csv_content: getCsvString(),
        }),
      });
      if (!res.ok) throw new Error("Could not save dataset to backend.");
      const data = (await res.json()) as { ok: boolean; file: string };
      setSaveStatus(`Saved as ${data.file}! You can now train.`);
      setSavedUpTo(samples.current.length);
    } catch (e) {
      setSaveStatus(e instanceof Error ? e.message : "Error saving dataset");
    }
  };

  const trainModel = async () => {
    try {
      setIsTraining(true);
      setTrainStatus("Training MLP model on your landmarks… (about 5-10s)");
      const res = await fetch("/api/train/run", { method: "POST" });
      if (!res.ok) {
        const err = (await res.json()) as { detail?: string };
        throw new Error(err.detail || "Training failed");
      }
      setTrainStatus("Training complete! Model exported to browser. Switch to Translate tab to use it!");
    } catch (e) {
      setTrainStatus(e instanceof Error ? e.message : "Error running training");
    } finally {
      setIsTraining(false);
    }
  };

  const downloadCsv = () => {
    if (samples.current.length === 0) return;
    const blob = new Blob([getCsvString()], { type: "text/csv" });
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${participant.trim() || "participant"}_${stamp}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
    setSavedUpTo(samples.current.length);
  };

  const clearLabel = () => {
    if (!window.confirm(`Delete all ${counts[label] ?? 0} samples for "${label}" in this session?`)) return;
    samples.current = samples.current.filter((s) => s.label !== label);
    setCounts((c) => ({ ...c, [label]: 0 }));
    setTotal(samples.current.length);
    setSavedUpTo(0);
  };

  const addClass = () => {
    const c = newClass.trim().toUpperCase();
    if (c && !classes.includes(c)) setClasses([...classes, c]);
    setNewClass("");
    if (c) setLabel(c);
  };

  const recording = phase !== "idle";
  const field = "w-full rounded-lg border-2 border-line bg-white px-3 py-2 text-lg";

  return (
    <div className="grid gap-6 lg:grid-cols-[1.5fr_1fr]">
      <div className="space-y-3">
        <CameraView onFrame={handleFrame} />
        <p className="text-lg" role="status">
          {handsSeen === 0 ? "No hand visible" : `${handsSeen} hand${handsSeen > 1 ? "s" : ""} visible`}
          {phase === "countdown" && " · Get ready…"}
          {phase === "recording" && ` · Recording "${label}" ${taken}/${takeSize}`}
        </p>
        <div className="h-3 overflow-hidden rounded-full bg-line">
          <div
            className="h-full bg-signal"
            style={{ width: `${recording ? Math.min(100, (taken / takeSize) * 100) : 0}%` }}
          />
        </div>
      </div>

      <div className="space-y-4 rounded-lg bg-white p-5">
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-lg font-bold">
            Participant ID
            <input className={field} value={participant} onChange={(e) => setParticipant(e.target.value)} />
          </label>
          <label className="block text-lg font-bold">
            Samples per take
            <input
              className={field}
              type="number"
              min={20}
              max={1000}
              step={20}
              value={takeSize || ""}
              onChange={(e) => setTakeSize(Number(e.target.value) || 200)}
            />
          </label>
        </div>

        <fieldset>
          <legend className="text-lg font-bold">Sign to record</legend>
          <div className="mt-2 grid grid-cols-6 gap-2 sm:grid-cols-7">
            {classes.map((c) => {
              const n = counts[c] ?? 0;
              const done = n >= GOAL_PER_CLASS;
              return (
                <button
                  key={c}
                  disabled={recording}
                  onClick={(e) => {
                    e.currentTarget.blur();
                    setLabel(c);
                  }}
                  aria-pressed={label === c}
                  title={`${n} samples`}
                  className={`rounded-md border-2 px-1 py-1 text-center font-bold ${
                    label === c ? "border-ink bg-ink text-white" : done ? "border-signal bg-signal/15" : "border-line"
                  }`}
                >
                  <div className="text-lg leading-tight">{c}</div>
                  <div className="text-xs">{n}</div>
                </button>
              );
            })}
          </div>
          <div className="mt-3 flex gap-2">
            <input
              className={field}
              placeholder="Add a word or sign (e.g. HELLO)"
              value={newClass}
              onChange={(e) => setNewClass(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addClass()}
            />
            <button className="rounded-lg border-2 border-ink px-4 font-bold" onClick={addClass}>
              Add
            </button>
          </div>
        </fieldset>

        <label className="flex items-center gap-2 text-lg">
          <input type="checkbox" checked={autoAdvance} onChange={(e) => setAutoAdvance(e.target.checked)} className="size-5" />
          Move to the next sign after each take
        </label>

        <button
          onClick={toggle}
          className={`flex w-full items-center justify-center gap-2 rounded-lg px-4 py-4 text-xl font-bold text-white ${
            recording ? "bg-alert" : "bg-signal hover:bg-signal-dark"
          }`}
        >
          {recording ? <Square aria-hidden /> : <Play aria-hidden />}
          {recording ? "Stop" : `Record "${label}" (Space)`}
        </button>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={saveToBackend}
            disabled={total === 0}
            className="flex items-center justify-center gap-2 rounded-lg bg-ink px-4 py-3 text-lg font-bold text-white hover:bg-ink/80 disabled:opacity-40"
          >
            💾 Save to Project
          </button>
          <button
            onClick={trainModel}
            disabled={isTraining || total === 0}
            className="flex items-center justify-center gap-2 rounded-lg bg-signal px-4 py-3 text-lg font-bold text-white hover:bg-signal-dark disabled:opacity-40"
          >
            ⚡ {isTraining ? "Training…" : "Train Model Now"}
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={downloadCsv}
            disabled={total === 0}
            className="flex items-center justify-center gap-2 rounded-lg border-2 border-ink px-4 py-3 text-lg font-bold disabled:opacity-40"
          >
            <Download aria-hidden /> Download CSV
          </button>
          <button
            onClick={clearLabel}
            disabled={recording || !(counts[label] > 0)}
            className="flex items-center justify-center gap-2 rounded-lg border-2 border-line px-4 py-3 text-lg font-bold disabled:opacity-40"
          >
            <Trash2 aria-hidden /> Clear “{label}”
          </button>
        </div>

        {saveStatus && (
          <p className="rounded-md bg-signal/10 p-2 text-sm font-semibold text-signal">
            {saveStatus}
          </p>
        )}
        {trainStatus && (
          <p className="rounded-md bg-amber/10 p-2 text-sm font-semibold text-ink">
            {trainStatus}
          </p>
        )}

        <p className={unsaved > 0 ? "text-lg font-bold text-alert" : "text-lg text-muted"}>
          {total} samples in this session{unsaved > 0 ? ` · ${unsaved} unsaved` : ""}
        </p>
      </div>
    </div>
  );
}
