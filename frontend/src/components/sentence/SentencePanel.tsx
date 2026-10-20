import { Delete, Eraser, Space, Volume2 } from "lucide-react";
import { isSpeechSupported } from "../../services/speechService";

interface Props {
  sentence: string;
  onSpace: () => void;
  onDelete: () => void;
  onClear: () => void;
  onSpeak: () => void;
}

const btn =
  "flex items-center justify-center gap-2 rounded-lg border-2 border-ink px-4 py-3 text-lg font-bold hover:bg-line";

export default function SentencePanel({ sentence, onSpace, onDelete, onClear, onSpeak }: Props) {
  return (
    <section className="rounded-lg bg-white p-5">
      <h2 className="text-lg font-bold text-muted">Sentence</h2>
      <div
        className="my-3 min-h-28 break-words rounded-lg border-2 border-line p-4 text-4xl font-bold"
        aria-live="polite"
      >
        {sentence || <span className="text-muted/60">Signs you make appear here</span>}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <button className={btn} onClick={onSpace}>
          <Space aria-hidden /> Space
        </button>
        <button className={btn} onClick={onDelete}>
          <Delete aria-hidden /> Delete
        </button>
        <button className={btn} onClick={onClear}>
          <Eraser aria-hidden /> Clear
        </button>
        <button
          className="flex items-center justify-center gap-2 rounded-lg bg-signal px-4 py-3 text-lg font-bold text-white hover:bg-signal-dark disabled:opacity-50"
          onClick={onSpeak}
          disabled={!sentence.trim() || !isSpeechSupported()}
        >
          <Volume2 aria-hidden /> Speak
        </button>
      </div>
      {!isSpeechSupported() && (
        <p className="mt-3 text-muted">Speech is not supported in this browser. Try Chrome or Edge.</p>
      )}
    </section>
  );
}
