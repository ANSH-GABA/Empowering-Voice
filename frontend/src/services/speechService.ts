export const isSpeechSupported = () =>
  typeof window !== "undefined" && "speechSynthesis" in window;

/** Speaks text with the browser's built-in voices (no server needed). */
export function speak(text: string, lang = "en-IN"): boolean {
  if (!isSpeechSupported() || !text.trim()) return false;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text.trim());
  utterance.lang = lang; // try "hi-IN" for Hindi
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
  return true;
}

export function stopSpeaking() {
  if (isSpeechSupported()) window.speechSynthesis.cancel();
}
