import { useEffect, useRef, useState } from "react";
import { DrawingUtils, HandLandmarker } from "@mediapipe/tasks-vision";
import { getHandLandmarker } from "../../ml/mediapipe/handLandmarker";
import type { Point } from "../../ml/features/normalize";

type Status = "starting" | "loading-model" | "running" | "error";

interface Props {
  /** Called for every processed video frame. `hands` is empty when none are visible. */
  onFrame: (hands: Point[][], aspect: number, video: HTMLVideoElement) => void;
}

function describeError(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError")
    return "Camera permission was blocked. Allow camera access in the address bar and reload.";
  if (name === "NotFoundError") return "No camera was found on this device.";
  if (name === "NotReadableError")
    return "The camera is in use by another app. Close it and reload.";
  return e instanceof Error ? e.message : "Could not start the camera.";
}

export default function CameraView({ onFrame }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  const [status, setStatus] = useState<Status>("starting");
  const [error, setError] = useState("");
  const [fps, setFps] = useState(0);

  useEffect(() => {
    let stopped = false;
    let raf = 0;
    let stream: MediaStream | null = null;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        });
        if (stopped) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current!;
        video.srcObject = stream;
        await video.play();

        setStatus("loading-model");
        const landmarker = await getHandLandmarker();
        if (stopped) return;
        setStatus("running");

        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d")!;
        const drawing = new DrawingUtils(ctx);
        let lastTime = -1;
        let frames = 0;
        let fpsStart = performance.now();

        const loop = () => {
          if (stopped) return;
          if (video.readyState >= 2 && video.currentTime !== lastTime) {
            lastTime = video.currentTime;
            if (canvas.width !== video.videoWidth) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
            }
            const result = landmarker.detectForVideo(video, performance.now());
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (const hand of result.landmarks) {
              drawing.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
                color: "#e8a317",
                lineWidth: 4,
              });
              drawing.drawLandmarks(hand, {
                color: "#ffffff",
                fillColor: "#0b7a75",
                lineWidth: 2,
                radius: 4,
              });
            }
            onFrameRef.current(result.landmarks, video.videoWidth / video.videoHeight, video);

            frames++;
            const now = performance.now();
            if (now - fpsStart >= 1000) {
              setFps(frames);
              frames = 0;
              fpsStart = now;
            }
          }
          raf = requestAnimationFrame(loop);
        };
        loop();
      } catch (e) {
        if (stopped) return;
        setStatus("error");
        setError(describeError(e));
      }
    }

    start();
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-ink">
      {/* Mirrored so it feels like a mirror. Landmarks are computed on the unmirrored frame. */}
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
        playsInline
        muted
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
      />
      {status !== "running" && (
        <div
          className="absolute inset-0 flex items-center justify-center bg-ink/80 p-6 text-center text-xl text-white"
          role={status === "error" ? "alert" : "status"}
        >
          {status === "starting" && "Starting camera…"}
          {status === "loading-model" && "Loading hand tracking model…"}
          {status === "error" && error}
        </div>
      )}
      {status === "running" && (
        <div className="absolute bottom-2 left-2 rounded bg-ink/70 px-2 py-1 text-sm text-white">
          {fps} FPS
        </div>
      )}
    </div>
  );
}
