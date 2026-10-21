import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const resolveFromFrontend = (path: string) =>
  decodeURIComponent(new URL(path, import.meta.url).pathname).replace(/^\/([A-Za-z]:\/)/, "$1");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@mediapipe/tasks-vision": resolveFromFrontend(
        "./node_modules/@mediapipe/tasks-vision/vision_bundle.mjs",
      ),
    },
  },
  server: {
    port: 5173,
    fs: { allow: [resolveFromFrontend("../")] },
    // Phase 12: forwards /api calls to the FastAPI backend during development
    proxy: { "/api": "http://localhost:8000" },
  },
});
