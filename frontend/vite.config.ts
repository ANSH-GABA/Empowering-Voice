import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Phase 12: forwards /api calls to the FastAPI backend during development
    proxy: { "/api": "http://localhost:8000" },
  },
});
