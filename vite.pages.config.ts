import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { copyFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

export default defineConfig({
  root: "static-client",
  base: "/Gestivo/",
  publicDir: "../public",
  plugins: [
    react(),
    {
      name: "gestivo-recognizer-route",
      closeBundle() {
        const output = resolve("pages-dist");
        mkdirSync(resolve(output, "recognizer"), { recursive: true });
        copyFileSync(resolve(output, "index.html"), resolve(output, "recognizer", "index.html"));
      },
    },
  ],
  build: {
    outDir: "../pages-dist",
    emptyOutDir: true,
  },
});
