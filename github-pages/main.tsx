import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import GestivoApp from "../app/GestivoApp";
import "../app/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <GestivoApp />
  </StrictMode>,
);
