import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import GestivoApp from "../app/GestivoApp";
import "../app/globals.css";

const view = window.location.pathname.replace(/\/+$/, "").endsWith("/recognizer") ? "recognizer" : "home";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <GestivoApp view={view} />
  </StrictMode>,
);
