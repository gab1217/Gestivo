import type { Metadata } from "next";
import GestivoApp from "../GestivoApp";

export const metadata: Metadata = {
  title: "Gestivo Online Recognizer",
  description: "Use Gestivo's private, browser-based Filipino Sign Language recognizer and conversation workspace.",
  alternates: { canonical: "https://gab1217.github.io/Gestivo/recognizer/" },
  robots: { index: true, follow: true },
  openGraph: {
    title: "Gestivo Online Recognizer",
    description: "Recognize Filipino Sign Language letters, compose messages, and speak them from one private browser workspace.",
    images: [],
  },
  twitter: {
    card: "summary",
    title: "Gestivo Online Recognizer",
    description: "Recognize Filipino Sign Language letters, compose messages, and speak them from one private browser workspace.",
    images: [],
  },
};

export default function RecognizerPage() {
  return <GestivoApp view="recognizer" />;
}
