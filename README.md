# Gestivo

Gestivo is a Filipino Sign Language assistant that recognizes hand signs and turns them into readable text and spoken words. This repository contains the public, phone-friendly web version.

## What works

- Real-time camera recognition for the 26-letter FSL alphabet
- Hybrid image and hand-landmark AI models
- Stable-letter confirmation, repeat-letter handling, and automatic spacing
- Text replies, conversation history, and browser speech
- Installable Progressive Web App (PWA)
- Private on-device processing: camera frames are not uploaded
- Direct Android APK download from GitHub Releases

## Run locally

Install Node.js 22 or newer and pnpm, then run:

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`. Camera access requires localhost or HTTPS.

## Production check

```bash
pnpm build
pnpm lint
```

## Project structure

- `app/` — website interface and recognition logic
- `public/models/` — TensorFlow Lite and MediaPipe models
- `public/mediapipe-wasm/` — local MediaPipe runtime
- `public/tflite-wasm/` — local TensorFlow Lite runtime
- `public/manifest.webmanifest` — installable web-app metadata

## Android download

The website points to the `Gestivo-Android.apk` asset in the latest GitHub Release. Upload that exact filename when creating a release.

## Privacy

Recognition runs in the browser. Gestivo requests camera permission only when the user starts the translator and does not transmit or store camera frames.

## Research notice

Gestivo is a student research project and assistive prototype. Recognition accuracy depends on lighting, camera angle, signing position, and the training data. It is not a certified interpreting service.
