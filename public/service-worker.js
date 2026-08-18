const CACHE_NAME = "gestivo-v1";
const BASE_PATH = new URL("./", self.location.href).pathname.replace(/\/$/, "");
const ROOT_PATH = `${BASE_PATH}/`;
const asset = (path) => `${BASE_PATH}${path}`;
const APP_SHELL = [
  ROOT_PATH,
  asset("/manifest.webmanifest"),
  asset("/gestivo-logo.png"),
  asset("/models/fsl_model.tflite"),
  asset("/models/landmark_model.tflite"),
  asset("/models/hand_landmarker.task"),
  asset("/models/labels.txt")
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request).then((cached) => cached || caches.match(ROOT_PATH)))
  );
});
