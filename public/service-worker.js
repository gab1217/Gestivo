const CACHE_NAME = "gestivo-v5";
const BASE_PATH = new URL("./", self.location.href).pathname.replace(/\/$/, "");
const ROOT_PATH = `${BASE_PATH}/`;
const asset = (path) => `${BASE_PATH}${path}`;
const APP_SHELL = [
  ROOT_PATH,
  asset("/recognizer/"),
  asset("/manifest.webmanifest"),
  asset("/gestivo-logo.png")
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
  const url = new URL(event.request.url);
  const isHeavyStaticAsset = url.origin === self.location.origin && [
    "/models/", "/mediapipe-wasm/", "/tflite-wasm/", "/vendor/", "/assets/"
  ].some((folder) => url.pathname.includes(folder));
  if (isHeavyStaticAsset) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request).then((response) => {
        if (response.ok) caches.open(CACHE_NAME).then((cache) => cache.put(event.request, response.clone()));
        return response;
      }))
    );
    return;
  }
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
