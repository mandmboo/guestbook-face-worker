import * as faceapi from "face-api.js";
import { Canvas, Image, ImageData, loadImage } from "canvas";
import { createClient } from "@supabase/supabase-js";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const MODEL_URL =
  process.env.MODEL_URL || "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";

if (!SUPABASE_URL) {
  throw new Error("SUPABASE_URL is required");
}

if (!SUPABASE_SERVICE_ROLE_KEY) {
  throw new Error("SUPABASE_SERVICE_ROLE_KEY is required");
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

const POLL_INTERVAL_MS = 5000;
const BATCH_SIZE = 3;
const LOAD_IMAGE_TIMEOUT_MS = 20000;
const FACE_DETECTION_TIMEOUT_MS = 45000;
const DB_TIMEOUT_MS = 10000;
const MIN_FACE_WIDTH = 90;
const MIN_FACE_HEIGHT = 90;
const MIN_CONFIDENCE = 0.75;

let modelsLoaded = false;

async function withTimeout(promise, ms, label) {
  let timeoutId;

  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function loadModels() {
  if (modelsLoaded) return;

  console.log("[Worker] VERSION 3 REALLY NEW BUILD");
  console.log("[Worker] Loading face-api models...");

  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
  ]);

  modelsLoaded = true;
  console.log("[Worker] Models loaded");
}

async function getPendingPhotos() {
  console.log("[Worker] Checking for pending photos...");

  const { data, error } = await withTimeout(
    supabase
      .from("event_photos")
      .select("id, event_id, storage_url, processing_status")
      .eq("processing_status", "pending")
      .not("storage_url", "is", null)
      .order("created_at", { ascending: true })
      .limit(BATCH_SIZE),
    DB_TIMEOUT_MS,
    "getPendingPhotos"
  );

  if (error) throw error;

  return data || [];
}

async function markProcessing(photoId) {
  console.log(`[Worker] About to mark processing ${photoId}`);

  const { error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "processing",
        processing_error: null
      })
      .eq("id", photoId),
    DB_TIMEOUT_MS,
    "markProcessing"
  );

  if (error) throw error;
}

async function markComplete(photoId, faceCount) {
  console.log(`[Worker] About to mark complete ${photoId}`);

  const { error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "complete",
        face_count: faceCount,
        processed_at: new Date().toISOString(),
        processing_error: null
      })
      .eq("id", photoId),
    DB_TIMEOUT_MS,
    "markComplete"
  );

  if (error) throw error;
}

async function markFailed(photoId, message) {
  console.log(`[Worker] About to mark failed ${photoId}: ${message}`);

  const { error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "failed",
        processing_error: String(message || "Unknown error").slice(0, 1000)
      })
      .eq("id", photoId),
    DB_TIMEOUT_MS,
    "markFailed"
  );

  if (error) throw error;
}

async function clearExistingDescriptors(photoId) {
  console.log(`[Worker] About to clear descriptors ${photoId}`);

  const { error } = await withTimeout(
    supabase
      .from("photo_face_descriptors")
      .delete()
      .eq("event_photo_id", photoId),
    DB_TIMEOUT_MS,
    "clearExistingDescriptors"
  );

  if (error) throw error;
}

async function saveDescriptors(eventId, photoId, faces) {
  if (!faces.length) {
    console.log(`[Worker] No descriptors to save for ${photoId}`);
    return;
  }

  const rows = faces.map((face) => ({
    event_id: eventId,
    event_photo_id: photoId,
    face_index: face.face_index,
    descriptor: face.descriptor,
    bounding_box: face.bounding_box,
    confidence: face.confidence,
    face_width: face.face_width,
    face_height: face.face_height
  }));

  const { error } = await withTimeout(
    supabase.from("photo_face_descriptors").insert(rows),
    DB_TIMEOUT_MS,
    "saveDescriptors"
  );

  if (error) throw error;
}

async function detectFacesFromUrl(imageUrl) {
  console.log(`[Worker] Loading image: ${imageUrl}`);

  const img = await withTimeout(
    loadImage(imageUrl),
    LOAD_IMAGE_TIMEOUT_MS,
    "loadImage"
  );

  console.log(`[Worker] Image loaded: ${img.width}x${img.height}`);

  const detections = await withTimeout(
    faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors(),
    FACE_DETECTION_TIMEOUT_MS,
    "face detection"
  );

  console.log(`[Worker] Raw detections found: ${detections.length}`);

  const faces = detections
    .map((detection, index) => ({
      face_index: index,
      descriptor: Array.from(detection.descriptor),
      bounding_box: {
        x: detection.detection.box.x,
        y: detection.detection.box.y,
        width: detection.detection.box.width,
        height: detection.detection.box.height
      },
      confidence: detection.detection.score,
      face_width: Math.round(detection.detection.box.width),
      face_height: Math.round(detection.detection.box.height)
    }))
    .filter(
      (face) =>
        face.face_width >= MIN_FACE_WIDTH &&
        face.face_height >= MIN_FACE_HEIGHT &&
        face.confidence >= MIN_CONFIDENCE
    );

  console.log(`[Worker] Filtered detections kept: ${faces.length}`);

  return faces;
}

async function processPhoto(photo) {
  if (!photo.storage_url) {
    throw new Error(`Photo ${photo.id} has no storage_url`);
  }

  console.log(`[Worker] Processing photo ${photo.id}`);

  await markProcessing(photo.id);
  console.log(`[Worker] Marked processing ${photo.id}`);

  await clearExistingDescriptors(photo.id);
  console.log(`[Worker] Cleared old descriptors ${photo.id}`);

  const faces = await detectFacesFromUrl(photo.storage_url);

  await saveDescriptors(photo.event_id, photo.id, faces);
  console.log(`[Worker] Saved ${faces.length} descriptor row(s) for ${photo.id}`);

  await markComplete(photo.id, faces.length);
  console.log(`[Worker] Marked complete ${photo.id} with ${faces.length} face(s)`);
}

async function runCycle() {
  await loadModels();

  const pending = await getPendingPhotos();

  if (!pending.length) {
    console.log("[Worker] No pending photos");
    return;
  }

  console.log(`[Worker] VERSION 3 found ${pending.length} pending photo(s)`);

  for (const photo of pending) {
    try {
      await processPhoto(photo);
    } catch (error) {
      console.error(`[Worker] Failed photo ${photo.id}:`, error);

      try {
        await markFailed(photo.id, error?.message || String(error));
      } catch (markError) {
        console.error(
          `[Worker] Failed to mark photo ${photo.id} as failed:`,
          markError
        );
      }
    }
  }
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  console.log("[Worker] VERSION 3 REALLY NEW BUILD");
  console.log("[Worker] Starting face processing worker");

  while (true) {
    try {
      await runCycle();
    } catch (error) {
      console.error("[Worker] Cycle error:", error);
    }

    await sleep(POLL_INTERVAL_MS);
  }
}

main().catch((error) => {
  console.error("[Worker] Fatal startup error:", error);
  process.exit(1);
});
