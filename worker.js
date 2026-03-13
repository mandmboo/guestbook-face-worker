import * as faceapi from "face-api.js";
import { Canvas, Image, ImageData, loadImage } from "canvas";
import { createClient } from "@supabase/supabase-js";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

const MODEL_URL = process.env.MODEL_URL || "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";

const POLL_INTERVAL_MS = 5000;
const BATCH_SIZE = 3;

let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;

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
  const { data, error } = await supabase
    .from("event_photos")
    .select("id, event_id, storage_url")
    .eq("processing_status", "pending")
    .order("created_at", { ascending: true })
    .limit(BATCH_SIZE);

  if (error) throw error;
  return data || [];
}

async function markProcessing(photoId) {
  const { error } = await supabase
    .from("event_photos")
    .update({
      processing_status: "processing",
      processing_error: null
    })
    .eq("id", photoId);

  if (error) throw error;
}

async function markComplete(photoId, faceCount) {
  const { error } = await supabase
    .from("event_photos")
    .update({
      processing_status: "complete",
      face_count: faceCount,
      processed_at: new Date().toISOString(),
      processing_error: null
    })
    .eq("id", photoId);

  if (error) throw error;
}

async function markFailed(photoId, message) {
  const { error } = await supabase
    .from("event_photos")
    .update({
      processing_status: "failed",
      processing_error: message?.slice(0, 1000) || "Unknown error"
    })
    .eq("id", photoId);

  if (error) throw error;
}

async function clearExistingDescriptors(photoId) {
  const { error } = await supabase
    .from("photo_face_descriptors")
    .delete()
    .eq("event_photo_id", photoId);

  if (error) throw error;
}

async function detectFacesFromUrl(imageUrl) {
  const img = await loadImage(imageUrl);

  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors();

  return detections.map((detection, index) => ({
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
  }));
}

async function saveDescriptors(eventId, photoId, faces) {
  if (!faces.length) return;

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

  const { error } = await supabase
    .from("photo_face_descriptors")
    .insert(rows);

  if (error) throw error;
}

async function processPhoto(photo) {
  console.log(`[Worker] Processing photo ${photo.id}`);

  await markProcessing(photo.id);
  await clearExistingDescriptors(photo.id);

  const faces = await detectFacesFromUrl(photo.storage_url);
  await saveDescriptors(photo.event_id, photo.id, faces);
  await markComplete(photo.id, faces.length);

  console.log(`[Worker] Completed photo ${photo.id} with ${faces.length} face(s)`);
}

async function runCycle() {
  await loadModels();

  const pending = await getPendingPhotos();

  if (!pending.length) {
    console.log("[Worker] No pending photos");
    return;
  }

  console.log(`[Worker] Found ${pending.length} pending photo(s)`);

  for (const photo of pending) {
    try {
      await processPhoto(photo);
    } catch (error) {
      console.error(`[Worker] Failed photo ${photo.id}:`, error);
      await markFailed(photo.id, error?.message || String(error));
    }
  }
}

async function main() {
  console.log("[Worker] Starting face processing worker");

  while (true) {
    try {
      await runCycle();
    } catch (error) {
      console.error("[Worker] Cycle error:", error);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

main().catch((error) => {
  console.error("[Worker] Fatal startup error:", error);
  process.exit(1);
});
