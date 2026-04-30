import * as faceapi from "face-api.js";
import { Canvas, Image, ImageData, loadImage } from "canvas";
import { createClient } from "@supabase/supabase-js";
import sharp from "sharp";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const MODEL_URL =
  process.env.MODEL_URL || "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";

if (!SUPABASE_URL) throw new Error("SUPABASE_URL is required");
if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("SUPABASE_SERVICE_ROLE_KEY is required");

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

const POLL_INTERVAL_MS = Number(process.env.POLL_INTERVAL_MS || 3000);
const BATCH_SIZE = Number(process.env.BATCH_SIZE || 5);
const TARGET_EVENT_ID = process.env.TARGET_EVENT_ID || "";

const LOAD_IMAGE_TIMEOUT_MS = 45000;
const FACE_DETECTION_TIMEOUT_MS = 60000;
const DB_TIMEOUT_MS = 60000;
const IMAGE_DOWNLOAD_TIMEOUT_MS = 45000;
const NORMALISE_TIMEOUT_MS = 90000;

const MIN_FACE_WIDTH = Number(process.env.MIN_FACE_WIDTH || 60);
const MIN_FACE_HEIGHT = Number(process.env.MIN_FACE_HEIGHT || 60);
const MIN_CONFIDENCE = Number(process.env.MIN_CONFIDENCE || 0.55);

const MAX_IMAGE_DIMENSION = 1280;
const PROCESSED_BUCKET = "processed-photos";

let modelsLoaded = false;

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

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

async function fetchImageBuffer(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), IMAGE_DOWNLOAD_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        Accept: "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to download image: ${response.status}`);
    }

    const contentType = response.headers.get("content-type") || "";
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    if (!buffer.length) {
      throw new Error("Downloaded image buffer was empty");
    }

    return { buffer, contentType };
  } finally {
    clearTimeout(timer);
  }
}

async function normaliseImageBuffer(buffer) {
  try {
    const metadata = await sharp(buffer, { failOn: "none" }).rotate().metadata();

    if (!metadata.width || !metadata.height) {
      throw new Error("Could not read image dimensions");
    }

    const jpgBuffer = await sharp(buffer, {
      failOn: "none",
      limitInputPixels: false,
    })
      .rotate()
      .resize({
        width: MAX_IMAGE_DIMENSION,
        height: MAX_IMAGE_DIMENSION,
        fit: "inside",
        withoutEnlargement: true,
        fastShrinkOnLoad: true,
      })
      .jpeg({
        quality: 88,
        mozjpeg: true,
      })
      .toBuffer();

    const outputMeta = await sharp(jpgBuffer).metadata();

    console.log(
      `[Worker] Normalised image ${metadata.width}x${metadata.height} -> ${outputMeta.width}x${outputMeta.height}`
    );

    return {
      buffer: jpgBuffer,
      width: outputMeta.width || metadata.width,
      height: outputMeta.height || metadata.height,
      contentType: "image/jpeg",
    };
  } catch (error) {
    throw new Error(`Sharp normalisation failed: ${error?.message || String(error)}`);
  }
}

async function loadModels() {
  if (modelsLoaded) return;

  console.log("[Worker] VERSION 11 PRIORITISED EVENT PROCESSING");
  console.log("[Worker] Loading face-api models...");

  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);

  modelsLoaded = true;
  console.log("[Worker] Models loaded");
}

async function recoverStuckProcessingPhotos() {
  const { error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "pending",
        processing_error: null,
      })
      .eq("processing_status", "processing"),
    DB_TIMEOUT_MS,
    "recoverStuckProcessingPhotos"
  );

  if (error) {
    console.warn("[Worker] Could not recover stuck processing photos:", error.message);
  }
}

async function getPendingPhotos() {
  console.log("[Worker] Checking for pending photos...");

  let query = supabase
    .from("event_photos")
    .select("id, event_id, storage_url, processing_status, created_at")
    .eq("processing_status", "pending")
    .not("storage_url", "is", null);

  if (TARGET_EVENT_ID) {
    query = query.eq("event_id", TARGET_EVENT_ID);
  }

  const { data, error } = await withTimeout(
    query.order("created_at", { ascending: false }).limit(BATCH_SIZE),
    DB_TIMEOUT_MS,
    "getPendingPhotos"
  );

  if (error) throw error;

  return data || [];
}

async function markProcessing(photoId) {
  const { data, error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "processing",
        processing_error: null,
      })
      .eq("id", photoId)
      .eq("processing_status", "pending")
      .select("id")
      .maybeSingle(),
    DB_TIMEOUT_MS,
    "markProcessing"
  );

  if (error) throw error;

  return !!data;
}

async function markComplete(photoId, faceCount) {
  const { error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "complete",
        face_count: faceCount,
        processed_at: new Date().toISOString(),
        processing_error: null,
      })
      .eq("id", photoId),
    DB_TIMEOUT_MS,
    "markComplete"
  );

  if (error) throw error;
}

async function markFailed(photoId, message) {
  const { error } = await withTimeout(
    supabase
      .from("event_photos")
      .update({
        processing_status: "failed",
        processing_error: String(message || "Unknown error").slice(0, 1000),
      })
      .eq("id", photoId),
    DB_TIMEOUT_MS,
    "markFailed"
  );

  if (error) throw error;
}

async function clearExistingDescriptors(photoId) {
  const { error } = await withTimeout(
    supabase.from("photo_face_descriptors").delete().eq("event_photo_id", photoId),
    DB_TIMEOUT_MS,
    "clearExistingDescriptors"
  );

  if (error) throw error;
}

async function clearExistingProcessedPhoto(photoId) {
  const { error } = await withTimeout(
    supabase.from("processed_event_photos").delete().eq("event_photo_id", photoId),
    DB_TIMEOUT_MS,
    "clearExistingProcessedPhoto"
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
    face_height: face.face_height,
  }));

  const { error } = await withTimeout(
    supabase.from("photo_face_descriptors").insert(rows),
    DB_TIMEOUT_MS,
    "saveDescriptors"
  );

  if (error) throw error;

  console.log(`[Worker] Saved ${rows.length} descriptor(s) for ${photoId}`);
}

async function saveProcessedImage(eventId, photoId, buffer) {
  const path = `events/${eventId}/processed/${photoId}.jpg`;

  const { error: uploadError } = await withTimeout(
    supabase.storage.from(PROCESSED_BUCKET).upload(path, buffer, {
      contentType: "image/jpeg",
      upsert: true,
    }),
    DB_TIMEOUT_MS,
    "saveProcessedImage upload"
  );

  if (uploadError) throw uploadError;

  const { data: publicUrlData } = supabase.storage
    .from(PROCESSED_BUCKET)
    .getPublicUrl(path);

  const processedUrl = publicUrlData?.publicUrl;

  if (!processedUrl) {
    throw new Error("Failed to get processed image public URL");
  }

  const row = {
    event_id: eventId,
    event_photo_id: photoId,
    processed_url: processedUrl,
    storage_path: path,
    bucket_name: PROCESSED_BUCKET,
    created_at: new Date().toISOString(),
  };

  const { error: dbError } = await withTimeout(
    supabase.from("processed_event_photos").upsert(row, {
      onConflict: "event_photo_id",
    }),
    DB_TIMEOUT_MS,
    "saveProcessedImage upsert"
  );

  if (dbError) throw dbError;

  console.log(`[Worker] Saved processed image for ${photoId}: ${processedUrl}`);

  return { processedUrl, path };
}

async function detectFacesFromBuffer(imageBuffer) {
  let img;

  try {
    img = await withTimeout(loadImage(imageBuffer), LOAD_IMAGE_TIMEOUT_MS, "loadImage");
  } catch (error) {
    throw new Error(
      `Image decode failed after normalisation: ${error?.message || String(error)}`
    );
  }

  console.log(`[Worker] Image loaded: ${img.width}x${img.height}`);

  const detections = await withTimeout(
    faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors(),
    FACE_DETECTION_TIMEOUT_MS,
    "face descriptor extraction"
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
        height: detection.detection.box.height,
      },
      confidence: detection.detection.score,
      face_width: Math.round(detection.detection.box.width),
      face_height: Math.round(detection.detection.box.height),
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

async function prepareImageForProcessing(imageUrl) {
  console.log(`[Worker] Downloading image: ${imageUrl}`);

  const { buffer, contentType } = await withTimeout(
    fetchImageBuffer(imageUrl),
    IMAGE_DOWNLOAD_TIMEOUT_MS + 5000,
    "fetchImageBuffer"
  );

  console.log(
    `[Worker] Image downloaded (${buffer.length} bytes, ${
      contentType || "unknown content type"
    })`
  );

  return await withTimeout(
    normaliseImageBuffer(buffer),
    NORMALISE_TIMEOUT_MS,
    "normaliseImageBuffer"
  );
}

async function processPhoto(photo) {
  if (!photo.storage_url) {
    throw new Error(`Photo ${photo.id} has no storage_url`);
  }

  console.log(`[Worker] Processing photo ${photo.id} for event ${photo.event_id}`);

  const locked = await markProcessing(photo.id);

  if (!locked) {
    console.log(`[Worker] Skipping ${photo.id}, another worker already picked it up.`);
    return;
  }

  await clearExistingDescriptors(photo.id);
  await clearExistingProcessedPhoto(photo.id);

  const normalised = await prepareImageForProcessing(photo.storage_url);

  await saveProcessedImage(photo.event_id, photo.id, normalised.buffer);

  const faces = await detectFacesFromBuffer(normalised.buffer);

  await saveDescriptors(photo.event_id, photo.id, faces);
  await markComplete(photo.id, faces.length);

  console.log(`[Worker] Marked complete ${photo.id} with ${faces.length} face(s)`);
}

async function processPhotoSafely(photo) {
  try {
    await processPhoto(photo);
  } catch (error) {
    console.error(`[Worker] Failed photo ${photo.id}:`, error);

    try {
      await markFailed(photo.id, error?.message || String(error));
    } catch (markError) {
      console.error(`[Worker] Failed to mark photo ${photo.id} as failed:`, markError);
    }
  }
}

async function runCycle() {
  await loadModels();

  await recoverStuckProcessingPhotos();

  const pending = await getPendingPhotos();

  if (!pending.length) {
    console.log("[Worker] No pending photos");
    return;
  }

  console.log(`[Worker] VERSION 11 found ${pending.length} pending photo(s)`);

  for (const photo of pending) {
    await processPhotoSafely(photo);
  }
}

async function main() {
  console.log("[Worker] VERSION 11 PRIORITISED EVENT PROCESSING");
  console.log(`[Worker] Batch size: ${BATCH_SIZE}`);
  console.log(`[Worker] Poll interval: ${POLL_INTERVAL_MS}ms`);

  if (TARGET_EVENT_ID) {
    console.log(`[Worker] Target event mode enabled: ${TARGET_EVENT_ID}`);
  } else {
    console.log("[Worker] Processing newest pending photos first.");
  }

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
