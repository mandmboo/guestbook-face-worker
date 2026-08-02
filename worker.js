import * as faceapi from "face-api.js";
import { Canvas, Image, ImageData, loadImage } from "canvas";
import { createClient } from "@supabase/supabase-js";
import sharp from "sharp";

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const MODEL_URL = process.env.MODEL_URL || "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";

if (!SUPABASE_URL) throw new Error("SUPABASE_URL is required");
if (!SUPABASE_SERVICE_ROLE_KEY) throw new Error("SUPABASE_SERVICE_ROLE_KEY is required");

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const POLL_INTERVAL_MS = Number(process.env.POLL_INTERVAL_MS || 3000);
const BATCH_SIZE = Math.max(1, Math.min(20, Number(process.env.BATCH_SIZE || 5)));
const TARGET_EVENT_ID = process.env.TARGET_EVENT_ID || "";

const LOAD_IMAGE_TIMEOUT_MS = 45000;
const FACE_DETECTION_TIMEOUT_MS = 120000;
const DB_TIMEOUT_MS = 60000;
const IMAGE_DOWNLOAD_TIMEOUT_MS = 45000;
const NORMALISE_TIMEOUT_MS = 90000;

const MIN_FACE_WIDTH = Number(process.env.MIN_FACE_WIDTH || 60);
const MIN_FACE_HEIGHT = Number(process.env.MIN_FACE_HEIGHT || 60);
const MIN_CONFIDENCE = Number(process.env.MIN_CONFIDENCE || 0.55);
const MAX_IMAGE_DIMENSION = 1280;
const PROCESSED_BUCKET = "processed-photos";
const MAX_ERROR_LENGTH = 1000;

let modelsLoaded = false;
let initialQueueRecoveryComplete = false;

function text(value) {
  return value == null ? "" : String(value).trim();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function withTimeout(promise, ms, label) {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutId);
  }
}

async function loadModels() {
  if (modelsLoaded) return;
  console.log("[Worker] VERSION 12 MEMORY LANE QUEUE WORKER");
  console.log("[Worker] Loading face-api models...");
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  modelsLoaded = true;
  console.log("[Worker] Models loaded");
}

async function fetchImageBuffer(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), IMAGE_DOWNLOAD_TIMEOUT_MS);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "User-Agent": "Memory Lane Railway face worker",
        Accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
      },
    });
    if (!response.ok) throw new Error(`Failed to download image: ${response.status}`);
    const buffer = Buffer.from(await response.arrayBuffer());
    if (!buffer.length) throw new Error("Downloaded image buffer was empty");
    return buffer;
  } finally {
    clearTimeout(timer);
  }
}

async function normaliseImageBuffer(buffer) {
  try {
    const metadata = await sharp(buffer, { failOn: "none" }).rotate().metadata();
    if (!metadata.width || !metadata.height) throw new Error("Could not read image dimensions");

    const jpgBuffer = await sharp(buffer, { failOn: "none", limitInputPixels: false })
      .rotate()
      .resize({
        width: MAX_IMAGE_DIMENSION,
        height: MAX_IMAGE_DIMENSION,
        fit: "inside",
        withoutEnlargement: true,
        fastShrinkOnLoad: true,
      })
      .jpeg({ quality: 88, mozjpeg: true })
      .toBuffer();

    const output = await sharp(jpgBuffer).metadata();
    console.log(`[Worker] Normalised ${metadata.width}x${metadata.height} -> ${output.width}x${output.height}`);
    return jpgBuffer;
  } catch (error) {
    throw new Error(`Sharp normalisation failed: ${error?.message || String(error)}`);
  }
}

async function prepareImage(photoUrl) {
  console.log(`[Worker] Downloading ${photoUrl}`);
  const source = await withTimeout(
    fetchImageBuffer(photoUrl),
    IMAGE_DOWNLOAD_TIMEOUT_MS + 5000,
    "fetchImageBuffer",
  );
  console.log(`[Worker] Downloaded ${source.length} bytes`);
  return withTimeout(normaliseImageBuffer(source), NORMALISE_TIMEOUT_MS, "normaliseImageBuffer");
}

async function detectFaces(imageBuffer) {
  const image = await withTimeout(loadImage(imageBuffer), LOAD_IMAGE_TIMEOUT_MS, "loadImage");
  console.log(`[Worker] Decoded image ${image.width}x${image.height}`);

  const detections = await withTimeout(
    faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors(),
    FACE_DETECTION_TIMEOUT_MS,
    "face descriptor extraction",
  );

  const faces = detections
    .map((detection, faceIndex) => ({
      face_index: faceIndex,
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
    .filter((face) =>
      face.face_width >= MIN_FACE_WIDTH
      && face.face_height >= MIN_FACE_HEIGHT
      && face.confidence >= MIN_CONFIDENCE,
    );

  console.log(`[Worker] Kept ${faces.length} of ${detections.length} detected face(s)`);
  return faces;
}

async function enqueueEligiblePhotos() {
  const { data, error } = await withTimeout(
    supabase.rpc("enqueue_photo_face_index_jobs", {
      p_event_id: TARGET_EVENT_ID || null,
    }),
    DB_TIMEOUT_MS,
    "enqueue_photo_face_index_jobs",
  );
  if (error) throw error;
  return Number(data || 0);
}

async function recoverPreviousEngineJobsOnce() {
  if (initialQueueRecoveryComplete) return;
  const now = new Date().toISOString();
  let query = supabase
    .from("photo_face_index_jobs")
    .update({
      status: "pending",
      attempts: 0,
      available_at: now,
      locked_at: null,
      completed_at: null,
      last_error: null,
      updated_at: now,
    })
    .in("status", ["failed", "processing"])
    .select("id");

  if (TARGET_EVENT_ID) query = query.eq("event_id", TARGET_EVENT_ID);
  const { data, error } = await withTimeout(query, DB_TIMEOUT_MS, "recoverPreviousEngineJobsOnce");
  if (error) throw error;
  initialQueueRecoveryComplete = true;
  console.log(`[Worker] Recovered ${(data || []).length} failed/stale queue job(s) from previous engines`);
}

async function claimJobs() {
  const { data, error } = await withTimeout(
    supabase.rpc("claim_photo_face_index_jobs", {
      p_limit: BATCH_SIZE,
      p_event_id: TARGET_EVENT_ID || null,
    }),
    DB_TIMEOUT_MS,
    "claim_photo_face_index_jobs",
  );
  if (error) throw error;
  return data || [];
}

async function clearDescriptors(job) {
  const sourceDelete = await withTimeout(
    supabase
      .from("photo_face_descriptors")
      .delete()
      .eq("event_id", job.event_id)
      .eq("source_type", job.source_type)
      .eq("source_id", String(job.source_id)),
    DB_TIMEOUT_MS,
    "clear source descriptors",
  );
  if (sourceDelete.error) throw sourceDelete.error;

  const legacyColumn = job.source_type === "guest_upload" ? "guest_upload_id" : "event_photo_id";
  const legacyDelete = await withTimeout(
    supabase
      .from("photo_face_descriptors")
      .delete()
      .eq("event_id", job.event_id)
      .eq(legacyColumn, job.source_id),
    DB_TIMEOUT_MS,
    "clear legacy descriptors",
  );
  if (legacyDelete.error) throw legacyDelete.error;
}

async function saveDescriptors(job, faces) {
  await clearDescriptors(job);
  if (!faces.length) return;

  const rows = faces.map((face) => ({
    event_id: job.event_id,
    event_photo_id: job.source_type === "event_photo" ? job.source_id : null,
    guest_upload_id: job.source_type === "guest_upload" ? job.source_id : null,
    source_type: job.source_type,
    source_id: String(job.source_id),
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
    "saveDescriptors",
  );
  if (error) throw error;
  console.log(`[Worker] Saved ${rows.length} descriptor(s) for ${job.source_type}:${job.source_id}`);
}

async function saveProcessedEventPhoto(job, buffer) {
  if (job.source_type !== "event_photo") return null;

  const path = `events/${job.event_id}/processed/${job.source_id}.jpg`;
  const { error: uploadError } = await withTimeout(
    supabase.storage.from(PROCESSED_BUCKET).upload(path, buffer, {
      contentType: "image/jpeg",
      upsert: true,
    }),
    DB_TIMEOUT_MS,
    "saveProcessedImage upload",
  );
  if (uploadError) throw uploadError;

  const { data } = supabase.storage.from(PROCESSED_BUCKET).getPublicUrl(path);
  const processedUrl = data?.publicUrl;
  if (!processedUrl) throw new Error("Failed to get processed image public URL");

  const { error: databaseError } = await withTimeout(
    supabase.from("processed_event_photos").upsert({
      event_id: job.event_id,
      event_photo_id: job.source_id,
      processed_url: processedUrl,
      storage_path: path,
      bucket_name: PROCESSED_BUCKET,
      created_at: new Date().toISOString(),
    }, { onConflict: "event_photo_id" }),
    DB_TIMEOUT_MS,
    "saveProcessedImage upsert",
  );
  if (databaseError) throw databaseError;
  return processedUrl;
}

async function updateEventPhotoStatus(job, status, faceCount = 0, errorMessage = null) {
  if (job.source_type !== "event_photo") return;
  const values = {
    processing_status: status,
    processing_error: errorMessage,
  };
  if (status === "complete") {
    values.face_count = faceCount;
    values.processed_at = new Date().toISOString();
  }
  const { error } = await withTimeout(
    supabase.from("event_photos").update(values).eq("id", job.source_id),
    DB_TIMEOUT_MS,
    "updateEventPhotoStatus",
  );
  if (error) console.warn(`[Worker] Could not update event_photos status: ${error.message}`);
}

async function markQueueComplete(job, faceCount) {
  const { error } = await withTimeout(
    supabase
      .from("photo_face_index_jobs")
      .update({
        status: "complete",
        face_count: faceCount,
        completed_at: new Date().toISOString(),
        locked_at: null,
        last_error: null,
        updated_at: new Date().toISOString(),
      })
      .eq("id", job.id)
      .eq("status", "processing"),
    DB_TIMEOUT_MS,
    "markQueueComplete",
  );
  if (error) throw error;
}

async function markQueueFailed(job, error) {
  const attempts = Number(job.attempts || 1);
  const delayMinutes = Math.min(60, Math.max(1, 2 ** Math.max(0, attempts - 1)));
  const message = text(error?.message || error).slice(0, MAX_ERROR_LENGTH) || "Unknown worker error";
  const { error: updateError } = await withTimeout(
    supabase
      .from("photo_face_index_jobs")
      .update({
        status: "failed",
        locked_at: null,
        available_at: new Date(Date.now() + delayMinutes * 60 * 1000).toISOString(),
        last_error: message,
        updated_at: new Date().toISOString(),
      })
      .eq("id", job.id),
    DB_TIMEOUT_MS,
    "markQueueFailed",
  );
  if (updateError) console.error(`[Worker] Could not mark queue job failed: ${updateError.message}`);
  await updateEventPhotoStatus(job, "failed", 0, message);
}

async function processJob(job) {
  console.log(`[Worker] Processing ${job.source_type}:${job.source_id} for event ${job.event_id}`);
  await updateEventPhotoStatus(job, "processing", 0, null);

  const normalised = await prepareImage(job.photo_url);
  await saveProcessedEventPhoto(job, normalised);
  const faces = await detectFaces(normalised);
  await saveDescriptors(job, faces);
  await markQueueComplete(job, faces.length);
  await updateEventPhotoStatus(job, "complete", faces.length, null);

  console.log(`[Worker] Completed ${job.source_type}:${job.source_id} with ${faces.length} face(s)`);
}

async function processJobSafely(job) {
  try {
    await processJob(job);
  } catch (error) {
    console.error(`[Worker] Failed ${job.source_type}:${job.source_id}`, error);
    await markQueueFailed(job, error);
  }
}

async function runCycle() {
  await loadModels();
  await recoverPreviousEngineJobsOnce();
  const queued = await enqueueEligiblePhotos();
  if (queued) console.log(`[Worker] Queued or refreshed ${queued} eligible photo(s)`);

  const jobs = await claimJobs();
  if (!jobs.length) {
    console.log("[Worker] No pending queue jobs");
    return;
  }

  console.log(`[Worker] Claimed ${jobs.length} queue job(s)`);
  for (const job of jobs) await processJobSafely(job);
}

async function main() {
  console.log("[Worker] VERSION 12 MEMORY LANE QUEUE WORKER");
  console.log(`[Worker] Batch size: ${BATCH_SIZE}`);
  console.log(`[Worker] Poll interval: ${POLL_INTERVAL_MS}ms`);
  console.log(TARGET_EVENT_ID
    ? `[Worker] Target event mode: ${TARGET_EVENT_ID}`
    : "[Worker] Processing all enabled Memory Lane events");

  while (true) {
    try {
      await runCycle();
    } catch (error) {
      console.error("[Worker] Cycle error", error);
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

main().catch((error) => {
  console.error("[Worker] Fatal startup error", error);
  process.exit(1);
});
