import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createWriteStream, promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { createClient } from "@supabase/supabase-js";

const PORT = Number(process.env.PORT || 3000);
const RENDER_SECRET = process.env.RENDER_SECRET || process.env.MEMORY_FILM_RENDER_SECRET || "";
const SUPABASE_URL = process.env.SUPABASE_URL || "";
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || "";
const OUTPUT_BUCKET = process.env.RENDER_OUTPUT_BUCKET || "rendered-films";
const MAX_MEDIA_ITEMS = Number(process.env.MAX_RENDER_MEDIA_ITEMS || 120);
const MAX_DOWNLOAD_BYTES = Number(process.env.MAX_RENDER_DOWNLOAD_BYTES || 450 * 1024 * 1024);
const DOWNLOAD_TIMEOUT_MS = Number(process.env.RENDER_DOWNLOAD_TIMEOUT_MS || 90000);
const JOB_RETENTION_MS = Number(process.env.RENDER_JOB_RETENTION_MS || 24 * 60 * 60 * 1000);
const MAX_CONCURRENT_RENDERS = Number(process.env.MAX_CONCURRENT_RENDERS || 1);

if (!RENDER_SECRET) console.warn("[VideoRender] RENDER_SECRET is not set. Render endpoints will reject requests.");
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) console.warn("[VideoRender] Supabase output upload variables are missing.");

const supabase = SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY ? createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) : null;
const jobs = new Map();
const queue = [];
let activeRenders = 0;
let bucketReady = false;

function json(res, status, body) {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type,x-render-secret,authorization",
  });
  res.end(JSON.stringify(body));
}

function safeMessage(error) { return String(error?.message || error || "Unknown error").slice(0, 2000); }
function nowIso() { return new Date().toISOString(); }
function updateJob(jobId, patch) { const next = { ...(jobs.get(jobId) || {}), ...patch, updatedAt: nowIso() }; jobs.set(jobId, next); return next; }
function cleanOldJobs() { const cutoff = Date.now() - JOB_RETENTION_MS; for (const [jobId, job] of jobs.entries()) { const updatedAt = Date.parse(job.updatedAt || job.createdAt || 0); if (updatedAt && updatedAt < cutoff && ["complete", "failed"].includes(job.status)) jobs.delete(jobId); } }

function isAuthed(req) {
  const provided = req.headers["x-render-secret"] || req.headers.authorization?.replace(/^Bearer\s+/i, "");
  if (!RENDER_SECRET || !provided) return false;
  const providedBuffer = Buffer.from(String(provided));
  const secretBuffer = Buffer.from(RENDER_SECRET);
  if (providedBuffer.length !== secretBuffer.length) return false;
  return crypto.timingSafeEqual(providedBuffer, secretBuffer);
}

async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw.trim() ? JSON.parse(raw) : {};
}

function slugify(value, fallback = "memory-film") { return String(value || fallback).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || fallback; }
function inferExtension(url, contentType = "") {
  const type = contentType.toLowerCase();
  if (type.includes("png")) return ".png";
  if (type.includes("webp")) return ".webp";
  if (type.includes("gif")) return ".gif";
  if (type.includes("jpeg") || type.includes("jpg")) return ".jpg";
  if (type.includes("quicktime")) return ".mov";
  if (type.includes("webm")) return ".webm";
  if (type.includes("mp4")) return ".mp4";
  if (type.includes("mpeg") || type.includes("mp3")) return ".mp3";
  if (type.includes("m4a")) return ".m4a";
  if (type.includes("aac")) return ".aac";
  if (type.includes("ogg")) return ".ogg";
  if (type.includes("wav")) return ".wav";
  try { const ext = path.extname(new URL(url).pathname).toLowerCase(); if (ext) return ext.slice(0, 10); } catch {}
  return ".bin";
}
function inferMediaType(item, contentType = "") {
  const explicit = String(item.type || item.kind || "").toLowerCase();
  if (explicit.includes("audio") || explicit.includes("voice")) return "audio";
  if (explicit.includes("video")) return "video";
  if (explicit.includes("image") || explicit.includes("photo")) return "image";
  const type = contentType.toLowerCase();
  if (type.startsWith("audio/")) return "audio";
  if (type.startsWith("video/")) return "video";
  if (type.startsWith("image/")) return "image";
  const url = String(item.url || item.storage_url || item.src || "").toLowerCase();
  if (/\.(mp3|m4a|wav|aac|ogg)(\?|#|$)/.test(url)) return "audio";
  if (/\.(mp4|mov|webm|m4v)(\?|#|$)/.test(url)) return "video";
  return "image";
}
function normaliseNumber(value, fallback, min, max) { const n = Number(value); if (!Number.isFinite(n)) return fallback; return Math.min(max, Math.max(min, n)); }
function explicitDuration(item, max = 1800) { if (item.duration == null && item.clipDuration == null) return null; return normaliseNumber(item.duration ?? item.clipDuration, null, 1, max); }

async function run(command, args, options = {}) {
  const label = options.label || command;
  const timeoutMs = options.timeoutMs || 10 * 60 * 1000;
  console.log(`[VideoRender] ${label}: ${command} ${args.join(" ")}`);
  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => { child.kill("SIGKILL"); reject(new Error(`${label} timed out after ${timeoutMs}ms`)); }, timeoutMs);
    child.stdout.on("data", (chunk) => { stdout = (stdout + chunk.toString()).slice(-12000); });
    child.stderr.on("data", (chunk) => { stderr = (stderr + chunk.toString()).slice(-20000); });
    child.on("error", (error) => { clearTimeout(timer); reject(error); });
    child.on("close", (code) => { clearTimeout(timer); if (code === 0) resolve({ stdout, stderr }); else reject(new Error(`${label} failed with code ${code}: ${stderr || stdout}`)); });
  });
}

async function probeDuration(inputPath) {
  try {
    const { stdout } = await run("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", inputPath], { label: "probe duration", timeoutMs: 30000 });
    const duration = Number.parseFloat(stdout.trim());
    return Number.isFinite(duration) && duration > 0 ? duration : null;
  } catch (error) { console.warn("[VideoRender] Could not probe duration:", safeMessage(error)); return null; }
}
async function hasAudioStream(inputPath) {
  try {
    const { stdout } = await run("ffprobe", ["-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", inputPath], { label: "probe audio", timeoutMs: 30000 });
    return stdout.toLowerCase().includes("audio");
  } catch (error) { console.warn("[VideoRender] Could not probe audio stream:", safeMessage(error)); return false; }
}

async function downloadToFile(url, destination) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal, headers: { "user-agent": "MemoryLaneVideoRender/1.0", accept: "image/*,video/*,audio/*,*/*;q=0.8" } });
    if (!response.ok || !response.body) throw new Error(`Download failed ${response.status} for ${url}`);
    const length = Number(response.headers.get("content-length") || 0);
    if (length && length > MAX_DOWNLOAD_BYTES) throw new Error(`Media file is too large (${length} bytes)`);
    let downloaded = 0;
    const writeStream = createWriteStream(destination);
    const reader = response.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      downloaded += value.byteLength;
      if (downloaded > MAX_DOWNLOAD_BYTES) throw new Error(`Media file exceeded max download size (${MAX_DOWNLOAD_BYTES} bytes)`);
      if (!writeStream.write(Buffer.from(value))) await new Promise((resolve) => writeStream.once("drain", resolve));
    }
    await new Promise((resolve, reject) => { writeStream.end(resolve); writeStream.on("error", reject); });
    return { contentType: response.headers.get("content-type") || "", bytes: downloaded };
  } finally { clearTimeout(timer); }
}

function dimensionsForPayload(payload) {
  const aspect = String(payload.aspectRatio || payload.aspect || "vertical").toLowerCase();
  const width = normaliseNumber(payload.width, aspect.includes("square") ? 1080 : aspect.includes("landscape") ? 1920 : 1080, 320, 3840);
  const height = normaliseNumber(payload.height, aspect.includes("square") ? 1080 : aspect.includes("landscape") ? 1080 : 1920, 320, 3840);
  return { width: width % 2 === 0 ? width : width + 1, height: height % 2 === 0 ? height : height + 1 };
}
function buildVideoFilter(width, height, fps, duration, fade = false) {
  const base = [`scale=${width}:${height}:force_original_aspect_ratio=decrease`, `pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2:color=black`, "setsar=1", "format=yuv420p"];
  if (fade && duration >= 2) { base.push("fade=t=in:st=0:d=0.35"); base.push(`fade=t=out:st=${Math.max(0, duration - 0.45).toFixed(2)}:d=0.35`); }
  base.push(`fps=${fps}`);
  return base.join(",");
}

async function renderImageSegment(inputPath, outputPath, options) {
  const { width, height, fps, duration } = options;
  await run("ffmpeg", [
    "-y", "-loop", "1", "-t", String(duration), "-i", inputPath,
    "-f", "lavfi", "-t", String(duration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-vf", buildVideoFilter(width, height, fps, duration, true),
    "-map", "0:v:0", "-map", "1:a:0", "-shortest",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", outputPath,
  ], { label: "render image segment" });
}
async function renderVideoSegment(inputPath, outputPath, options) {
  const { width, height, fps, duration } = options;
  const filter = buildVideoFilter(width, height, fps, duration, false);
  const hasAudio = await hasAudioStream(inputPath);
  if (hasAudio) {
    const args = ["-y", "-i", inputPath];
    if (duration) args.push("-t", String(duration));
    args.push("-vf", filter, "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", "-shortest", "-movflags", "+faststart", outputPath);
    await run("ffmpeg", args, { label: "render video segment with audio", timeoutMs: 30 * 60 * 1000 });
    return;
  }
  const silentDuration = duration || await probeDuration(inputPath) || 10;
  await run("ffmpeg", [
    "-y", "-i", inputPath, "-f", "lavfi", "-t", String(silentDuration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    "-t", String(silentDuration), "-vf", filter, "-map", "0:v:0", "-map", "1:a:0", "-shortest",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", outputPath,
  ], { label: "render video segment with silent audio", timeoutMs: 30 * 60 * 1000 });
}
async function renderAudioSegment(inputPath, outputPath, options) {
  const { width, height, fps, duration } = options;
  const args = ["-y", "-f", "lavfi", "-i", `color=c=0x111827:s=${width}x${height}:r=${fps}`, "-i", inputPath];
  if (duration) args.push("-t", String(duration));
  args.push("-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", outputPath);
  await run("ffmpeg", args, { label: "render audio segment", timeoutMs: 30 * 60 * 1000 });
}
async function mixBackgroundMusic(joinedPath, audioPath, outputPath, volume = 0.18) {
  const filmDuration = await probeDuration(joinedPath);
  const fadeStart = Math.max(0, (filmDuration || 0) - 2.5).toFixed(2);
  const musicVolume = normaliseNumber(volume, 0.18, 0.02, 0.7);
  const bgChain = filmDuration ? `volume=${musicVolume},afade=t=out:st=${fadeStart}:d=2` : `volume=${musicVolume}`;
  await run("ffmpeg", [
    "-y", "-i", joinedPath, "-stream_loop", "-1", "-i", audioPath,
    "-filter_complex", `[0:a]volume=1.0[main];[1:a]${bgChain}[bg];[main][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]`,
    "-map", "0:v:0", "-map", "[aout]", "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", outputPath,
  ], { label: "mix background music", timeoutMs: 30 * 60 * 1000 });
}

async function ensureBucket() {
  if (bucketReady) return;
  if (!supabase) throw new Error("Supabase is not configured for video output uploads");
  const { data: existing, error: listError } = await supabase.storage.listBuckets();
  if (listError) throw listError;
  if (!existing?.some((bucket) => bucket.name === OUTPUT_BUCKET)) {
    const { error: createError } = await supabase.storage.createBucket(OUTPUT_BUCKET, { public: true, fileSizeLimit: MAX_DOWNLOAD_BYTES, allowedMimeTypes: ["video/mp4"] });
    if (createError && !String(createError.message || "").toLowerCase().includes("already")) throw createError;
  }
  bucketReady = true;
}
async function uploadOutput(eventId, jobId, outputPath, outputName) {
  await ensureBucket();
  const buffer = await fs.readFile(outputPath);
  const storagePath = `events/${slugify(eventId, "event")}/films/${jobId}-${slugify(outputName, "memory-film")}.mp4`;
  const { error: uploadError } = await supabase.storage.from(OUTPUT_BUCKET).upload(storagePath, buffer, { contentType: "video/mp4", cacheControl: "3600", upsert: true });
  if (uploadError) throw uploadError;
  const { data } = supabase.storage.from(OUTPUT_BUCKET).getPublicUrl(storagePath);
  if (!data?.publicUrl) throw new Error("Could not create public MP4 URL");
  return { url: data.publicUrl, bucket: OUTPUT_BUCKET, path: storagePath, bytes: buffer.length };
}
function normaliseMedia(payload) { const media = Array.isArray(payload.media) ? payload.media : Array.isArray(payload.items) ? payload.items : []; return media.map((item) => ({ ...item, url: item.url || item.storage_url || item.storageUrl || item.src })).filter((item) => typeof item.url === "string" && item.url.startsWith("http")).slice(0, MAX_MEDIA_ITEMS); }

async function renderJob(jobId) {
  const job = jobs.get(jobId);
  const payload = job.payload;
  const eventId = payload.eventId || payload.event_id || "event";
  const media = normaliseMedia(payload);
  if (!media.length) throw new Error("No renderable media URLs were provided");
  const { width, height } = dimensionsForPayload(payload);
  const fps = normaliseNumber(payload.fps, 30, 15, 60);
  const defaultPhotoDuration = normaliseNumber(payload.durationPerPhoto || payload.photoDuration, 4, 1.5, 12);
  const outputName = payload.outputName || payload.title || "memory-film";
  const workDir = await fs.mkdtemp(path.join(tmpdir(), `memory-film-${jobId}-`));
  try {
    const segmentPaths = [];
    for (let index = 0; index < media.length; index += 1) {
      const item = media[index];
      updateJob(jobId, { status: "processing", progress: Math.round((index / media.length) * 70), currentStep: `Preparing media ${index + 1} of ${media.length}` });
      const probePath = path.join(workDir, `input-${index}.bin`);
      const downloaded = await downloadToFile(item.url, probePath);
      const mediaType = inferMediaType(item, downloaded.contentType);
      const inputPath = path.join(workDir, `input-${index}${inferExtension(item.url, downloaded.contentType)}`);
      await fs.rename(probePath, inputPath);
      const segmentPath = path.join(workDir, `segment-${String(index).padStart(4, "0")}.mp4`);
      const duration = mediaType === "image" ? normaliseNumber(item.duration || defaultPhotoDuration, defaultPhotoDuration, 1, 20) : explicitDuration(item);
      if (mediaType === "video") await renderVideoSegment(inputPath, segmentPath, { width, height, fps, duration });
      else if (mediaType === "audio") await renderAudioSegment(inputPath, segmentPath, { width, height, fps, duration });
      else await renderImageSegment(inputPath, segmentPath, { width, height, fps, duration });
      segmentPaths.push(segmentPath);
    }
    updateJob(jobId, { status: "processing", progress: 76, currentStep: "Joining film sections" });
    const concatListPath = path.join(workDir, "concat.txt");
    await fs.writeFile(concatListPath, segmentPaths.map((segmentPath) => `file '${segmentPath.replace(/'/g, "'\\''")}'`).join("\n"), "utf8");
    const joinedPath = path.join(workDir, "joined.mp4");
    await run("ffmpeg", ["-y", "-f", "concat", "-safe", "0", "-i", concatListPath, "-c", "copy", joinedPath], { label: "concat segments", timeoutMs: 30 * 60 * 1000 });
    updateJob(jobId, { status: "processing", progress: 88, currentStep: "Finalising MP4" });
    const outputPath = path.join(workDir, "final.mp4");
    const audioUrl = payload.audioUrl || payload.musicUrl || payload.soundtrackUrl || "";
    if (audioUrl) {
      const audioTempPath = path.join(workDir, "background-music-source.bin");
      const audioInfo = await downloadToFile(audioUrl, audioTempPath);
      const audioPath = path.join(workDir, `background-music${inferExtension(audioUrl, audioInfo.contentType)}`);
      await fs.rename(audioTempPath, audioPath);
      await mixBackgroundMusic(joinedPath, audioPath, outputPath, payload.musicVolume || payload.backgroundMusicVolume || 0.18);
    } else {
      await run("ffmpeg", ["-y", "-i", joinedPath, "-c", "copy", "-movflags", "+faststart", outputPath], { label: "faststart final", timeoutMs: 30 * 60 * 1000 });
    }
    updateJob(jobId, { status: "processing", progress: 94, currentStep: "Uploading MP4" });
    const upload = await uploadOutput(eventId, jobId, outputPath, outputName);
    updateJob(jobId, { status: "complete", progress: 100, currentStep: "Complete", outputUrl: upload.url, outputBucket: upload.bucket, outputPath: upload.path, outputBytes: upload.bytes, completedAt: nowIso(), payload: undefined });
  } finally {
    await fs.rm(workDir, { recursive: true, force: true }).catch((error) => console.warn("[VideoRender] Could not remove temp dir:", safeMessage(error)));
  }
}

function processQueue() { cleanOldJobs(); while (activeRenders < MAX_CONCURRENT_RENDERS && queue.length) { const jobId = queue.shift(); activeRenders += 1; updateJob(jobId, { status: "processing", progress: 1, startedAt: nowIso(), currentStep: "Starting render" }); renderJob(jobId).catch((error) => { console.error(`[VideoRender] Job ${jobId} failed:`, error); updateJob(jobId, { status: "failed", progress: 100, currentStep: "Failed", error: safeMessage(error), failedAt: nowIso(), payload: undefined }); }).finally(() => { activeRenders -= 1; processQueue(); }); } }
function publicJob(job) { if (!job) return null; const { payload, ...safe } = job; return safe; }
async function handle(req, res) {
  if (req.method === "OPTIONS") return json(res, 204, {});
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (req.method === "GET" && url.pathname === "/health") return json(res, 200, { ok: true, service: "memory-lane-video-render-worker", activeRenders, queued: queue.length, hasSupabase: !!supabase, hasSecret: !!RENDER_SECRET });
  if (!isAuthed(req)) return json(res, 401, { error: "Unauthorised" });
  if (req.method === "POST" && url.pathname === "/render") {
    try {
      const payload = await readJsonBody(req);
      const media = normaliseMedia(payload);
      if (!media.length) return json(res, 400, { error: "No media URLs supplied" });
      const jobId = crypto.randomUUID();
      const createdAt = nowIso();
      jobs.set(jobId, { jobId, status: "queued", progress: 0, currentStep: "Queued", createdAt, updatedAt: createdAt, mediaCount: media.length, payload: { ...payload, media } });
      queue.push(jobId);
      processQueue();
      return json(res, 202, { jobId, status: "queued", statusPath: `/jobs/${jobId}` });
    } catch (error) { return json(res, 400, { error: safeMessage(error) }); }
  }
  const jobMatch = url.pathname.match(/^\/jobs\/([a-f0-9-]+)$/i);
  if (req.method === "GET" && jobMatch) { const job = publicJob(jobs.get(jobMatch[1])); return job ? json(res, 200, job) : json(res, 404, { error: "Job not found" }); }
  return json(res, 404, { error: "Not found" });
}

createServer((req, res) => { handle(req, res).catch((error) => { console.error("[VideoRender] Request error:", error); json(res, 500, { error: safeMessage(error) }); }); }).listen(PORT, () => { console.log(`[VideoRender] Worker listening on ${PORT}`); console.log(`[VideoRender] Output bucket: ${OUTPUT_BUCKET}`); console.log(`[VideoRender] Max concurrent renders: ${MAX_CONCURRENT_RENDERS}`); });
