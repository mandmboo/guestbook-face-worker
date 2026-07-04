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

if (!RENDER_SECRET) {
  console.warn("[VideoRender] RENDER_SECRET is not set. Render endpoints will reject requests.");
}

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.warn("[VideoRender] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing. Uploads will fail until configured.");
}

const supabase = SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY
  ? createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
  : null;

const jobs = new Map();
const queue = [];
let activeRenders = 0;
let bucketReady = false;

function json(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type,x-render-secret,authorization",
  });
  res.end(payload);
}

function safeMessage(error) {
  return String(error?.message || error || "Unknown error").slice(0, 2000);
}

function nowIso() {
  return new Date().toISOString();
}

function updateJob(jobId, patch) {
  const existing = jobs.get(jobId) || {};
  const next = {
    ...existing,
    ...patch,
    updatedAt: nowIso(),
  };
  jobs.set(jobId, next);
  return next;
}

function cleanOldJobs() {
  const cutoff = Date.now() - JOB_RETENTION_MS;
  for (const [jobId, job] of jobs.entries()) {
    const updatedAt = Date.parse(job.updatedAt || job.createdAt || 0);
    if (updatedAt && updatedAt < cutoff && ["complete", "failed"].includes(job.status)) {
      jobs.delete(jobId);
    }
  }
}

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
  if (!raw.trim()) return {};
  return JSON.parse(raw);
}

function slugify(value, fallback = "memory-film") {
  return String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || fallback;
}

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
  if (type.includes("wav")) return ".wav";
  try {
    const ext = path.extname(new URL(url).pathname).toLowerCase();
    if (ext) return ext.slice(0, 10);
  } catch {
    // ignore
  }
  return ".bin";
}

function inferMediaType(item, contentType = "") {
  const explicit = String(item.type || item.kind || "").toLowerCase();
  if (explicit.includes("video")) return "video";
  if (explicit.includes("image") || explicit.includes("photo")) return "image";
  const type = contentType.toLowerCase();
  if (type.startsWith("video/")) return "video";
  if (type.startsWith("image/")) return "image";
  const url = String(item.url || item.storage_url || item.src || "").toLowerCase();
  if (/\.(mp4|mov|webm|m4v)(\?|#|$)/.test(url)) return "video";
  return "image";
}

async function run(command, args, options = {}) {
  const label = options.label || command;
  const timeoutMs = options.timeoutMs || 10 * 60 * 1000;
  console.log(`[VideoRender] ${label}: ${command} ${args.join(" ")}`);

  return await new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
      stdout = stdout.slice(-12000);
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
      stderr = stderr.slice(-20000);
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`${label} failed with code ${code}: ${stderr || stdout}`));
      }
    });
  });
}

async function downloadToFile(url, destination) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DOWNLOAD_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: {
        "user-agent": "MemoryLaneVideoRender/1.0",
        accept: "image/*,video/*,audio/*,*/*;q=0.8",
      },
    });

    if (!response.ok || !response.body) {
      throw new Error(`Download failed ${response.status} for ${url}`);
    }

    const length = Number(response.headers.get("content-length") || 0);
    if (length && length > MAX_DOWNLOAD_BYTES) {
      throw new Error(`Media file is too large (${length} bytes)`);
    }

    let downloaded = 0;
    const writeStream = createWriteStream(destination);
    const reader = response.body.getReader();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      downloaded += value.byteLength;
      if (downloaded > MAX_DOWNLOAD_BYTES) {
        throw new Error(`Media file exceeded max download size (${MAX_DOWNLOAD_BYTES} bytes)`);
      }
      if (!writeStream.write(Buffer.from(value))) {
        await new Promise((resolve) => writeStream.once("drain", resolve));
      }
    }

    await new Promise((resolve, reject) => {
      writeStream.end(resolve);
      writeStream.on("error", reject);
    });

    return {
      contentType: response.headers.get("content-type") || "",
      bytes: downloaded,
    };
  } finally {
    clearTimeout(timer);
  }
}

function normaliseNumber(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function dimensionsForPayload(payload) {
  const aspect = String(payload.aspectRatio || payload.aspect || "vertical").toLowerCase();
  const width = normaliseNumber(payload.width, aspect.includes("square") ? 1080 : aspect.includes("landscape") ? 1920 : 1080, 320, 3840);
  const height = normaliseNumber(payload.height, aspect.includes("square") ? 1080 : aspect.includes("landscape") ? 1080 : 1920, 320, 3840);
  return {
    width: width % 2 === 0 ? width : width + 1,
    height: height % 2 === 0 ? height : height + 1,
  };
}

function buildVideoFilter(width, height, fps, duration, fade = false) {
  const base = [
    `scale=${width}:${height}:force_original_aspect_ratio=decrease`,
    `pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2:color=black`,
    "setsar=1",
    "format=yuv420p",
  ];

  if (fade && duration >= 2) {
    const fadeOutStart = Math.max(0, duration - 0.45).toFixed(2);
    base.push("fade=t=in:st=0:d=0.35");
    base.push(`fade=t=out:st=${fadeOutStart}:d=0.35`);
  }

  base.push(`fps=${fps}`);
  return base.join(",");
}

async function renderImageSegment(inputPath, outputPath, options) {
  const { width, height, fps, duration } = options;
  const filter = buildVideoFilter(width, height, fps, duration, true);
  await run("ffmpeg", [
    "-y",
    "-loop", "1",
    "-t", String(duration),
    "-i", inputPath,
    "-vf", filter,
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "21",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    outputPath,
  ], { label: "render image segment" });
}

async function renderVideoSegment(inputPath, outputPath, options) {
  const { width, height, fps, duration } = options;
  const filter = buildVideoFilter(width, height, fps, duration, false);
  const args = ["-y", "-i", inputPath];
  if (duration) args.push("-t", String(duration));
  args.push(
    "-vf", filter,
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "21",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    outputPath
  );
  await run("ffmpeg", args, { label: "render video segment", timeoutMs: 20 * 60 * 1000 });
}

async function ensureBucket() {
  if (bucketReady) return;
  if (!supabase) throw new Error("Supabase is not configured for video output uploads");

  const { data: existing, error: listError } = await supabase.storage.listBuckets();
  if (listError) throw listError;

  if (!existing?.some((bucket) => bucket.name === OUTPUT_BUCKET)) {
    const { error: createError } = await supabase.storage.createBucket(OUTPUT_BUCKET, {
      public: true,
      fileSizeLimit: MAX_DOWNLOAD_BYTES,
      allowedMimeTypes: ["video/mp4"],
    });

    if (createError && !String(createError.message || "").toLowerCase().includes("already")) {
      throw createError;
    }
  }

  bucketReady = true;
}

async function uploadOutput(eventId, jobId, outputPath, outputName) {
  await ensureBucket();
  const buffer = await fs.readFile(outputPath);
  const storagePath = `events/${slugify(eventId, "event")}/films/${jobId}-${slugify(outputName, "memory-film")}.mp4`;

  const { error: uploadError } = await supabase.storage.from(OUTPUT_BUCKET).upload(storagePath, buffer, {
    contentType: "video/mp4",
    cacheControl: "3600",
    upsert: true,
  });
  if (uploadError) throw uploadError;

  const { data } = supabase.storage.from(OUTPUT_BUCKET).getPublicUrl(storagePath);
  if (!data?.publicUrl) throw new Error("Could not create public MP4 URL");

  return {
    url: data.publicUrl,
    bucket: OUTPUT_BUCKET,
    path: storagePath,
    bytes: buffer.length,
  };
}

function normaliseMedia(payload) {
  const media = Array.isArray(payload.media)
    ? payload.media
    : Array.isArray(payload.items)
      ? payload.items
      : [];

  return media
    .map((item) => ({
      ...item,
      url: item.url || item.storage_url || item.storageUrl || item.src,
    }))
    .filter((item) => typeof item.url === "string" && item.url.startsWith("http"))
    .slice(0, MAX_MEDIA_ITEMS);
}

async function renderJob(jobId) {
  const job = jobs.get(jobId);
  const payload = job.payload;
  const eventId = payload.eventId || payload.event_id || "event";
  const media = normaliseMedia(payload);

  if (!media.length) {
    throw new Error("No renderable media URLs were provided");
  }

  const { width, height } = dimensionsForPayload(payload);
  const fps = normaliseNumber(payload.fps, 30, 15, 60);
  const defaultPhotoDuration = normaliseNumber(payload.durationPerPhoto || payload.photoDuration, 4, 1.5, 12);
  const maxVideoDuration = normaliseNumber(payload.maxVideoClipDuration || payload.videoDuration, 10, 2, 60);
  const outputName = payload.outputName || payload.title || "memory-film";

  const workDir = await fs.mkdtemp(path.join(tmpdir(), `memory-film-${jobId}-`));
  console.log(`[VideoRender] Job ${jobId} using temp dir ${workDir}`);

  try {
    const segmentPaths = [];

    for (let index = 0; index < media.length; index += 1) {
      const item = media[index];
      updateJob(jobId, {
        status: "processing",
        progress: Math.round((index / media.length) * 70),
        currentStep: `Preparing media ${index + 1} of ${media.length}`,
      });

      const probePath = path.join(workDir, `input-${index}.bin`);
      const downloaded = await downloadToFile(item.url, probePath);
      const mediaType = inferMediaType(item, downloaded.contentType);
      const ext = inferExtension(item.url, downloaded.contentType);
      const inputPath = path.join(workDir, `input-${index}${ext}`);
      await fs.rename(probePath, inputPath);

      const segmentPath = path.join(workDir, `segment-${String(index).padStart(4, "0")}.mp4`);
      const duration = mediaType === "video"
        ? normaliseNumber(item.duration || item.clipDuration || maxVideoDuration, maxVideoDuration, 1, 120)
        : normaliseNumber(item.duration || defaultPhotoDuration, defaultPhotoDuration, 1, 20);

      if (mediaType === "video") {
        await renderVideoSegment(inputPath, segmentPath, { width, height, fps, duration });
      } else {
        await renderImageSegment(inputPath, segmentPath, { width, height, fps, duration });
      }

      segmentPaths.push(segmentPath);
    }

    updateJob(jobId, {
      status: "processing",
      progress: 76,
      currentStep: "Joining film sections",
    });

    const concatListPath = path.join(workDir, "concat.txt");
    const concatList = segmentPaths
      .map((segmentPath) => `file '${segmentPath.replace(/'/g, "'\\''")}'`)
      .join("\n");
    await fs.writeFile(concatListPath, concatList, "utf8");

    const joinedPath = path.join(workDir, "joined.mp4");
    await run("ffmpeg", [
      "-y",
      "-f", "concat",
      "-safe", "0",
      "-i", concatListPath,
      "-c", "copy",
      joinedPath,
    ], { label: "concat segments", timeoutMs: 20 * 60 * 1000 });

    updateJob(jobId, {
      status: "processing",
      progress: 86,
      currentStep: "Adding audio and finalising MP4",
    });

    const outputPath = path.join(workDir, "final.mp4");
    const audioUrl = payload.audioUrl || payload.musicUrl || payload.soundtrackUrl || "";

    if (audioUrl) {
      const audioTempPath = path.join(workDir, "audio-source.bin");
      const audioInfo = await downloadToFile(audioUrl, audioTempPath);
      const audioExt = inferExtension(audioUrl, audioInfo.contentType);
      const audioPath = path.join(workDir, `audio${audioExt}`);
      await fs.rename(audioTempPath, audioPath);

      await run("ffmpeg", [
        "-y",
        "-i", joinedPath,
        "-i", audioPath,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        outputPath,
      ], { label: "mux soundtrack", timeoutMs: 20 * 60 * 1000 });
    } else {
      await run("ffmpeg", [
        "-y",
        "-i", joinedPath,
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        outputPath,
      ], { label: "add silent audio", timeoutMs: 20 * 60 * 1000 });
    }

    updateJob(jobId, {
      status: "processing",
      progress: 94,
      currentStep: "Uploading MP4",
    });

    const upload = await uploadOutput(eventId, jobId, outputPath, outputName);

    updateJob(jobId, {
      status: "complete",
      progress: 100,
      currentStep: "Complete",
      outputUrl: upload.url,
      outputBucket: upload.bucket,
      outputPath: upload.path,
      outputBytes: upload.bytes,
      completedAt: nowIso(),
      payload: undefined,
    });

    console.log(`[VideoRender] Job ${jobId} complete: ${upload.url}`);
  } finally {
    try {
      await fs.rm(workDir, { recursive: true, force: true });
    } catch (error) {
      console.warn(`[VideoRender] Could not remove temp dir ${workDir}:`, safeMessage(error));
    }
  }
}

function processQueue() {
  cleanOldJobs();

  while (activeRenders < MAX_CONCURRENT_RENDERS && queue.length) {
    const jobId = queue.shift();
    activeRenders += 1;

    updateJob(jobId, {
      status: "processing",
      progress: 1,
      startedAt: nowIso(),
      currentStep: "Starting render",
    });

    renderJob(jobId)
      .catch((error) => {
        console.error(`[VideoRender] Job ${jobId} failed:`, error);
        updateJob(jobId, {
          status: "failed",
          progress: 100,
          currentStep: "Failed",
          error: safeMessage(error),
          failedAt: nowIso(),
          payload: undefined,
        });
      })
      .finally(() => {
        activeRenders -= 1;
        processQueue();
      });
  }
}

function publicJob(job) {
  if (!job) return null;
  const { payload, ...safe } = job;
  return safe;
}

async function handle(req, res) {
  if (req.method === "OPTIONS") {
    return json(res, 204, {});
  }

  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "GET" && url.pathname === "/health") {
    return json(res, 200, {
      ok: true,
      service: "memory-lane-video-render-worker",
      activeRenders,
      queued: queue.length,
      hasSupabase: !!supabase,
      hasSecret: !!RENDER_SECRET,
    });
  }

  if (!isAuthed(req)) {
    return json(res, 401, { error: "Unauthorised" });
  }

  if (req.method === "POST" && url.pathname === "/render") {
    try {
      const payload = await readJsonBody(req);
      const media = normaliseMedia(payload);

      if (!media.length) {
        return json(res, 400, { error: "No media URLs supplied" });
      }

      const jobId = crypto.randomUUID();
      const createdAt = nowIso();

      jobs.set(jobId, {
        jobId,
        status: "queued",
        progress: 0,
        currentStep: "Queued",
        createdAt,
        updatedAt: createdAt,
        mediaCount: media.length,
        payload: {
          ...payload,
          media,
        },
      });

      queue.push(jobId);
      processQueue();

      return json(res, 202, {
        jobId,
        status: "queued",
        statusPath: `/jobs/${jobId}`,
      });
    } catch (error) {
      return json(res, 400, { error: safeMessage(error) });
    }
  }

  const jobMatch = url.pathname.match(/^\/jobs\/([a-f0-9-]+)$/i);
  if (req.method === "GET" && jobMatch) {
    const job = publicJob(jobs.get(jobMatch[1]));
    if (!job) return json(res, 404, { error: "Job not found" });
    return json(res, 200, job);
  }

  return json(res, 404, { error: "Not found" });
}

createServer((req, res) => {
  handle(req, res).catch((error) => {
    console.error("[VideoRender] Request error:", error);
    json(res, 500, { error: safeMessage(error) });
  });
}).listen(PORT, () => {
  console.log(`[VideoRender] Worker listening on ${PORT}`);
  console.log(`[VideoRender] Output bucket: ${OUTPUT_BUCKET}`);
  console.log(`[VideoRender] Max concurrent renders: ${MAX_CONCURRENT_RENDERS}`);
});
