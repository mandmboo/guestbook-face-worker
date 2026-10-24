import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, promises as fs, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import vm from "node:vm";
import test, { before, after } from "node:test";

// Real codecs, synthetic media, stubbed HTTP/Storage: no credentials, network,
// customer photos, buckets or production services are used by these tests.
const source = readFileSync(new URL("../video-render-worker.js", import.meta.url), "utf8");
const options = { width: 320, height: 320, fps: 15, duration: 1.8 };
let dir;
let fixtures;

function ffmpeg(args) {
  const result = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], {
    timeout: 15000, encoding: "utf8", maxBuffer: 1024 * 1024,
  });
  assert.equal(result.status, 0, result.error?.message || result.stderr);
}

function probe(file) {
  const result = spawnSync("ffprobe", ["-v", "error", "-show_format", "-show_streams", "-of", "json", file], {
    timeout: 5000, encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}

function assertVideo(file, duration, audio = false) {
  const info = probe(file);
  const video = info.streams.find((stream) => stream.codec_type === "video");
  assert.ok(video, "video stream exists");
  assert.equal(video.codec_name, "h264");
  assert.equal(video.width, options.width);
  assert.equal(video.height, options.height);
  assert.ok(["yuv420p", "yuvj420p"].includes(video.pix_fmt), video.pix_fmt);
  assert.ok(Math.abs(Number(video.duration) - duration) < 0.15, `video duration ${video.duration} != ${duration}`);
  assert.ok(Number.isFinite(Number(info.format.duration)) && Number(info.format.duration) > 0);
  assert.equal(info.streams.some((stream) => stream.codec_type === "audio"), audio);
  // Decode every frame, not just the metadata, to catch damaged output.
  ffmpeg(["-xerror", "-i", file, "-f", "null", "-"]);
}

function loadWorker({ processSpawn = spawn, runTimeout, inputFiles = {} } = {}) {
  const uploads = [];
  const workDirs = [];
  const children = [];
  const context = {
    process: { env: { SUPABASE_URL: "https://unit-test.invalid", RENDER_SECRET: "test-only" } },
    getSupabaseServerKey: () => "non-credential-test-fixture",
    createClient: () => ({ storage: {
      listBuckets: async () => ({ data: [{ name: "rendered-films" }] }),
      from: () => ({
        upload: async (key, bytes) => { uploads.push({ key, bytes }); return {}; },
        getPublicUrl: (key) => ({ data: { publicUrl: `https://unit-test.invalid/${key}` } }),
      }),
    } }),
    createServer: () => ({ listen() {} }),
    spawn: (command, args, opts) => {
      const child = processSpawn(command, args, opts);
      children.push(child);
      return child;
    },
    createWriteStream,
    fs: { ...fs, mkdtemp: async (prefix) => { const result = await fs.mkdtemp(prefix); workDirs.push(result); return result; } },
    tmpdir: () => dir,
    path, crypto, Buffer, URL, AbortController,
    setTimeout: (callback, delay) => setTimeout(callback, runTimeout && delay >= 30000 ? runTimeout : delay),
    clearTimeout,
    console: { log() {}, warn() {}, error() {} },
    fetch: async (url) => {
      const fixture = inputFiles[url];
      assert.ok(fixture, `Unexpected network access: ${url}`);
      return new Response(await fs.readFile(fixture.file), { headers: { "content-type": fixture.type } });
    },
  };
  vm.runInNewContext(source.replace(/^import .*;\n/gm, "") + `
    globalThis.worker = { run, renderImageSegment, renderVideoSegment, renderJob,
      jobs, queue, processQueue, health: () => ({ activeRenders, queued: queue.length }) };
  `, context, { filename: "video-render-worker.js" });
  return { ...context.worker, uploads, workDirs, children };
}

async function assertRemoved(dirs) {
  assert.ok(dirs.length > 0);
  for (const directory of dirs) await assert.rejects(fs.stat(directory), { code: "ENOENT" });
}

async function waitForIdle(worker) {
  const deadline = Date.now() + 15000;
  while (worker.health().activeRenders || worker.health().queued) {
    assert.ok(Date.now() < deadline, "queue failed to become idle");
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

function queueJob(worker, id, media, extras = {}) {
  worker.jobs.set(id, { status: "queued", payload: { eventId: "synthetic-event", ...options, media, ...extras } });
  worker.queue.push(id);
}

before(async () => {
  for (const binary of ["ffmpeg", "ffprobe"]) {
    const result = spawnSync(binary, ["-version"]);
    assert.equal(result.status, 0, `${binary} is required; run in Dockerfile.video or install FFmpeg`);
  }
  dir = await fs.mkdtemp(path.join(tmpdir(), "video-render-regressions-"));
  fixtures = Object.fromEntries(["jpg", "png", "webp", "gif", "still.gif", "finite.gif", "limited.gif", "variable.gif", "mislabelled.jpg", "broken.png", "video.mp4", "audio.wav"].map((name) => [name, path.join(dir, `fixture.${name}`)]));
  for (const ext of ["jpg", "png", "webp", "still.gif"]) {
    ffmpeg(["-f", "lavfi", "-i", "color=c=red:size=64x48", "-frames:v", "1", fixtures[ext]]);
  }
  ffmpeg(["-f", "lavfi", "-i", "color=c=red:size=64x48:rate=2:duration=0.5", "-f", "lavfi", "-i", "color=c=blue:size=64x48:rate=2:duration=0.5", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-loop", "0", fixtures.gif]);
  ffmpeg(["-i", fixtures.gif, "-loop", "-1", fixtures["finite.gif"]]);
  ffmpeg(["-i", fixtures.gif, "-loop", "1", fixtures["limited.gif"]]);
  const variable = await fs.readFile(fixtures.gif);
  let controlOffset = 0;
  for (const delay of [30, 70]) {
    controlOffset = variable.indexOf(Buffer.from([0x21, 0xf9, 0x04]), controlOffset);
    assert.ok(controlOffset >= 0, "synthetic GIF has two frame-control extensions");
    variable.writeUInt16LE(delay, controlOffset + 4);
    controlOffset += 8;
  }
  await fs.writeFile(fixtures["variable.gif"], variable);
  await fs.copyFile(fixtures.gif, fixtures["mislabelled.jpg"]);
  const corrupt = await fs.readFile(fixtures.png);
  const idat = corrupt.indexOf(Buffer.from("IDAT"));
  assert.ok(idat > 0);
  // Break the zlib header of a synthetic PNG; this used to decode-loop until
  // the worker's ten-minute timer fired, rather than fail at the bad frame.
  corrupt[idat + 4] = 0;
  await fs.writeFile(fixtures["broken.png"], corrupt);
  ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=64x48:rate=15:duration=2", "-c:v", "libx264", fixtures["video.mp4"]]);
  ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=10", fixtures["audio.wav"]]);
});

after(async () => { if (dir) await fs.rm(dir, { recursive: true, force: true }); });

for (const format of ["jpg", "png", "webp", "still.gif", "gif", "finite.gif", "limited.gif", "variable.gif", "mislabelled.jpg"]) {
  test(`real FFmpeg renders ${format} for the whole bounded photo slot`, { timeout: 15000 }, async () => {
    const worker = loadWorker();
    const output = path.join(dir, `${format}.mp4`);
    await worker.renderImageSegment(fixtures[format], output, options);
    assertVideo(output, options.duration);
    await assert.rejects(fs.stat(`${output}.loop.gif`), { code: "ENOENT" });
  });
}

for (const format of ["gif", "finite.gif", "limited.gif", "variable.gif", "mislabelled.jpg"]) {
  test(`${format} animation survives with original frame delays and repeats`, { timeout: 15000 }, async () => {
    const worker = loadWorker();
    const output = path.join(dir, `animated-${format}.mp4`);
    await worker.renderImageSegment(fixtures[format], output, options);
    for (const [time, expected] of [[0.1, "red"], [0.4, format === "variable.gif" ? "blue" : "red"], [0.6, "blue"], [1.1, "red"], [1.4, format === "variable.gif" ? "blue" : "red"], [1.6, "blue"]]) {
      const result = spawnSync("ffmpeg", ["-v", "error", "-ss", String(time), "-i", output, "-frames:v", "1", "-vf", "scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], { timeout: 5000 });
      assert.equal(result.status, 0, result.stderr.toString());
      const [red, , blue] = result.stdout;
      assert.ok(expected === "red" ? red > blue + 100 : blue > red + 100, `expected ${expected} at ${time}s, got ${[...result.stdout]}`);
    }
  });
}

test("bad PNG fails fast with decoder details instead of waiting ten minutes", { timeout: 10000 }, async () => {
  const worker = loadWorker();
  const start = Date.now();
  await assert.rejects(worker.renderImageSegment(fixtures["broken.png"], path.join(dir, "bad.mp4"), options), /render image segment failed.*(?:inflate|Invalid|Error|error)/s);
  assert.ok(Date.now() - start < 5000);
  assert.ok(worker.children.every((child) => child.exitCode !== null || child.signalCode !== null));
});

test("video clips keep their existing bounded conversion", { timeout: 15000 }, async () => {
  const worker = loadWorker();
  const output = path.join(dir, "video-segment.mp4");
  await worker.renderVideoSegment(fixtures["video.mp4"], output, options);
  assertVideo(output, options.duration);
});

for (const soundtrack of [false, true]) {
  test(`full JPEG + GIF + PNG outro job completes with ${soundtrack ? "soundtrack" : "silent audio"} and cleans its files`, { timeout: 30000 }, async () => {
    const media = ["jpg", "gif", "png"].map((ext) => ({ url: `https://unit-test.invalid/${ext}`, type: "image", duration: ext === "png" ? 5 : 1.8 }));
    const inputFiles = Object.fromEntries(["jpg", "gif", "png"].map((ext) => [`https://unit-test.invalid/${ext}`, { file: fixtures[ext], type: `image/${ext}` }]));
    inputFiles["https://unit-test.invalid/audio"] = { file: fixtures["audio.wav"], type: "audio/wav" };
    const worker = loadWorker({ inputFiles });
    queueJob(worker, "good", media, soundtrack ? { audioUrl: "https://unit-test.invalid/audio" } : {});
    worker.processQueue();
    await waitForIdle(worker);
    assert.equal(worker.jobs.get("good").status, "complete", worker.jobs.get("good").error);
    assert.equal(worker.uploads.length, 1);
    const output = path.join(dir, `full-${soundtrack}.mp4`);
    await fs.writeFile(output, worker.uploads[0].bytes);
    assertVideo(output, 8.6, true);
    await assertRemoved(worker.workDirs);
  });
}

test("corrupt media fails with its position, no upload, cleanup and queue recovery", { timeout: 30000 }, async () => {
  const worker = loadWorker({ inputFiles: {
    "https://unit-test.invalid/bad": { file: fixtures["broken.png"], type: "image/png" },
    "https://unit-test.invalid/good": { file: fixtures.jpg, type: "image/jpeg" },
  } });
  queueJob(worker, "bad", [{ url: "https://unit-test.invalid/good" }, { url: "https://unit-test.invalid/bad" }]);
  queueJob(worker, "next", [{ url: "https://unit-test.invalid/good" }]);
  worker.processQueue();
  await waitForIdle(worker);
  assert.equal(worker.jobs.get("bad").status, "failed");
  assert.match(worker.jobs.get("bad").error, /Cannot render image 2 of 2.*no film was uploaded.*failed/s);
  assert.equal(worker.jobs.get("bad").payload, undefined);
  assert.equal(worker.jobs.get("next").status, "complete", worker.jobs.get("next").error);
  assert.equal(worker.uploads.length, 1);
  assert.match(worker.uploads[0].key, /next/);
  await assertRemoved(worker.workDirs);
});

for (const mode of ["timeout", "cancel"]) {
  test(`${mode} reaps the subprocess before rejecting`, { timeout: 5000 }, async () => {
    const worker = loadWorker();
    const controller = new AbortController();
    const pending = worker.run(process.execPath, ["-e", "setInterval(() => {}, 1000)"], {
      label: "test child", timeoutMs: mode === "timeout" ? 100 : 3000, signal: controller.signal,
    });
    if (mode === "cancel") setTimeout(() => controller.abort(), 100);
    await assert.rejects(pending, mode === "timeout" ? /timed out after 100ms/ : /cancelled/);
    const child = worker.children[0];
    assert.equal(child.signalCode, "SIGKILL");
    assert.throws(() => process.kill(child.pid, 0), { code: "ESRCH" });
  });
}

test("pre-cancelled subprocess does not spawn", async () => {
  const worker = loadWorker();
  await assert.rejects(worker.run(process.execPath, [], { signal: AbortSignal.abort() }), /cancelled/);
  assert.equal(worker.children.length, 0);
});

test("timed-out render leaves no upload or running child and removes its temp directory", { timeout: 5000 }, async () => {
  const worker = loadWorker({
    inputFiles: { "https://unit-test.invalid/photo": { file: fixtures.jpg, type: "image/jpeg" } },
    processSpawn: () => spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: ["ignore", "pipe", "pipe"] }),
    runTimeout: 100,
  });
  queueJob(worker, "timeout", [{ url: "https://unit-test.invalid/photo" }]);
  worker.processQueue();
  await waitForIdle(worker);
  assert.equal(worker.jobs.get("timeout").status, "failed");
  assert.match(worker.jobs.get("timeout").error, /timed out/);
  assert.equal(worker.uploads.length, 0);
  for (const child of worker.children) assert.throws(() => process.kill(child.pid, 0), { code: "ESRCH" });
  await assertRemoved(worker.workDirs);
});

test("missing executable rejects without a hanging timer", { timeout: 5000 }, async () => {
  const worker = loadWorker();
  await assert.rejects(worker.run("/this-executable-does-not-exist", []), { code: "ENOENT" });
});

test("GIF loop preparation preserves every decoded pixel and unequal frame delays", { timeout: 15000 }, async () => {
  let remuxed;
  const worker = loadWorker({ processSpawn: (command, args, opts) => {
    if (command === "ffmpeg" && args.includes("-ignore_loop") && args[args.indexOf("-ignore_loop") + 1] === "0") {
      remuxed = readFileSync(args[args.indexOf("-i") + 1]);
    }
    return spawn(command, args, opts);
  } });
  await worker.renderImageSegment(fixtures["variable.gif"], path.join(dir, "lossless.mp4"), options);
  assert.ok(remuxed);
  const copy = path.join(dir, "remuxed.gif");
  await fs.writeFile(copy, remuxed);
  const frames = (file) => {
    const result = spawnSync("ffmpeg", ["-v", "error", "-ignore_loop", "1", "-i", file, "-f", "framemd5", "-"], { timeout: 5000, encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.split("\n").filter((line) => line && !line.startsWith("#")).map((line) => line.split(",").at(-1).trim());
  };
  const delays = (file) => {
    const result = spawnSync("ffprobe", ["-v", "error", "-show_entries", "packet=duration_time", "-of", "csv=p=0", file], { timeout: 5000, encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  assert.deepEqual(frames(copy), frames(fixtures["variable.gif"]));
  assert.equal(delays(copy), "0.300000\n0.700000");
  assert.equal(delays(copy), delays(fixtures["variable.gif"]));
});

test("a GIF cycle longer than the photo slot is cut at the slot, not sped up", { timeout: 15000 }, async () => {
  const bytes = await fs.readFile(fixtures["variable.gif"]);
  const control = bytes.indexOf(Buffer.from([0x21, 0xf9, 0x04]));
  bytes.writeUInt16LE(250, control + 4);
  const longGif = path.join(dir, "long.gif");
  await fs.writeFile(longGif, bytes);
  const worker = loadWorker();
  const output = path.join(dir, "long.mp4");
  await worker.renderImageSegment(longGif, output, options);
  assertVideo(output, options.duration);
  const frames = spawnSync("ffmpeg", ["-v", "error", "-i", output, "-vf", "scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], { timeout: 5000 });
  assert.equal(frames.status, 0, frames.stderr.toString());
  assert.equal(frames.stdout.length, 27 * 3);
  for (let offset = 0; offset < frames.stdout.length; offset += 3) {
    assert.ok(frames.stdout[offset] > frames.stdout[offset + 2] + 100, "red first frame lasts longer than the entire requested slot");
  }
});

test("GIF encoding failure removes the intermediate loop file", { timeout: 15000 }, async () => {
  const output = path.join(dir, "gif-encode-failure.mp4");
  const worker = loadWorker({ processSpawn: (command, args, opts) => {
    if (args.includes("libx264")) return spawn(process.execPath, ["-e", "process.stderr.write('synthetic encoder failure'); process.exit(3)"], opts);
    return spawn(command, args, opts);
  } });
  await assert.rejects(worker.renderImageSegment(fixtures.gif, output, options), /render image segment failed.*synthetic encoder failure/s);
  await assert.rejects(fs.stat(`${output}.loop.gif`), { code: "ENOENT" });
});

test("GIF timing timeout is bounded, reaped and cleans up the job", { timeout: 5000 }, async () => {
  const worker = loadWorker({
    inputFiles: { "https://unit-test.invalid/gif": { file: fixtures.gif, type: "image/gif" } },
    processSpawn: () => spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { stdio: ["ignore", "pipe", "pipe"] }),
    runTimeout: 100,
  });
  queueJob(worker, "gif-timeout", [{ url: "https://unit-test.invalid/gif" }]);
  worker.processQueue();
  await waitForIdle(worker);
  assert.match(worker.jobs.get("gif-timeout").error, /probe GIF timing timed out/);
  assert.equal(worker.uploads.length, 0);
  for (const child of worker.children) assert.throws(() => process.kill(child.pid, 0), { code: "ESRCH" });
  await assertRemoved(worker.workDirs);
});
