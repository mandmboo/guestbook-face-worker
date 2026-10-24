# Memory Lane MP4 render worker

This adds a separate Railway service for server-side MP4 rendering.

It does not replace or change the Face ID worker.

## What stays the same

The existing Face ID service still starts with:

```bash
node worker.js
```

The new MP4 renderer uses a different file and a different Dockerfile:

```bash
node video-render-worker.js
Dockerfile.video
```

## Railway setup

Create a new Railway service from this same repo and point it at:

```text
Dockerfile.video
```

Do not change the existing Face ID Railway service start command.

## Required environment variables on the new video service

```text
RENDER_SECRET=make-a-long-random-secret
SUPABASE_URL=your-supabase-url
SUPABASE_SECRET_KEY=your-server-only-supabase-secret-key
RENDER_OUTPUT_BUCKET=rendered-films
MAX_CONCURRENT_RENDERS=1
```

`RENDER_SECRET` must match the value used in Vercel as `MEMORY_FILM_RENDER_SECRET`.

Both workers support `SUPABASE_SECRET_KEY`, but keep preferring the existing
`SUPABASE_SERVICE_ROLE_KEY` while `MEMORY_LANE_CREDENTIAL_MODE` is unset or
`legacy-compatible`. This lets a replacement key be staged without activating
it. After separate approval, `secret-compatible` prefers the new key while
retaining the legacy fallback. A publishable key,
anon/user JWT, or missing privileged credential stops startup; the renderer no
longer advertises a healthy process that cannot upload its results.

After a separately approved migration has installed and tested a new secret key
for each worker, set `MEMORY_LANE_CREDENTIAL_MODE=strict`. Strict mode requires
`SUPABASE_SECRET_KEY` containing a modern secret key and never falls back to a
legacy key. Invalid modes fail startup. This is configuration preparation only:
it does not rotate, revoke, or disable any Supabase key. Do not disable legacy
keys until every consumer (including the Vercel app and Edge Functions) has been
migrated and verified. Keep the already-rotated render secret unchanged.

Run the local credential/startup regression tests with `npm test`. These tests
stub Supabase rather than contacting production; a deployment and real worker
smoke test still require separate approval.

## Endpoints

Health check:

```text
GET /health
```

Start a render:

```text
POST /render
x-render-secret: <RENDER_SECRET>
```

Body example:

```json
{
  "eventId": "event-id",
  "title": "Chris and Ali",
  "aspectRatio": "vertical",
  "durationPerPhoto": 4,
  "media": [
    { "type": "image", "url": "https://.../photo.jpg" },
    { "type": "video", "url": "https://.../clip.mp4", "duration": 8 }
  ],
  "audioUrl": "https://.../music.mp3"
}
```

Check status:

```text
GET /jobs/<jobId>
x-render-secret: <RENDER_SECRET>
```

Completed jobs return an `outputUrl` pointing at the generated MP4 in Supabase Storage.

## Output

The worker uploads final MP4 files to Supabase Storage:

```text
rendered-films/events/<eventId>/films/<jobId>-memory-film.mp4
```

The worker will create the bucket if the Supabase service role allows it.

## Image compatibility and fail-fast rendering

GIFs are treated as animated photos: all frames retain their timing and repeat
for the requested photo duration, even if the original GIF does not loop. Still
JPEG, PNG and WebP photos continue to fill the same finite photo slot. The
renderer uses FFmpeg's generic `-stream_loop -1` input option and an output `-t`
limit rather than the image2-only `-loop` option. No customer media is silently
skipped or converted to a poster frame.

Image decoder errors stop immediately (`-xerror`), rather than repeatedly trying
a corrupt frame until the ten-minute process timeout. The job error identifies
the media position to check or replace. Failed jobs do not upload partial films;
the temporary directory is removed and the next queued job can run. Timeouts
(and internal `run()` abort signals) wait for the killed process to close before
cleanup. This does not add a public job-cancellation endpoint.

`npm test` now requires `ffmpeg` and `ffprobe` and tests real synthetic JPEG, PNG,
WebP, static/animated GIF, a mislabelled GIF, video, corrupt PNG, film assembly,
silent/music audio, queue recovery, subprocess timeouts/cancellation and temp
cleanup. HTTP and Storage are stubbed, so these tests neither access customers'
media nor prove live Storage authorization. CI also runs the render regressions
inside `Dockerfile.video` with networking disabled, using the same Debian
Bookworm FFmpeg family as production. A separately approved deployment and
controlled live render are still required before claiming production recovery.
