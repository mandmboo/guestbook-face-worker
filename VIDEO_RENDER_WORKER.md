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
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
RENDER_OUTPUT_BUCKET=rendered-films
MAX_CONCURRENT_RENDERS=1
```

`RENDER_SECRET` must match the value used in Vercel as `MEMORY_FILM_RENDER_SECRET`.

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
