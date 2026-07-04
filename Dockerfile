FROM node:22-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN printf '%s\n' \
  '{' \
  '  "name": "memory-lane-video-render-worker",' \
  '  "version": "1.0.0",' \
  '  "type": "module",' \
  '  "dependencies": {' \
  '    "@supabase/supabase-js": "^2.49.1"' \
  '  }' \
  '}' > package.json \
  && npm install --omit=dev

COPY video-render-worker.js ./video-render-worker.js

ENV NODE_ENV=production

CMD ["node", "video-render-worker.js"]
