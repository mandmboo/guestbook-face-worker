FROM node:20-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY video-render-worker.js ./video-render-worker.js

ENV NODE_ENV=production

CMD ["node", "video-render-worker.js"]
