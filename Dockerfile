FROM node:22-bookworm-slim

# canvas@2 compiles from source on Node 22. These packages provide node-gyp,
# Python and the Cairo/Pango image libraries required by the face worker.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    libpixman-1-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
ENV PYTHON=/usr/bin/python3
RUN npm install --omit=dev --no-audit --no-fund

COPY worker.js ./worker.js

ENV NODE_ENV=production

CMD ["node", "worker.js"]
