

FROM node:20-bookworm

# ── Python + pip ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencias de Node ────────────────────────────────────────────────────
COPY package*.json ./
RUN npm install --omit=dev

# ── Dependencias de Python ──────────────────────────────────────────────────
COPY python/requirements.txt ./python/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r src/python/requirements.txt
# ── Resto del código ─────────────────────────────────────────────────────────
COPY . .

# Carpeta de temporales (efímera en Render, se recrea en cada deploy)
RUN mkdir -p uploads

# Render inyecta PORT; exponemos el default por documentación
EXPOSE 10000

CMD ["node", "index.js"]
