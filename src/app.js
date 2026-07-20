import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';

import audioRoutes from './routes/audioRoutes.js';
import authRoutes from './routes/authRoutes.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// ─── Helmet — cabeceras de seguridad HTTP ─────────────────────────────────────
app.use(helmet());
app.use(helmet.contentSecurityPolicy({
  directives: {
    defaultSrc: ["'self'"],
    scriptSrc:  ["'self'"],
    objectSrc:  ["'none'"],
    upgradeInsecureRequests: [],
  }
}));

// ─── CORS restrictivo ─────────────────────────────────────────────────────────
const allowedOrigins = [
  'http://localhost:8100',      // Ionic 
  'http://localhost:4200',      // Angular 
  'https://tudominio.com',      // Dominio real
];

app.use(cors({
  origin: (origin, callback) => {
    // Permite requests sin origin
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Bloqueado por CORS'));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],
  credentials: true,
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

// ─── Rate limiting global ─────────────────────────────────────────────────────
const globalLimiter = rateLimit({
  windowMs: 10 * 60 * 1000, // 10 minutos
  max: 100,                  // máx 100 requests por IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { mensaje: 'Demasiadas solicitudes, intenta más tarde.' }
});
app.use(globalLimiter);

// ─── Rate limiting estricto para auth ────────────────────────────────────────
const authLimiter = rateLimit({
  windowMs: 5 * 60 * 1000, 
  max: 10,                   // máx 10 intentos de login por IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { mensaje: 'Demasiados intentos de acceso. Espera 5 minutos.' }
});

const audioLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minuto
  max: 5,              // máx 5 análisis por minuto por IP
  message: { mensaje: 'Límite de análisis alcanzado. Espera un momento.' }
});

// ─── Body parsers ─────────────────────────────────────────────────────────────
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true, limit: '1mb' }));

// ─── Rutas ───────────────────────────────────────────────────────────────────
app.use('/api/auth', authLimiter, authRoutes);
app.use('/api/audio', audioLimiter, audioRoutes);
// ─── 404 ──────────────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({ mensaje: 'Endpoint no encontrado' });
});

// ─── Error global ─────────────────────────────────────────────────────────────
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ mensaje: 'Error interno del servidor' });
});

export default app;