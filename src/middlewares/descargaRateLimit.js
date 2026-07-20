import rateLimit from 'express-rate-limit';

export const limitadorDescargaReporte = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutos
  max: 20,                  // 20 descargas por ventana
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => {
    // Preferimos limitar por usuario autenticado; si no hay token, por IP
    return req.usuario?.id
      ? `user_${req.usuario.id}`
      : ipKeyGenerator(req.ip);
  },
  message: {
    error: 'Demasiadas descargas de reportes en poco tiempo. Intenta de nuevo en unos minutos.',
  },
});

import rateLimit from 'express-rate-limit';

export const audioLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minuto
  max: 5,              // máx 5 análisis por minuto por IP
  standardHeaders: true,
  legacyHeaders: false,
  message: { mensaje: 'Límite de análisis alcanzado. Espera un momento.' }
});

export const estadoPdfLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minuto
  max: 60,              // suficiente para el polling de estado-pdf
  standardHeaders: true,
  legacyHeaders: false,
  message: { mensaje: 'Demasiadas consultas de estado. Espera un momento.' }
});
