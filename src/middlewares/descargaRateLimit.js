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
      : req.ip;
  },
  message: {
    error: 'Demasiadas descargas de reportes en poco tiempo. Intenta de nuevo en unos minutos.',
  },
});