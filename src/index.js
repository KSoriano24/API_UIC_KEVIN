import app from "./app.js";
import { PORT } from './config.js';
import cron from 'node-cron';
import { limpiarPdfsHuerfanos } from './scripts/limpiarPdfsHuerfanos.js';

// ─── Limpieza de PDFs huérfanos ────────────────────────────────────────────

cron.schedule('0 * * * *', limpiarPdfsHuerfanos);

// ─── Arranque del servidor ──────────────────────────────────────────────────
app.listen(PORT, '0.0.0.0', () => {
  console.log('El servidor está escuchando por el puerto:', PORT);
});