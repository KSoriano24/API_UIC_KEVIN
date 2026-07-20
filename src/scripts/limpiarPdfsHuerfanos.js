import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const UPLOADS_DIR = path.resolve(__dirname, '../uploads');

// Patrón de los PDFs temporales generados por generarReportePDF():
const PATRON_PDF_TEMPORAL = /^reporte_\d+_\d+\.pdf$/;

// Debe ser mayor al timeout de generación de PDF (120s) más margen para la subida a Cloudinary.
const EDAD_MAXIMA_MS = 30 * 60 * 1000; // 30 minutos

export function limpiarPdfsHuerfanos() {
  if (!fs.existsSync(UPLOADS_DIR)) return;

  const ahora = Date.now();
  const archivos = fs.readdirSync(UPLOADS_DIR);
  let borrados = 0;

  for (const nombre of archivos) {
    if (!PATRON_PDF_TEMPORAL.test(nombre)) continue;

    const rutaCompleta = path.join(UPLOADS_DIR, nombre);
    try {
      const stats = fs.statSync(rutaCompleta);
      const edadMs = ahora - stats.mtimeMs;

      if (edadMs > EDAD_MAXIMA_MS) {
        fs.unlinkSync(rutaCompleta);
        borrados++;
        console.log(`[limpieza-pdf] Borrado huérfano: ${nombre} (${Math.round(edadMs / 60000)} min de antigüedad)`);
      }
    } catch (err) {
      console.warn(`[limpieza-pdf] No se pudo procesar ${nombre}:`, err.message);
    }
  }

  if (borrados > 0) {
    console.log(`[limpieza-pdf] Limpieza completada: ${borrados} archivo(s) borrado(s).`);
  }
}