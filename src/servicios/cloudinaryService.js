import cloudinary from '../config/cloudinary.js';

/**
 * Sube un PDF (ya escrito en disco por el script Python) a Cloudinary
 * como recurso privado/autenticado, y devuelve su public_id.
 *
 * @param {string} localPath - ruta local del PDF generado por Python
 * @param {number} analisisId - id del análisis, se usa como public_id
 * @returns {Promise<string>} public_id del recurso subido en Cloudinary
 */
export async function subirReportePDF(localPath, analisisId) {
  const resultado = await cloudinary.uploader.upload(localPath, {
    resource_type: 'raw',      
    type: 'authenticated',     
    folder: 'reportes-glowvox',
    public_id: `reporte_${analisisId}`,
    overwrite: true,
    format: 'pdf',
  });
  return resultado.public_id;
}

/**
 * Genera una URL firmada temporal para descargar un PDF privado.
 *
 * @param {string} publicId
 * @param {number} minutosExpiracion
 * @returns {string} URL firmada, válida solo por minutosExpiracion
 */
export function generarUrlDescarga(publicId, minutosExpiracion = 10) {
  const expiresAt = Math.floor(Date.now() / 1000) + minutosExpiracion * 60;
  return cloudinary.utils.private_download_url(publicId, 'pdf', {
    resource_type: 'raw',
    type: 'authenticated',
    expires_at: expiresAt,
  });
}