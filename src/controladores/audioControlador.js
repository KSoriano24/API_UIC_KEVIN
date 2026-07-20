import jwt from 'jsonwebtoken';
import { clasificar } from '../servicios/servicio_audio.js';
import { decryptFile } from '../servicios/crypto.js';
import { subirReportePDF, generarUrlDescarga } from '../servicios/cloudinaryService.js';
import { conmysql } from '../bd.js';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import fs from 'fs';
import path from 'path';
import { parseFile } from 'music-metadata';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const UPLOADS_DIR = path.resolve(__dirname, '../uploads');
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// ─── Reglas de validación de audio ─────────────────────────────────────────
const FORMATOS_PERMITIDOS = ['.wav', '.flac', '.mp3', '.opus', '.ogg'];
const MIN_DURACION_SEG = 4.0;
const MAX_DURACION_SEG = 300.0; 

// ─── Helper: fecha actual en zona horaria de Guayaquil ────────────────────
const fechaAhora = () =>
  new Date()
    .toLocaleString('sv-SE', { timeZone: 'America/Guayaquil' })
    .replace('T', ' ');

// ─── Helper: extrae y verifica el usuario del header Authorization ────────
function obtenerUsuarioDeToken(req) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;

  const token = authHeader.split(' ')[1];
  if (!token) return null;

  try {
    return jwt.verify(token, process.env.JWT_SECRET);
  } catch {
    return null;
  }
}

// ─── Helper: valida extensión contra los formatos soportados ──────────────
function validarFormato(nombreArchivo) {
  const ext = path.extname(nombreArchivo).toLowerCase();
  return FORMATOS_PERMITIDOS.includes(ext);
}

// ─── Helper: obtiene la duración de un audio en segundos ──────────────────
async function obtenerDuracionSegundos(audioPath) {
  const metadata = await parseFile(audioPath);
  const duracion = metadata.format.duration;
  if (typeof duracion !== 'number' || isNaN(duracion)) {
    throw new Error('No se pudo determinar la duración del audio');
  }
  return duracion;
}

// ─── Helper: sube el PDF local a Cloudinary y borra el archivo local ──────
async function subirYLimpiar(reportePathLocal, analisisId) {
  const publicId = await subirReportePDF(reportePathLocal, analisisId);
  fs.unlink(reportePathLocal, (err) => {
    if (err) console.warn(`No se pudo borrar PDF local temporal: ${reportePathLocal}`, err.message);
  });
  return publicId;
}

// ─── Helper: descarga el binario del PDF desde Cloudinary y lo envía ──────
async function enviarPDFDesdeCloudinary(res, publicId, analisisId) {
  const urlFirmada = generarUrlDescarga(publicId);
  const respuesta = await fetch(urlFirmada);

  if (!respuesta.ok) {
    throw new Error(`Cloudinary respondió ${respuesta.status} al descargar ${publicId}`);
  }

  const buffer = Buffer.from(await respuesta.arrayBuffer());
  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', `attachment; filename="reporte_glowvox_${analisisId}.pdf"`);
  res.send(buffer);
}

// ─── Clasificar audio ──────────────────────────────────────────────────────
export const clasificarAudio = async (req, res) => {
  let decryptedPath = null;

  try {
    const audio = req.file;
    if (!audio) return res.status(400).json({ error: 'Debes subir un archivo de audio' });

    const nombreOriginal = audio.originalname.replace('.enc', '');

    if (!validarFormato(nombreOriginal)) {
      fs.unlink(audio.path, () => { });
      return res.status(400).json({
        error: `Formato no soportado. Formatos permitidos: ${FORMATOS_PERMITIDOS.join(', ')}.`
      });
    }

    let usuarioId = null;
    let usuarioNombre = 'Usuario';

    const payload = obtenerUsuarioDeToken(req);
    if (payload) {
      usuarioId = payload.id;
      const [urows] = await conmysql.execute(
        'SELECT nombre FROM usuarios WHERE id = ?', [usuarioId]
      );
      if (urows.length) usuarioNombre = urows[0].nombre;
    }

    let audioPath = path.resolve(audio.path);

    if (audio.originalname.endsWith('.enc')) {
      const encryptedBuffer = fs.readFileSync(audioPath);
      const decryptedBuffer = decryptFile(encryptedBuffer);

      const decryptedFilename = `${Date.now()}-dec-${nombreOriginal}`;
      const decryptedAbsPath = path.join(UPLOADS_DIR, decryptedFilename);

      fs.writeFileSync(decryptedAbsPath, decryptedBuffer);
      fs.unlinkSync(audioPath);

      audioPath = decryptedAbsPath;
      decryptedPath = decryptedAbsPath;
    }

    if (!fs.existsSync(audioPath)) {
      return res.status(500).json({ error: 'Error preparando el archivo de audio' });
    }

    let duracionCruda;
    try {
      duracionCruda = await obtenerDuracionSegundos(audioPath);
    } catch (err) {
      fs.unlink(audioPath, () => { });
      return res.status(400).json({ error: 'No se pudo leer el archivo de audio. ¿Está corrupto?' });
    }

    if (duracionCruda < MIN_DURACION_SEG) {
      fs.unlink(audioPath, () => { });
      return res.status(400).json({
        error: `El audio dura ${duracionCruda.toFixed(1)}s. El mínimo permitido es ${MIN_DURACION_SEG}s.`
      });
    }

    if (duracionCruda > MAX_DURACION_SEG) {
      fs.unlink(audioPath, () => { });
      return res.status(400).json({
        error: `El audio dura ${duracionCruda.toFixed(1)}s. El máximo permitido es ${MAX_DURACION_SEG}s (5 minutos). Divide el archivo en segmentos más cortos.`
      });
    }

    const resultado = await clasificar(audioPath);

    const prediccion = resultado.decision === 'REAL' ? 'real' : 'falso';
    const veredicto = resultado.decision; // 'REAL' | 'DEEPFAKE' | 'SOSPECHOSO'
    const confianza = resultado.decision === 'REAL'
      ? resultado.prob_real
      : resultado.prob_fake;
    const duracion = resultado.duracion ?? 0;

    const fechaAnalisis = fechaAhora();

    let insertId = null;
    let reportePublicId = null;

    if (usuarioId) {
      const [dbResult] = await conmysql.execute(
        `INSERT INTO analisis_audios 
        (usuario_id, nombre_audio, url_audio, estado, prediccion, veredicto, confianza, modelo_usado, duracion_segundos, analizado_en)
        VALUES (?, ?, ?, 'procesado', ?, ?, ?, 'GlowVox-v1', ?, ?)`,
        [usuarioId, nombreOriginal, audioPath, prediccion, veredicto, confianza, duracion, fechaAnalisis]
      );
      insertId = dbResult.insertId;

      const nivelRiesgo = veredicto === 'DEEPFAKE' ? 'alto' : (veredicto === 'SOSPECHOSO' ? 'medio' : 'bajo');

      const resumenPorVeredicto = {
        DEEPFAKE: `Audio clasificado como DEEPFAKE con ${(confianza * 100).toFixed(1)}% de confianza.`,
        SOSPECHOSO: `Audio clasificado como SOSPECHOSO (resultado no concluyente) con ${(confianza * 100).toFixed(1)}% de confianza.`,
        REAL: `Audio clasificado como REAL con ${(confianza * 100).toFixed(1)}% de confianza.`,
      };
      const recomendacionesPorVeredicto = {
        DEEPFAKE: 'No difundir este audio. Se recomienda verificar la fuente original.',
        SOSPECHOSO: 'Resultado no concluyente. Se recomienda un análisis adicional antes de sacar conclusiones.',
        REAL: 'Audio auténtico. No se detectaron señales de manipulación.',
      };
      const resumen = resumenPorVeredicto[veredicto];
      const recomendaciones = recomendacionesPorVeredicto[veredicto];

      await conmysql.execute(
        `INSERT INTO reportes
         (usuario_id, analisis_id, resumen, nivel_riesgo, recomendaciones, reporte_public_id, creado_en, bloques_json)
         VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)`,
        [usuarioId, insertId, resumen, nivelRiesgo, recomendaciones, JSON.stringify(resultado.bloques || [])]
      );

      // ─── PDF: ahora se ESPERA (await) a que termine, para que el   ───
      // ─── frontend pueda descargar de inmediato al recibir la respuesta ──
      if (fs.existsSync(audioPath)) {
        const fechaGeneracion = fechaAhora();

        const pdfData = {
          audio_path: audioPath,
          nombre_audio: nombreOriginal,
          decision: resultado.decision,
          prob_real: resultado.prob_real,
          prob_fake: resultado.prob_fake,
          duracion: resultado.duracion,
          bloques: resultado.bloques,
          usuario: usuarioNombre,
          analisis_id: insertId,
          fecha: fechaGeneracion,
        };

        try {
          console.time(`pdf-gen-${insertId}`);
          const reportePathLocal = await generarReportePDF(pdfData);
          console.timeEnd(`pdf-gen-${insertId}`);

          if (reportePathLocal) {
            console.time(`cloudinary-upload-${insertId}`);
            const publicId = await subirYLimpiar(reportePathLocal, insertId);
            console.timeEnd(`cloudinary-upload-${insertId}`);

            const fechaPDF = fechaAhora();

            await conmysql.execute(
              'UPDATE reportes SET reporte_public_id = ?, creado_en = ? WHERE analisis_id = ?',
              [publicId, fechaPDF, insertId]
            );

            reportePublicId = publicId;
            console.log(`PDF subido a Cloudinary para analisis_id=${insertId}: ${publicId}`);
          } else {
            console.warn(`PDF no generado para analisis_id=${insertId}`);
          }
        } catch (err) {
          // No tumbamos la respuesta completa: el análisis ya se guardó,
          // solo el PDF falló. El usuario podrá reintentar la descarga
          // y el flujo de "regenerar" en descargarReporte se hará cargo.
          console.error('Error generando/subiendo PDF:', err);
        }

      } else {
        console.warn(`Audio no encontrado para generar PDF: ${audioPath}`);
      }
    }

    return res.json({
      mensaje: 'Clasificación realizada',
      analisis_id: insertId,
      reporte_id: reportePublicId ? insertId : null,
      pdf_listo: !!reportePublicId,
      ...resultado
    });

  } catch (error) {
    if (req.file?.path && fs.existsSync(req.file.path)) {
      fs.unlink(req.file.path, () => { });
    }
    console.error('Error en clasificarAudio:', error);
    res.status(500).json({ error: 'Error al clasificar el audio' });
  }
};

// ─── Generar PDF (local, en /uploads) ──────────────────────────────────────
export const generarReportePDF = (data) => {
  return new Promise((resolve) => {
    const outputPath = path.join(UPLOADS_DIR, `reporte_${data.analisis_id}_${Date.now()}.pdf`);
    const pythonScript = path.join(__dirname, '../python/generar_reporte.py');

    const pythonCmd = process.platform === 'win32'
      ? 'C:\\Users\\KEVIN SORIANO\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
      : 'python3';

    if (!fs.existsSync(pythonScript)) {
      console.error(`Script Python no encontrado: ${pythonScript}`);
      return resolve(null);
    }

    if (!fs.existsSync(data.audio_path)) {
      console.error(`Audio no encontrado para PDF: ${data.audio_path}`);
      return resolve(null);
    }

    console.log(`Iniciando Python para analisis_id=${data.analisis_id}`);

    const pythonEnv = {
      ...process.env,
      MPLBACKEND: 'Agg',
      NUMBA_CACHE_DIR: path.join(__dirname, '../uploads/__numba_cache__'),
    };

    const proceso = spawn(pythonCmd, [
      pythonScript,
      JSON.stringify(data),
      outputPath
    ], { env: pythonEnv });

    let stdout = '';
    let stderr = '';
    proceso.stdout.on('data', d => { stdout += d.toString(); });
    proceso.stderr.on('data', d => { stderr += d.toString(); });

    const timer = setTimeout(() => {
      proceso.kill();
      console.error(`PDF timeout para analisis_id=${data.analisis_id}`);
      resolve(null);
    }, 120000);

    proceso.on('close', (code) => {
      clearTimeout(timer);

      if (stderr.trim()) {
        console.log(`Python stderr (analisis_id=${data.analisis_id}):\n${stderr.slice(0, 1000)}`);
      }

      if (code !== 0) {
        console.error(`Python salió con código ${code}`);
        console.error(`   stdout: ${stdout.slice(0, 500)}`);
        return resolve(null);
      }

      try {
        const resultado = JSON.parse(stdout.trim());
        if (resultado.ok && fs.existsSync(outputPath)) {
          console.log(`PDF generado: ${outputPath}`);
          resolve(outputPath);
        } else {
          console.error(`Python reportó error: ${JSON.stringify(resultado)}`);
          resolve(null);
        }
      } catch {
        const match = stdout.match(/\{.*\}/s);
        if (match) {
          try {
            const resultado = JSON.parse(match[0]);
            if (resultado.ok && fs.existsSync(outputPath)) {
              resolve(outputPath);
              return;
            }
          } catch { }
        }
        console.error(`No se pudo parsear stdout de Python: ${stdout.slice(0, 300)}`);
        resolve(null);
      }
    });

    proceso.on('error', (err) => {
      clearTimeout(timer);
      console.error(`No se pudo iniciar Python:`, err.message);
      resolve(null);
    });
  });
};

// ─── Obtener historial ─────────────────────────────────────────────────────
export const obtenerHistorial = async (req, res) => {
  try {
    const payload = req.usuario ?? obtenerUsuarioDeToken(req);
    if (!payload) return res.status(401).json({ error: 'No autorizado' });

    const [rows] = await conmysql.execute(
      `SELECT 
     a.id,
     a.nombre_audio,
     a.prediccion,
     a.veredicto,
     a.confianza,
     a.modelo_usado,
     DATE_FORMAT(a.analizado_en, '%Y-%m-%dT%H:%i:%s') AS analizado_en,
     a.duracion_segundos,
     r.id          AS reporte_id,
     r.reporte_public_id,
     r.nivel_riesgo,
     r.resumen,
     DATE_FORMAT(r.creado_en, '%Y-%m-%dT%H:%i:%s') AS pdf_creado_en
    FROM analisis_audios a
    LEFT JOIN reportes r ON r.analisis_id = a.id
    WHERE a.usuario_id = ?
    ORDER BY a.analizado_en DESC
    LIMIT 50`,
      [payload.id]
    );

    return res.json({ historial: rows });

  } catch (error) {
    console.error('Error obteniendo historial:', error);
    res.status(500).json({ error: 'Error al obtener historial' });
  }
};

// ─── Descargar reporte ─────────────────────────────────────────────────────
export const descargarReporte = async (req, res) => {
  try {
    const payload = req.usuario ?? obtenerUsuarioDeToken(req);
    if (!payload) return res.status(401).json({ error: 'No autorizado' });

    const { analisis_id } = req.params;

    const [rows] = await conmysql.execute(
      `SELECT 
      a.*,
      u.nombre  AS usuario_nombre,
      r.id      AS reporte_id,
      r.reporte_public_id,
      r.resumen,
      r.nivel_riesgo,
      r.recomendaciones,
      r.bloques_json
      FROM analisis_audios a
      JOIN usuarios u ON u.id = a.usuario_id
      LEFT JOIN reportes r ON r.analisis_id = a.id
      WHERE a.id = ? AND a.usuario_id = ?`,
      [analisis_id, payload.id]
    );

    if (!rows.length) return res.status(404).json({ error: 'Análisis no encontrado' });

    const analisis = rows[0];

    // Caso 1: ya existe un PDF subido a Cloudinary (reporte_public_id)
    if (analisis.reporte_public_id) {
      try {
        await enviarPDFDesdeCloudinary(res, analisis.reporte_public_id, analisis_id);
        return;
      } catch (err) {
        console.error('Error obteniendo PDF de Cloudinary, se regenerará:', err.message);
        
      }
    }

    if (!analisis.url_audio || !fs.existsSync(analisis.url_audio)) {
      return res.status(404).json({ error: 'Audio no disponible para regenerar reporte' });
    }

    let bloquesGuardados = [];
    try {
      bloquesGuardados = analisis.bloques_json ? JSON.parse(analisis.bloques_json) : [];
    } catch {
      bloquesGuardados = [];
    }

    const decisionCompleta = analisis.veredicto
      ?? (analisis.prediccion === 'real' ? 'REAL' : 'DEEPFAKE');

    const probFakeGuardado = decisionCompleta === 'REAL'
      ? 1 - analisis.confianza
      : analisis.confianza;
    const probRealGuardado = 1 - probFakeGuardado;

    const fechaGeneracion = fechaAhora();

    const reportePathLocal = await generarReportePDF({
      audio_path: analisis.url_audio,
      nombre_audio: analisis.nombre_audio,
      decision: decisionCompleta,
      prob_real: probRealGuardado,
      prob_fake: probFakeGuardado,
      usuario: analisis.usuario_nombre,
      analisis_id: analisis.id,
      fecha: fechaGeneracion,
      bloques: bloquesGuardados,
    });

    if (!reportePathLocal) {
      return res.status(500).json({ error: 'Error generando reporte' });
    }

    // Sube el PDF recién generado a Cloudinary y limpia el temporal local
    let publicId;
    try {
      publicId = await subirYLimpiar(reportePathLocal, analisis.id);
    } catch (err) {
      console.error('Error subiendo PDF regenerado a Cloudinary:', err);
      return res.status(500).json({ error: 'Error subiendo el reporte' });
    }

    const fechaPDF = fechaAhora();

    if (analisis.reporte_id) {
      await conmysql.execute(
        'UPDATE reportes SET reporte_public_id = ?, creado_en = ? WHERE id = ?',
        [publicId, fechaPDF, analisis.reporte_id]
      );
    } else {
      await conmysql.execute(
        `INSERT INTO reportes (usuario_id, analisis_id, resumen, nivel_riesgo, recomendaciones, reporte_public_id, creado_en, bloques_json)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          analisis.usuario_id,
          analisis.id,
          analisis.resumen ?? '',
          analisis.nivel_riesgo ?? 'bajo',
          analisis.recomendaciones ?? '',
          publicId,
          fechaPDF,
          JSON.stringify(bloquesGuardados),
        ]
      );
    }

    await enviarPDFDesdeCloudinary(res, publicId, analisis_id);

  } catch (error) {
    console.error('Error descargando reporte:', error);
    res.status(500).json({ error: 'Error interno' });
  }
};
