import { Router } from 'express';
import multer from 'multer';
import { clasificarAudio, obtenerHistorial, descargarReporte, obtenerEstadoPDF } from '../controladores/audioControlador.js';
import { verificarToken } from '../middlewares/verificarToken.js';
import { limitadorDescargaReporte } from '../middlewares/descargaRateLimit.js';

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.enc'];
const MAX_SIZE = 20 * 1024 * 1024;

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, 'uploads/'),
  filename:    (req, file, cb) => cb(null, Date.now() + '-' + file.originalname),
});

const fileFilter = (req, file, cb) => {
  if (ALLOWED_EXTENSIONS.some(e => file.originalname.endsWith(e))) {
    cb(null, true);
  } else {
    cb(new Error('Formato no permitido. Solo MP3, WAV o FLAC.'));
  }
};

const upload = multer({ storage, fileFilter, limits: { fileSize: MAX_SIZE } });

const router = Router();

router.post('/clasificar', upload.single('audio'), verificarToken, clasificarAudio);
router.get('/historial', verificarToken, obtenerHistorial);
router.get('/estado-pdf/:analisis_id', verificarToken, obtenerEstadoPDF);
router.get('/reportes/:analisis_id/descargar', verificarToken, limitadorDescargaReporte, descargarReporte);

export default router;
