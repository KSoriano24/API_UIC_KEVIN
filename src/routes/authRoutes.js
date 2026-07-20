import { Router } from 'express';
import { body, validationResult } from 'express-validator';
import { register, login, googleAuth, facebookAuth } from '../controladores/authControlador.js';

const router = Router();

// ─── Middleware validador ─────────────────────────────────────────────────────
const validar = (req, res, next) => {
  const errores = validationResult(req);
  if (!errores.isEmpty()) {
    return res.status(400).json({ mensaje: errores.array()[0].msg });
  }
  next();
};

// ─── Reglas de validación ─────────────────────────────────────────────────────
const reglasRegister = [
  body('nombre').notEmpty().withMessage('Nombre requerido').trim().escape(),
  body('correo').notEmpty().withMessage('Correo requerido'),
  body('contrasena').notEmpty().withMessage('Contraseña requerida'),
];

const reglasLogin = [
  body('correo').notEmpty().withMessage('Correo requerido'),
  body('contrasena').notEmpty().withMessage('Contraseña requerida'),
];

// ─── Rutas ────────────────────────────────────────────────────────────────────
router.post('/register', reglasRegister, validar, register);
router.post('/login',    reglasLogin,    validar, login);
router.post('/google',   body('credential').notEmpty(), validar, googleAuth);
router.post('/facebook', body('accessToken').notEmpty().withMessage('Token requerido'), validar, facebookAuth);

export default router;