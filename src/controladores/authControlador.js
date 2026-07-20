import { registrarConEmail, loginConEmail, authConGoogle, authConFacebook } from '../servicios/authService.js';

// ─── Registro ─────────────────────────────────────────────────────────────────

export const register = async (req, res) => {
  try {
    const result = await registrarConEmail(req.body);
    return res.status(201).json(result);
  } catch (err) {
    const status = err.message.includes('Ya existe') ? 409 : 400;
    return res.status(status).json({ mensaje: err.message });
  }
};

// ─── Login ────────────────────────────────────────────────────────────────────

export const login = async (req, res) => {
  try {
    const result = await loginConEmail(req.body);
    return res.json(result);
  } catch (err) {
    const status = err.message.includes('incorrectas') ? 401 : 400;
    return res.status(status).json({ mensaje: err.message });
  }
};

// ─── Google ───────────────────────────────────────────────────────────────────

export const googleAuth = async (req, res) => {
  try {
    const result = await authConGoogle(req.body);
    return res.json(result);
  } catch (err) {
    return res.status(401).json({ mensaje: err.message });
  }
};

// ─── Facebook ─────────────────────────────────────────────────────────────────

export const facebookAuth = async (req, res) => {
  try {
    const result = await authConFacebook(req.body);
    return res.json(result);
  } catch (err) {
    return res.status(401).json({ mensaje: err.message });
  }
};