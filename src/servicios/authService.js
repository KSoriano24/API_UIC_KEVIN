import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import dns from 'dns';
import { OAuth2Client } from 'google-auth-library';
import User from '../models/user.js';
import { decryptText } from './crypto.js';

const googleClient = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

// ─── Helpers ──────────────────────────────────────────────────────────────────

function generarToken(user) {
  return jwt.sign(
    { id: user.id, correo: user.correo },
    process.env.JWT_SECRET,
    { expiresIn: '7d' }
  );
}

function perfilPublico(user) {
  return {
    id:        user.id,
    nombre:    user.nombre,
    correo:    user.correo,
    avatar:    user.avatar || null,
    proveedor: user.proveedor,
    rol:       user.rol,
  };
}

async function validarDominioEmail(correo) {
  const regex = /^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$/;
  if (!regex.test(correo)) throw new Error('Formato de correo inválido.');

  const [usuario, dominio] = correo.split('@');
  if (!/[a-zA-Z]/.test(usuario))
    throw new Error('El usuario del correo debe contener letras.');

  try {
    const mx = await dns.promises.resolveMx(dominio);
    if (!mx || mx.length === 0)
      throw new Error(`El dominio "${dominio}" no es un servidor de correo válido.`);
  } catch (err) {
    if (err.message.includes('dominio')) throw err;
    throw new Error(`El dominio "${dominio}" no existe o no recibe correos.`);
  }
}

// ─── Registro con email ───────────────────────────────────────────────────────

export const registrarConEmail = async ({ nombre, correo, contrasena, fechaAcceso }) => {
  nombre     = decryptText(nombre);
  correo     = decryptText(correo);
  contrasena = decryptText(contrasena);

  console.log('fechaAcceso recibida (register):', fechaAcceso);

  if (!nombre || !correo || !contrasena)
    throw new Error('Todos los campos son obligatorios.');

  if (contrasena.length < 8 || !/[A-Z]/.test(contrasena) || !/\d/.test(contrasena))
    throw new Error('La contraseña debe tener mínimo 8 caracteres, una mayúscula y un número.');

  await validarDominioEmail(correo);

  const existe = await User.findByCorreo(correo);
  if (existe) throw new Error('Ya existe una cuenta con ese correo.');

  const hash = await bcrypt.hash(contrasena, 12);
  const user = await User.create({
    nombre,
    correo,
    contrasena: hash,
    proveedor: 'email',
    fechaAcceso
  });

  await User.updateUltimoAcceso(user.id, fechaAcceso);
  return { token: generarToken(user), usuario: perfilPublico(user) };
};

// ─── Login con email ──────────────────────────────────────────────────────────

export const loginConEmail = async ({ correo, contrasena, fechaAcceso }) => {
  correo     = decryptText(correo);
  contrasena = decryptText(contrasena);

  console.log('fechaAcceso recibida (login):', fechaAcceso);

  if (!correo || !contrasena)
    throw new Error('Correo y contraseña son obligatorios.');

  const user = await User.findByCorreo(correo);

  if (!user || user.proveedor !== 'email')
    throw new Error('Credenciales incorrectas.');

  const match = await bcrypt.compare(contrasena, user.contrasena);
  if (!match) throw new Error('Credenciales incorrectas.');

  await User.updateUltimoAcceso(user.id, fechaAcceso);
  return { token: generarToken(user), usuario: perfilPublico(user) };
};

// ─── Google ───────────────────────────────────────────────────────────────────

export const authConGoogle = async ({ credential, fechaAcceso }) => {  // ← agregado fechaAcceso
  if (!credential) throw new Error('Token de Google requerido.');

  console.log('fechaAcceso recibida (google):', fechaAcceso);

  const ticket = await googleClient.verifyIdToken({
    idToken:  credential,
    audience: process.env.GOOGLE_CLIENT_ID,
  });

  const { sub: googleId, email, name, picture } = ticket.getPayload();

  let user = await User.findByCorreo(email);

  if (user) {
    if (!user.google_id) await User.linkGoogle(user.id, googleId);
  } else {
    user = await User.create({
      nombre:    name,
      correo:    email,
      avatar:    picture,
      googleId,
      proveedor: 'google',
      fechaAcceso
    });
  }

  await User.updateUltimoAcceso(user.id, fechaAcceso);  // ← agregado fechaAcceso
  return { token: generarToken(user), usuario: perfilPublico(user) };
};

// ─── Facebook ─────────────────────────────────────────────────────────────────

export const authConFacebook = async ({ accessToken, userId, fechaAcceso }) => {  // ← agregado fechaAcceso
  if (!accessToken || !userId) throw new Error('Token de Facebook requerido.');

  console.log('fechaAcceso recibida (facebook):', fechaAcceso);

  const verifyRes = await fetch(
    `https://graph.facebook.com/debug_token` +
    `?input_token=${accessToken}` +
    `&access_token=${process.env.FACEBOOK_APP_ID}|${process.env.FACEBOOK_APP_SECRET}`
  );
  const { data } = await verifyRes.json();

  if (!data?.is_valid || data.user_id !== userId)
    throw new Error('Token de Facebook inválido.');

  const profileRes = await fetch(
    `https://graph.facebook.com/${userId}?fields=id,name,email,picture&access_token=${accessToken}`
  );
  const profile = await profileRes.json();

  const correoFb = profile.email?.toLowerCase() || `fb_${profile.id}@noemail.com`;

  let user = await User.findByCorreo(correoFb);

  if (user) {
    if (!user.facebook_id) await User.linkFacebook(user.id, profile.id);
  } else {
    user = await User.create({
      nombre:     profile.name,
      correo:     correoFb,
      avatar:     profile.picture?.data?.url,
      facebookId: profile.id,
      proveedor:  'facebook',
      fechaAcceso
    });
  }

  await User.updateUltimoAcceso(user.id, fechaAcceso);  // ← corregido
  return { token: generarToken(user), usuario: perfilPublico(user) };
};