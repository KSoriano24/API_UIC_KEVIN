import { conmysql } from '../bd.js';

const User = {

  async findByCorreo(correo) {
    const [rows] = await conmysql.execute(
      'SELECT * FROM usuarios WHERE correo = ? LIMIT 1',
      [correo.toLowerCase()]
    );
    return rows[0] || null;
  },

  async findById(id) {
    const [rows] = await conmysql.execute(
      'SELECT * FROM usuarios WHERE id = ? LIMIT 1',
      [id]
    );
    return rows[0] || null;
  },

  async create({ nombre, correo, contrasena = null, proveedor = 'email', googleId = null, facebookId = null, avatar = null, fechaAcceso = null }) {

    const pad = (n) => n.toString().padStart(2, '0');
    const ahora = new Date();
    const fechaCreacion = fechaAcceso || `${ahora.getFullYear()}-${pad(ahora.getMonth() + 1)}-${pad(ahora.getDate())} ${pad(ahora.getHours())}:${pad(ahora.getMinutes())}:${pad(ahora.getSeconds())}`;

    const [result] = await conmysql.execute(
      `INSERT INTO usuarios (nombre, correo, contrasena, proveedor, google_id, facebook_id, avatar, creado_en)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [nombre, correo.toLowerCase(), contrasena, proveedor, googleId, facebookId, avatar, fechaCreacion]
    );
    return User.findById(result.insertId);
  },

  async linkGoogle(id, googleId) {
    await conmysql.execute(
      'UPDATE usuarios SET google_id = ?, proveedor = ? WHERE id = ?',
      [googleId, 'google', id]
    );
  },

  async linkFacebook(id, facebookId) {
    await conmysql.execute(
      'UPDATE usuarios SET facebook_id = ? WHERE id = ?',
      [facebookId, id]
    );
  },

  async updateUltimoAcceso(id, fechaAcceso = null) {
    const fecha = fechaAcceso || (() => {
      const ahora = new Date();
      const pad = (n) => n.toString().padStart(2, '0');
      return `${ahora.getFullYear()}-${pad(ahora.getMonth() + 1)}-${pad(ahora.getDate())} ${pad(ahora.getHours())}:${pad(ahora.getMinutes())}:${pad(ahora.getSeconds())}`;
    })();

    await conmysql.execute(
      'UPDATE usuarios SET ultimo_acceso = ? WHERE id = ?',
      [fecha, id]
    );
  },
};

export default User;