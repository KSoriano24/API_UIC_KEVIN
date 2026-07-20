import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PYTHON_CMD = process.platform === 'win32'
  ? 'C:\\Users\\KEVIN SORIANO\\AppData\\Local\\Programs\\Python\\Python311\\python.exe'
  : 'python3';
const NUMBA_CACHE_DIR = path.join(__dirname, '../uploads/__numba_cache__');
if (!fs.existsSync(NUMBA_CACHE_DIR)) fs.mkdirSync(NUMBA_CACHE_DIR, { recursive: true });

const PYTHON_SERVER_PORT = 8001;
let pythonServerProcess = null;
let servidorListo = false;

export function iniciarServidorPython() {
  if (pythonServerProcess) return;

  const scriptDir = path.join(__dirname, "../python");
  const pythonEnv = { ...process.env, MPLBACKEND: 'Agg', NUMBA_CACHE_DIR };

  pythonServerProcess = spawn(
    PYTHON_CMD,
    ["-m", "uvicorn", "servidor_clasificador:app", "--host", "127.0.0.1", "--port", String(PYTHON_SERVER_PORT)],
    { cwd: scriptDir, env: pythonEnv }
  );

  pythonServerProcess.stdout.on("data", (d) => console.log(`[python-server] ${d}`));
  pythonServerProcess.stderr.on("data", (d) => console.log(`[python-server] ${d}`));

  pythonServerProcess.on("close", (code) => {
    console.error(`Servidor Python de clasificación terminó con código ${code}`);
    pythonServerProcess = null;
    servidorListo = false;
  });
}

async function esperarServidorListo(maxIntentos = 90) {
  for (let i = 0; i < maxIntentos; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PYTHON_SERVER_PORT}/health`);
      if (res.ok) {
        const data = await res.json();
        if (data.model_loaded) {
          servidorListo = true;
          return;
        }
      }
    } catch {
      // aún no responde, seguimos esperando
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("El servidor Python de clasificación no arrancó a tiempo");
}

export const clasificar = async (audioPath) => {
  if (!pythonServerProcess) iniciarServidorPython();
  if (!servidorListo) await esperarServidorListo();

  const respuesta = await fetch(`http://127.0.0.1:${PYTHON_SERVER_PORT}/clasificar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
  });

  if (!respuesta.ok) {
    throw new Error(`Servidor Python respondió con estado ${respuesta.status}`);
  }

  const resultado = await respuesta.json();
  if (resultado.error) {
    throw new Error(`Python reportó error: ${resultado.error}`);
  }
  return resultado;
};
