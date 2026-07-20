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

export const clasificar = (audioPath) => {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, "../python/clasificador.py");
    const pythonEnv = {
      ...process.env,
      MPLBACKEND: 'Agg',
      NUMBA_CACHE_DIR,
    };
    const proceso = spawn(PYTHON_CMD, [pythonScript, audioPath], { env: pythonEnv });
    let stdout = "";
    let stderr = "";

    const timer = setTimeout(() => {
      proceso.kill();
      reject(new Error("Timeout: el proceso de clasificación tardó demasiado (>120s)"));
    }, 120000);

    proceso.stdout.on("data", (data) => { stdout += data.toString(); });
    proceso.stderr.on("data", (data) => { stderr += data.toString(); });

    proceso.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        return reject(new Error(
          `Python terminó con código ${code}. stderr: ${stderr.slice(0, 500)} | stdout: ${stdout.slice(0, 500)}`
        ));
      }
      try {
        const resultado = JSON.parse(stdout.trim());
        if (resultado.error) {
          return reject(new Error(`Python reportó error: ${resultado.error}`));
        }
        resolve(resultado);
      } catch (error) {
        reject(new Error("Error parseando JSON de Python: " + stdout.slice(0, 500)));
      }
    });

    proceso.on("error", (err) => {
      clearTimeout(timer);
      reject(new Error(`No se pudo iniciar Python: ${err.message}`));
    });
  });
};
