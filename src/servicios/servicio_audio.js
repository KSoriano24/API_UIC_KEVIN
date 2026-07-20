import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import { Agent } from "undici";

// Nota: Eliminamos child_process y spawn porque ya no necesitamos levantar Python localmente.

// Mantenemos la configuración de directorios por si la necesitas en otras partes de tu app
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 1. URL DE TU HUGGING FACE SPACE (Reemplaza con tu URL directa obtenida en 'Embed this Space')
const HF_SPACE_URL = "https://tu_usuario-tu_space.hf.space/clasificar"; 

// 2. Configuramos un agente de Node para darle tiempo al modelo de procesar el audio sin colgarse
const agentDispatcher = new Agent({
  headersTimeout: 180000, // 3 minutos máximo para recibir respuesta
  bodyTimeout: 180000
});

export function iniciarServidorPython() {
  // Ya no hace nada porque el servidor está 24/7 en Hugging Face,
  // pero la dejamos declarada vacía por si tu app la llama desde el index.js/server.js y no te lance error.
  console.log("[Node] Conectado exitosamente con el servidor remoto en Hugging Face.");
}

export const clasificar = async (audioPath) => {
  try {
    // Verificamos que el archivo de audio exista localmente antes de enviarlo
    if (!fs.existsSync(audioPath)) {
      throw new Error(`El archivo de audio no existe en la ruta: ${audioPath}`);
    }

    // 3. Convertir el archivo local a un FormData para mandarlo por la red
    const formData = new FormData();
    const blob = new Blob([fs.readFileSync(audioPath)]);
    formData.append("file", blob, path.basename(audioPath));

    console.log(`[Node] Enviando ${path.basename(audioPath)} a Hugging Face para análisis...`);

    // 4. Hacemos la petición HTTP a tu Space remoto
    const respuesta = await fetch(HF_SPACE_URL, {
      method: "POST",
      body: formData,
      dispatcher: agentDispatcher // Evita el HeadersTimeoutError
    });

    if (!respuesta.ok) {
      throw new Error(`Hugging Face respondió con estado ${respuesta.status}`);
    }

    const resultado = await respuesta.json();

    if (resultado.error) {
      throw new Error(`Hugging Face reportó error: ${resultado.error}`);
    }

    // Retorna el JSON con los resultados de la clasificación tal como lo espera tu controlador
    return resultado;

  } catch (error) {
    console.error("Error en la clasificación remota:", error.message);
    throw error;
  }
};
