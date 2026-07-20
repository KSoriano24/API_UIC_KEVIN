import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// 1. URL EXACTA DE TU API EN HUGGING FACE SPACE (CORREGIDA)
const HF_SPACE_URL = "https://kevin24sm-api-modelo.hf.space/proxy/7860/clasificar"; 

export function iniciarServidorPython() {
  console.log("[Node] Conectado exitosamente con el servidor remoto en Hugging Face.");
}

export const clasificar = async (audioPath) => {
  try {
    if (!fs.existsSync(audioPath)) {
      throw new Error(`El archivo de audio no existe en la ruta: ${audioPath}`);
    }

    // 2. Crear FormData usando las herramientas nativas de Node v20
    const formData = new FormData();
    const archivoBuffer = fs.readFileSync(audioPath);
    
    // CORRECCIÓN: Añadimos el MIME type explícito para que FastAPI lo reconozca sin problemas
    const blob = new Blob([archivoBuffer], { type: "audio/mpeg" });
    formData.append("file", blob, path.basename(audioPath));

    console.log(`[Node] Enviando ${path.basename(audioPath)} a Hugging Face para análisis...`);

    // 3. Crear un Timeout nativo de 3 minutos (180,000 milisegundos)
    const controladorTimeout = new AbortController();
    const timeoutId = setTimeout(() => controladorTimeout.abort(), 180000);

    // 4. Petición HTTP directa
    const respuesta = await fetch(HF_SPACE_URL, {
      method: "POST",
      body: formData,
      signal: controladorTimeout.signal
    });

    // Limpiamos el temporizador en cuanto el servidor responda
    clearTimeout(timeoutId);

    if (!respuesta.ok) {
      throw new Error(`Hugging Face respondió con estado ${respuesta.status}`);
    }

    const resultado = await respuesta.json();

    if (resultado.error) {
      throw new Error(`Hugging Face reportó error: ${resultado.error}`);
    }

    return resultado;

  } catch (error) {
    if (error.name === "AbortError") {
      console.error("Error: La petición a Hugging Face excedió el tiempo límite (Timeout).");
      throw new Error("El servidor de Hugging Face tardó demasiado en responder.");
    }
    console.error("Error en la clasificación remota:", error.message);
    throw error;
  }
};
