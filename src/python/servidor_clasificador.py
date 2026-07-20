import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

# Importa TODO desde tu clasificador.py original, sin tocarlo ni ejecutar su __main__
from clasificador import (
    CNNLSTMAttention,
    build_mel_transform,
    predecir_audio_robusto,
    MODEL_PATH,
    N_MELS,
    SR,
    N_FFT,
    HOP_LENGTH,
)

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
mel_transform = None


@app.on_event("startup")
def cargar_modelo():
    """Se ejecuta UNA sola vez, al arrancar el servidor -- no en cada request."""
    global model, mel_transform
    print(f"[INFO] Dispositivo de ejecución: {device}", file=sys.stderr)

    model = CNNLSTMAttention(n_mels=N_MELS).to(device)
    mel_transform = build_mel_transform(SR, N_FFT, HOP_LENGTH, N_MELS, device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("[INFO] Modelo cargado y listo. Servidor de clasificación disponible.", file=sys.stderr)


class ClasificarRequest(BaseModel):
    audio_path: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/clasificar")
def clasificar_endpoint(req: ClasificarRequest):
    audio_path = req.audio_path

    if not os.path.exists(audio_path):
        return {"error": f"No se encuentra el archivo de audio '{audio_path}'"}

    try:
        resultado = predecir_audio_robusto(audio_path, model, device, mel_transform)
    except Exception as e:
        return {"error": f"Fallo durante la inferencia: {e}"}

    if "error" in resultado:
        return {"error": resultado["error"]}

    # Misma lógica de agregación que tenías en el bloque __main__ de clasificador.py
    total_ventanas = sum(b["ventanas_utiles"] for b in resultado["bloques"])
    if total_ventanas > 0:
        prob_fake = sum(
            b["score_promedio"] * b["ventanas_utiles"] for b in resultado["bloques"]
        ) / total_ventanas
    else:
        prob_fake = float(np.mean([b["score_promedio"] for b in resultado["bloques"]]))

    veredictos = [b["veredicto"] for b in resultado["bloques"]]
    if any(v == "FAKE" for v in veredictos):
        decision = "DEEPFAKE"
    elif any(v == "INCONSISTENTE / MIXTO" for v in veredictos):
        decision = "SOSPECHOSO"
    else:
        decision = "REAL"

    return {
        "archivo": resultado["archivo"],
        "decision": decision,
        "prob_fake": round(float(prob_fake), 4),
        "prob_real": round(1.0 - float(prob_fake), 4),
        "duracion": resultado["duracion_seg"],
        "n_bloques": resultado["n_bloques"],
        "prediccion_detalle": resultado["prediccion"],
        "bloques": resultado["bloques"],
    }
