import os
import sys
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import soxr
import json

# Formatos de audio aceptados
FORMATOS_PERMITIDOS = {".wav", ".flac", ".mp3", ".opus", ".ogg"}

# 1. CONFIGURACIÓN CLAVE -- alineada con el script de entrenamiento
SR = 16000
WINDOW_SEC = 5.0
OVERLAP = 0.5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
MAX_WINDOWS_PER_FILE = 20      
RMS_NORMALIZE = True           
TARGET_RMS = 0.1               

# Filtro de energía
# Se aplica DESPUES de la normalización RMS, para no descartar ventanas
UMBRAL_ENERGIA_MINIMA = 0.005

TOP_DB_TRIM = 30.0

# Umbral de decisión: score >= umbral -> FAKE, score < umbral -> REAL.
UMBRAL_DECISION = 0.5

# Si la desviación estándar entre ventanas supera esto, el resultado se marca
# como INCONSISTENTE en vez de forzarse a una clase
UMBRAL_VARIABILIDAD_REVISION = 0.15

MIN_VENTANAS_CONFIABLE = 2

# Límites de duración
MIN_DURACION_SEG = 4.0

# Duración máxima que se acepta evaluar en una sola corrida
MAX_DURACION_TOTAL_SEG = 300.0  # 5 minutos

# Duración de cada bloque de evaluación independiente
_stride_seg = WINDOW_SEC * (1 - OVERLAP)
BLOCK_DURATION_SEG = WINDOW_SEC + (MAX_WINDOWS_PER_FILE - 1) * _stride_seg  # 52.5s


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")


# 2. DEFINICIÓN DE LA ARQUITECTURA
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        weights = torch.softmax(self.score(seq).squeeze(-1), dim=1)
        context = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return context, weights


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 1), dropout_cnn=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(kernel_size=pool)
        self.drop = nn.Dropout2d(dropout_cnn) if dropout_cnn > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(torch.relu(self.bn(self.conv(x)))))


class CNNLSTMAttention(nn.Module):
    def __init__(self, n_mels=128, in_channels=3, cnn_channels=(32, 64, 128),
                 lstm_hidden=128, lstm_layers=2, dropout=0.4, dropout_cnn=0.2):
        super().__init__()
        blocks = []
        in_ch = in_channels
        for out_ch in cnn_channels:
            blocks.append(ConvBlock(in_ch, out_ch, pool=(2, 1), dropout_cnn=dropout_cnn))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        freq_after = n_mels // (2 ** len(cnn_channels))
        lstm_input_dim = in_ch * freq_after

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2
        self.attention = Attention(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights


# 3. CARGA Y PREPROCESAMIENTO
def read_audio_any(filepath: str):
    
    try:
        audio, sr = sf.read(str(filepath), dtype="float32", always_2d=False)
    except Exception:
        audio, sr = librosa.load(str(filepath), sr=None, mono=False)
    if audio.ndim > 1:
        axis = -1 if audio.shape[-1] < audio.shape[0] else 0
        audio = audio.mean(axis=axis).astype(np.float32)
    return audio.astype(np.float32), sr


def resample_high_quality(audio: np.ndarray, sr_in: int, sr_out: int):
    if sr_in == sr_out:
        return audio.astype(np.float32)
    return soxr.resample(audio, sr_in, sr_out, quality="VHQ").astype(np.float32)


def preparar_audio_como_entrenamiento(filepath: str, target_sr: int, top_db: float):
    
    audio, sr_in = read_audio_any(filepath)
    resampled = resample_high_quality(audio, sr_in, target_sr)
    trimmed, _ = librosa.effects.trim(resampled, top_db=top_db)
    return trimmed


# 4. VENTANEO Y ESPECTROGRAMA 

def plan_windows(n_samples: int, sr: int, window_sec: float, overlap: float,
                  max_windows: int = MAX_WINDOWS_PER_FILE):
    
    window_len = int(window_sec * sr)
    if n_samples == 0:
        return []
    if n_samples <= window_len:
        return [(0, n_samples, True)]

    stride = max(1, int(window_len * (1 - overlap)))
    plans = []
    start = 0
    while start < n_samples:
        end = start + window_len
        if end <= n_samples:
            plans.append((start, end, False))
        else:
            plans.append((start, n_samples, True))
        start += stride
        if end >= n_samples:
            break
        if len(plans) >= max_windows:
            break
    return plans


def rms_normalize_scale(audio: np.ndarray, target_rms: float = TARGET_RMS) -> float:
    
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return 1.0
    return float(target_rms / rms)


def load_window_from_array(audio_full, start, stop, tile, window_len):
    segment = audio_full[start:stop]
    if tile:
        if len(segment) == 0:
            segment = np.zeros(window_len, dtype=np.float32)
        else:
            reps = int(np.ceil(window_len / len(segment)))
            segment = np.tile(segment, reps)[:window_len]
    return segment


def build_mel_transform(sr: int, n_fft: int, hop_length: int, n_mels: int, device):
    
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        f_min=0, f_max=sr // 2, power=2.0,
    ).to(device)


def compute_log_mel_delta(audio: np.ndarray, mel_transform, device):
    
    audio_t = torch.from_numpy(audio).float().to(device)
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)

    mel = mel_transform(audio_t)
    log_mel = torch.log(mel + 1e-6)

    delta1 = torchaudio.functional.compute_deltas(log_mel)
    delta2 = torchaudio.functional.compute_deltas(delta1)

    spec = torch.cat([log_mel, delta1, delta2], dim=0)  # (3, n_mels, n_frames)
    spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    return spec


def evaluar_bloque(audio_bloque, rms_scale, mel_transform, device, model):
    
    n_samples = len(audio_bloque)
    window_len = int(WINDOW_SEC * SR)
    plans = plan_windows(n_samples, SR, WINDOW_SEC, OVERLAP, MAX_WINDOWS_PER_FILE)

    if not plans:
        return None

    window_tensors = []
    valid_window_indices = []

    for idx, (start, stop, tile) in enumerate(plans):
        segment = load_window_from_array(audio_bloque, start, stop, tile, window_len)
        segment = segment * rms_scale
        rms_energia = np.sqrt(np.mean(segment ** 2))

        if rms_energia >= UMBRAL_ENERGIA_MINIMA:
            spec = compute_log_mel_delta(segment, mel_transform, device)
            window_tensors.append(spec.cpu())
            valid_window_indices.append(idx)

    if not window_tensors:
        for start, stop, tile in plans:
            segment = load_window_from_array(audio_bloque, start, stop, tile, window_len)
            segment = segment * rms_scale
            spec = compute_log_mel_delta(segment, mel_transform, device)
            window_tensors.append(spec.cpu())
            valid_window_indices.append(len(window_tensors) - 1)

    batch_tensor = torch.stack(window_tensors).to(device)
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(batch_tensor)
        scores_ventanas = torch.sigmoid(logits).cpu().numpy()
        attn_weights = attn_weights.cpu().numpy()  # (n_ventanas, n_frames)

    sec_por_frame = HOP_LENGTH / SR  # 256/16000 = 0.016s
    picos_seg, picos_peso, picos_ratio = [], [], []
    for w in attn_weights:
        idx_pico = int(np.argmax(w))
        picos_seg.append(round(idx_pico * sec_por_frame, 4))
        picos_peso.append(float(w[idx_pico]))
        picos_ratio.append(float(w[idx_pico] * len(w))) 

    score_promedio = float(np.mean(scores_ventanas))
    desviacion_estandar = float(np.std(scores_ventanas)) if len(scores_ventanas) > 1 else 0.0

    votos_fake = sum(1 for s in scores_ventanas if s >= UMBRAL_DECISION)
    votos_real = len(scores_ventanas) - votos_fake

    requiere_revision = (
        desviacion_estandar >= UMBRAL_VARIABILIDAD_REVISION and len(scores_ventanas) > 1
    )
    duracion_insuficiente = len(scores_ventanas) < MIN_VENTANAS_CONFIABLE

    total_votos = votos_fake + votos_real
    if requiere_revision:
        veredicto = "INCONSISTENTE / MIXTO"
        confianza = max(votos_fake, votos_real) / total_votos * 100
    elif votos_fake >= votos_real:
        veredicto = "FAKE"
        confianza = (votos_fake / total_votos) * 100
    else:
        veredicto = "REAL"
        confianza = (votos_real / total_votos) * 100

    return {
        "ventanas_totales": len(plans),
        "ventanas_utiles": len(scores_ventanas),
        "scores": scores_ventanas.tolist(),
        "atencion_pico_seg": picos_seg,
        "atencion_pico_peso": picos_peso,
        "atencion_ratio_concentracion": picos_ratio,        
        "score_promedio": score_promedio,
        "std_desviacion": desviacion_estandar,
        "veredicto": veredicto,
        "confianza": round(confianza, 2),
        "requiere_revision": requiere_revision,
        "duracion_insuficiente": duracion_insuficiente,
    }


# 5. INFERENCIA
def predecir_audio_robusto(audio_path, model, device, mel_transform):
    # Preprocesado idéntico al de entrenamiento s(resample soxr VHQ + trim de bordes)
    audio_full = preparar_audio_como_entrenamiento(audio_path, SR, TOP_DB_TRIM)

    duracion_seg = len(audio_full) / SR

    if duracion_seg < MIN_DURACION_SEG:
        return {
            "archivo": os.path.basename(audio_path),
            "error": (
                f"Audio de {duracion_seg:.2f}s (post-trim de silencio) por debajo del "
                f"mínimo permitido de {MIN_DURACION_SEG:.0f}s. No se evalúa: no hay "
                f"evidencia suficiente para un veredicto confiable."
            ),
        }

    # Límite superior: no se evalúan audios de media hora/una hora.
    if duracion_seg > MAX_DURACION_TOTAL_SEG:
        return {
            "archivo": os.path.basename(audio_path),
            "error": (
                f"Audio de {duracion_seg:.1f}s excede el máximo permitido de "
                f"{MAX_DURACION_TOTAL_SEG:.0f}s para evaluación en producción. "
                f"Divide el archivo en segmentos más cortos antes de evaluarlo."
            ),
        }

    # Normalización RMS idéntica a entrenamiento
    rms_scale = rms_normalize_scale(audio_full, TARGET_RMS) if RMS_NORMALIZE else 1.0

    block_len = int(BLOCK_DURATION_SEG * SR)
    n_samples = len(audio_full)

    bloques_resultado = []
    for i, start in enumerate(range(0, n_samples, block_len)):
        stop = min(start + block_len, n_samples)
        audio_bloque = audio_full[start:stop]
        r = evaluar_bloque(audio_bloque, rms_scale, mel_transform, device, model)
        if r is None:
            continue
        r["bloque_idx"] = i
        r["inicio_seg"] = round(start / SR, 2)
        r["fin_seg"] = round(stop / SR, 2)
        bloques_resultado.append(r)

        print(f"\n[DEBUG] BLOQUE {i+1} ({r['inicio_seg']}s - {r['fin_seg']}s):", file=sys.stderr)
        for idx, score in enumerate(r["scores"]):
            etiqueta = "FAKE" if score >= UMBRAL_DECISION else "REAL"
            print(f" -> Ventana {idx+1:02d}: Score = {score:.6f} | Evaluación = {etiqueta}", file=sys.stderr)
        print(f"    Veredicto del bloque: {r['veredicto']} ({r['confianza']}%)", file=sys.stderr)

    if not bloques_resultado:
        raise ValueError("El audio procesado no tiene datos válidos suficientes.")

    # Veredicto final del archivo: si CUALQUIER bloque da FAKE o INCONSISTENTE, el archivo completo se marca como sospechoso
    veredictos = [b["veredicto"] for b in bloques_resultado]
    if any(v == "FAKE" for v in veredictos):
        prediccion_final = "FAKE (al menos un bloque del audio se detectó como sintético)"
    elif any(v == "INCONSISTENTE / MIXTO" for v in veredictos):
        prediccion_final = "INCONSISTENTE / MIXTO (al menos un bloque mezcla señales reales y sintéticas)"
    else:
        prediccion_final = "REAL (todos los bloques evaluados como voz humana)"

    algun_bloque_baja_confianza = any(b["duracion_insuficiente"] for b in bloques_resultado)
    if algun_bloque_baja_confianza:
        prediccion_final += " [algún bloque tuvo pocas ventanas -- confianza reducida en ese tramo]"

    return {
        "archivo": os.path.basename(audio_path),
        "duracion_seg": round(duracion_seg, 2),
        "n_bloques": len(bloques_resultado),
        "bloques": bloques_resultado,
        "prediccion": prediccion_final,
    }


# 6. INICIALIZAR Y EJECUTAR
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Uso: python clasificador.py <ruta_audio>"}))
        sys.exit(1)
    AUDIO_A_PROBAR = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de ejecución: {device}", file=sys.stderr)

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not os.path.exists(AUDIO_A_PROBAR):
        print(json.dumps({"error": f"No se encuentra el archivo de audio '{AUDIO_A_PROBAR}'"}))
        sys.exit(1)

    _ext = os.path.splitext(AUDIO_A_PROBAR)[1].lower()
    if _ext not in FORMATOS_PERMITIDOS:
        print(json.dumps({
            "error": (
                f"Formato '{_ext or 'sin extensión'}' no soportado. "
                f"Formatos permitidos: {', '.join(sorted(FORMATOS_PERMITIDOS))}."
            )
        }))
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": f"No se encontró el modelo en {MODEL_PATH}"}))
        sys.exit(1)

    try:
        model = CNNLSTMAttention(n_mels=N_MELS).to(device)
        mel_transform = build_mel_transform(SR, N_FFT, HOP_LENGTH, N_MELS, device)

        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[INFO] Pesos cargados correctamente.", file=sys.stderr)
    except Exception as e:
        print(json.dumps({"error": f"Fallo cargando el modelo: {e}"}))
        sys.exit(1)

    print(f"[INFO] Analizando archivo: '{AUDIO_A_PROBAR}'...", file=sys.stderr)

    try:
        resultado = predecir_audio_robusto(AUDIO_A_PROBAR, model, device, mel_transform)
    except Exception as e:
        print(json.dumps({"error": f"Fallo durante la inferencia: {e}"}))
        sys.exit(1)

    if "error" in resultado:
        print(json.dumps({"error": resultado["error"]}))
        sys.exit(1)

    print("\n" + "=" * 65, file=sys.stderr)
    print("          RESULTADO DE INFERENCIA MULTI-CRITERIO ROBUSTO        ", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    print(f" ARCHIVO             : {resultado['archivo']}", file=sys.stderr)
    print(f" DURACIÓN            : {resultado['duracion_seg']}s", file=sys.stderr)
    print(f" BLOQUES EVALUADOS   : {resultado['n_bloques']}", file=sys.stderr)
    for b in resultado["bloques"]:
        print(f"   Bloque {b['bloque_idx']+1} [{b['inicio_seg']}s-{b['fin_seg']}s]: "
              f"{b['veredicto']} ({b['confianza']}%) "
              f"-- {b['ventanas_utiles']}/{b['ventanas_totales']} ventanas"
              + ("  [confianza reducida]" if b["duracion_insuficiente"] else ""), file=sys.stderr)
    print(f" PREDICCIÓN FINAL    : {resultado['prediccion']}", file=sys.stderr)
    print("=" * 65 + "\n", file=sys.stderr)

    # Salida JSON para Node (servicio_audio.js hace JSON.parse(stdout))
    # Promedio ponderado por ventanas útiles de cada bloque (más peso a bloques con más ventanas evaluadas)
    total_ventanas = sum(b["ventanas_utiles"] for b in resultado["bloques"])
    if total_ventanas > 0:
        prob_fake = sum(b["score_promedio"] * b["ventanas_utiles"] for b in resultado["bloques"]) / total_ventanas
    else:
        prob_fake = float(np.mean([b["score_promedio"] for b in resultado["bloques"]]))

    veredictos = [b["veredicto"] for b in resultado["bloques"]]
    if any(v == "FAKE" for v in veredictos):
        decision = "DEEPFAKE"
    elif any(v == "INCONSISTENTE / MIXTO" for v in veredictos):
        decision = "SOSPECHOSO"
    else:
        decision = "REAL"

    salida = {
        "archivo": resultado["archivo"],
        "decision": decision,
        "prob_fake": round(float(prob_fake), 4),
        "prob_real": round(1.0 - float(prob_fake), 4),
        "duracion": resultado["duracion_seg"],
        "n_bloques": resultado["n_bloques"],
        "prediccion_detalle": resultado["prediccion"],
        "bloques": resultado["bloques"],
    }

    print(json.dumps(salida, ensure_ascii=False))