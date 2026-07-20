import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import librosa
import librosa.display
import soundfile as sf
import soxr

os.environ['MPLBACKEND'] = 'Agg'
os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.dirname(__file__), '__numba_cache__')

import json, traceback, tempfile, shutil, warnings, threading
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT, TA_LEFT
from reportlab.platypus.flowables import Flowable

PRIMARY     = colors.HexColor('#1E3A8A')
PRIMARY_L   = colors.HexColor('#2563EB')
ACCENT      = colors.HexColor('#3B82F6')
REAL_COL    = colors.HexColor('#059669')
REAL_BG     = colors.HexColor('#ECFDF5')
REAL_BDR    = colors.HexColor('#6EE7B7')
FAKE_COL    = colors.HexColor('#DC2626')
FAKE_BG     = colors.HexColor('#FEF2F2')
FAKE_BDR    = colors.HexColor('#FCA5A5')
WARN_COL    = colors.HexColor('#D97706')
WARN_BG     = colors.HexColor('#FFFBEB')
WARN_BDR    = colors.HexColor('#FCD34D')
GRAY_100    = colors.HexColor('#F3F4F6')
GRAY_200    = colors.HexColor('#E5E7EB')
GRAY_300    = colors.HexColor('#D1D5DB')
GRAY_500    = colors.HexColor('#6B7280')
GRAY_700    = colors.HexColor('#374151')
GRAY_900    = colors.HexColor('#111827')
WHITE       = colors.white
BLACK       = colors.black

REAL_HEX = '#059669'
FAKE_HEX = '#DC2626'
WARN_HEX = '#D97706'
ATTN_HEX = '#7C3AED'

PAGE_W, PAGE_H = A4
MARGIN   = 1.8 * cm
INNER_W  = PAGE_W - 2 * MARGIN

UMBRAL_DECISION = 0.5
WINDOW_SEC = 5.0
OVERLAP = 0.5
STRIDE_SEC = WINDOW_SEC * (1 - OVERLAP)


def clasificar_veredicto(veredicto):
    
    v = (veredicto or '').upper()
    if 'FAKE' in v:
        return 'FAKE'
    if 'INCONSISTENTE' in v or 'SOSPECHOSO' in v or 'MIXTO' in v:
        return 'INCONSISTENTE'
    if 'REAL' in v:
        return 'REAL'
    return v


def identificar_ventana_atipica(b):
    scores = b.get('scores', [])
    if not scores:
        return None
    cat = clasificar_veredicto(b['veredicto'])
    if cat == 'FAKE':
        idx = max(range(len(scores)), key=lambda i: scores[i])
    elif cat == 'INCONSISTENTE':
        mediana = sorted(scores)[len(scores) // 2]
        idx = max(range(len(scores)), key=lambda i: abs(scores[i] - mediana))
    else:
        return None
    return idx, scores[idx]


def tiempo_ventana_en_audio(b, idx_local):
    return b['inicio_seg'] + idx_local * STRIDE_SEC


def tiempo_atencion_real(b, idx_local):
    picos = b.get('atencion_pico_seg')
    if not picos or idx_local >= len(picos):
        return None
    return tiempo_ventana_en_audio(b, idx_local) + picos[idx_local]


class SectionHeader(Flowable):
    def __init__(self, text, width=INNER_W, color=PRIMARY):
        super().__init__()
        self.text   = text
        self.width  = width
        self.height = 0.85 * cm
        self.color  = color

    def wrap(self, *args):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(GRAY_100)
        c.roundRect(0, 0, self.width, self.height, 3, fill=1, stroke=0)
        c.setStrokeColor(GRAY_200)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self.width, self.height, 3, fill=0, stroke=1)
        c.setFillColor(self.color)
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 9)
        c.setFillColor(PRIMARY)
        c.drawString(14, (self.height - 9) / 2 + 1, self.text.upper())


class ProgressBar(Flowable):
    def __init__(self, label, pct, color, width=INNER_W, height=0.38 * cm):
        super().__init__()
        self.label  = label
        self.pct    = min(max(pct, 0), 100)
        self.color  = color
        self.width  = width
        self.height = height

    def wrap(self, *args):
        return self.width, self.height + 16

    def draw(self):
        c    = self.canv
        bar_w = self.width - 60
        bar_y = 4

        c.setFont('Helvetica', 8)
        c.setFillColor(GRAY_700)
        c.drawString(0, bar_y + self.height + 4, self.label)

        c.setFillColor(GRAY_200)
        c.roundRect(0, bar_y, bar_w, self.height, self.height / 2, fill=1, stroke=0)

        fill_w = max(self.height, bar_w * self.pct / 100)
        c.setFillColor(self.color)
        c.roundRect(0, bar_y, fill_w, self.height, self.height / 2, fill=1, stroke=0)

        c.setFont('Helvetica-Bold', 8.5)
        c.setFillColor(self.color)
        c.drawRightString(self.width, bar_y + 1, f'{self.pct:.1f}%')


class VerdictBox(Flowable):
    def __init__(self, decision, prob, width=INNER_W):
        super().__init__()
        self.decision = decision
        self.prob     = prob
        self.width    = width
        self.height   = 2.8 * cm

    def wrap(self, *args):
        return self.width, self.height

    def draw(self):
        c       = self.canv
        is_fake = self.decision == 'DEEPFAKE'
        is_warn = self.decision == 'SOSPECHOSO'
        col  = FAKE_COL  if is_fake else (WARN_COL if is_warn else REAL_COL)
        bg   = FAKE_BG   if is_fake else (WARN_BG  if is_warn else REAL_BG)
        bdr  = FAKE_BDR  if is_fake else (WARN_BDR if is_warn else REAL_BDR)
        icon = '✕' if is_fake else ('?' if is_warn else '✓')

        c.setFillColor(bg)
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        c.setStrokeColor(bdr)
        c.setLineWidth(1.2)
        c.roundRect(0, 0, self.width, self.height, 6, fill=0, stroke=1)
        c.setFillColor(col)
        c.rect(0, 0, 5, self.height, fill=1, stroke=0)

        cx, cy = 1.5 * cm, self.height / 2
        c.setFillColor(col)
        c.circle(cx, cy, 0.45 * cm, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont('Helvetica-Bold', 14)
        c.drawCentredString(cx, cy - 5, icon)

        c.setFont('Helvetica', 8)
        c.setFillColor(GRAY_500)
        c.drawString(2.8 * cm, self.height - 0.7 * cm, 'AUDIO CLASIFICADO COMO')

        c.setFont('Helvetica-Bold', 24)
        c.setFillColor(col)
        c.drawString(2.8 * cm, self.height / 2 - 8, self.decision)

        c.setFont('Helvetica-Bold', 11)
        c.setFillColor(GRAY_700)
        c.drawRightString(self.width - 0.5 * cm, self.height - 0.8 * cm, f'Confianza: {self.prob:.1f}%')
        c.setFont('Helvetica', 8)
        c.setFillColor(GRAY_500)
        c.drawRightString(self.width - 0.5 * cm, self.height - 1.35 * cm, 'Score del modelo')


class StatBox(Flowable):
    def __init__(self, label, value, color, width=4.0 * cm, height=1.6 * cm):
        super().__init__()
        self.label  = label
        self.value  = value
        self.color  = color
        self.width  = width
        self.height = height

    def wrap(self, *args):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(WHITE)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        c.setStrokeColor(GRAY_200)
        c.setLineWidth(0.8)
        c.roundRect(0, 0, self.width, self.height, 4, fill=0, stroke=1)
        c.setFillColor(self.color)
        c.rect(0, self.height - 3, self.width, 3, fill=1, stroke=0)
        c.setFont('Helvetica-Bold', 13)
        c.setFillColor(self.color)
        c.drawCentredString(self.width / 2, self.height / 2 - 2, str(self.value))
        c.setFont('Helvetica', 7)
        c.setFillColor(GRAY_500)
        c.drawCentredString(self.width / 2, 5, self.label)


def analizar_audio_visual(audio_path, img_out, bloques):
    resultado = {'ok': False, 'duracion': None, 'error': None}

    def _run():
        try:
            warnings.filterwarnings('ignore')

            SR = 16000; N_MELS = 128; N_FFT = 1024; HOP = 256
            TOP_DB_TRIM = 30.0

            try:
                audio_raw, sr_in = sf.read(str(audio_path), dtype="float32", always_2d=False)
            except Exception:
                audio_raw, sr_in = librosa.load(audio_path, sr=None, mono=False)
            if audio_raw.ndim > 1:
                axis = -1 if audio_raw.shape[-1] < audio_raw.shape[0] else 0
                audio_raw = audio_raw.mean(axis=axis).astype(np.float32)
            audio_raw = audio_raw.astype(np.float32)

            if sr_in == SR:
                resampled = audio_raw
            else:
                resampled = soxr.resample(audio_raw, sr_in, SR, quality="VHQ").astype(np.float32)

            y, _ = librosa.effects.trim(resampled, top_db=TOP_DB_TRIM)
            sr = SR
            duracion = float(len(y) / sr)

            mel_db = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP),
                ref=np.max
            )

            BG   = '#FFFFFF'; AX_BG = '#F9FAFB'; GRID = '#E5E7EB'
            TICK = '#6B7280'; LAB  = '#374151'
            color_map = {'FAKE': FAKE_HEX, 'INCONSISTENTE': WARN_HEX, 'REAL': REAL_HEX}

            hay_bloques = bool(bloques)
            n_rows = 3 if hay_bloques else 2
            fig = plt.figure(figsize=(13, 9.2 if hay_bloques else 6.4), facecolor=BG)
            gs  = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.60,
                                     left=0.07, right=0.96, top=0.94, bottom=0.07)

            def ax_style(ax, title):
                ax.set_facecolor(AX_BG)
                ax.set_title(title, color='#111827', fontsize=9.5, fontweight='bold', pad=7)
                ax.tick_params(colors=TICK, labelsize=7.5)
                ax.xaxis.label.set_color(LAB)
                ax.yaxis.label.set_color(LAB)
                for spn in ax.spines.values():
                    spn.set_edgecolor(GRID)
                ax.grid(True, color=GRID, linewidth=0.4, alpha=0.8)

            puntos_atencion = []
            for b in (bloques or []):
                if clasificar_veredicto(b['veredicto']) not in ('FAKE', 'INCONSISTENTE'):
                    continue
                atipica = identificar_ventana_atipica(b)
                if not atipica:
                    continue
                idx_local, _ = atipica
                t_attn = tiempo_atencion_real(b, idx_local)
                if t_attn is None:
                    continue
                picos_peso = b.get('atencion_pico_peso') or []
                picos_ratio = b.get('atencion_ratio_concentracion') or []
                peso = picos_peso[idx_local] if idx_local < len(picos_peso) else None
                ratio = picos_ratio[idx_local] if idx_local < len(picos_ratio) else None
                col = color_map.get(clasificar_veredicto(b['veredicto']), '#9CA3AF')
                puntos_atencion.append((b, t_attn, peso, ratio, col))

            ax0 = fig.add_subplot(gs[0, 0])
            librosa.display.waveshow(y, sr=sr, ax=ax0, color='#2563EB', alpha=0.75)
            titulo0 = 'Forma de Onda — Bloques Evaluados por el Modelo' if hay_bloques else 'Forma de Onda'
            ax_style(ax0, titulo0)
            ax0.set_xlabel('Tiempo (s)', fontsize=8)
            ax0.set_ylabel('Amplitud', fontsize=8)
            ax0.set_xlim(0, duracion)

            for b in (bloques or []):
                col = color_map.get(clasificar_veredicto(b['veredicto']), '#9CA3AF')
                ax0.axvspan(b['inicio_seg'], b['fin_seg'], color=col, alpha=0.16, lw=0)
                ymax = ax0.get_ylim()[1]
                ax0.text((b['inicio_seg'] + b['fin_seg']) / 2, ymax * 0.85,
                          f"B{b['bloque_idx'] + 1}", ha='center', fontsize=7,
                          color=col, fontweight='bold')

            for b, t_attn, peso, ratio, col in puntos_atencion:
                ymax = ax0.get_ylim()[1]
                ymin = ax0.get_ylim()[0]
                halo = mpatches.Ellipse((t_attn, 0), width=duracion * 0.012 + 3,
                         height=(ymax - ymin) * 0.85 + 3,
                         fill=False, edgecolor='white', linewidth=3.4, zorder=5)
                ax0.add_patch(halo)
                circ = mpatches.Ellipse((t_attn, 0), width=duracion * 0.012,
                                         height=(ymax - ymin) * 0.85,
                                         fill=False, edgecolor=ATTN_HEX, linewidth=1.6, zorder=6)
                ax0.add_patch(circ)
                if ratio and ratio >= 8:
                    etiqueta = f'{ratio:.1f}× (foco muy concentrado)'
                elif ratio and ratio >= 3:
                    etiqueta = f'{ratio:.1f}× (mayor peso relativo)'
                else:
                    etiqueta = f'{ratio:.1f}×' if ratio else ''

                ax0.annotate(f'región relevante {etiqueta}', (t_attn, ymax * 0.55),
                              textcoords='offset points', xytext=(0, 6),
                              fontsize=6.3, ha='center', color=ATTN_HEX, fontweight='bold')

            for b in (bloques or []):
                if clasificar_veredicto(b['veredicto']) != 'INCONSISTENTE':
                    continue
                ymax_marca = ax0.get_ylim()[1] * 0.97
                for local_i, s in enumerate(b.get('scores', [])):
                    t_ventana = tiempo_ventana_en_audio(b, local_i)
                    col_marca = FAKE_HEX if s >= UMBRAL_DECISION else REAL_HEX
                    ax0.plot(t_ventana, ymax_marca, marker='v', color=col_marca,
                            markersize=5, zorder=7, markeredgecolor='white', markeredgewidth=0.6)

            ax1 = fig.add_subplot(gs[1, 0])
            i1 = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP,
                                           x_axis='time', y_axis='mel', ax=ax1, cmap='Blues')
            ax_style(ax1, 'Mel-Espectrograma (128 bandas) — Representación de Entrada del Modelo')
            cb1 = fig.colorbar(i1, ax=ax1, format='%+2.0f dB', pad=0.015)
            cb1.ax.tick_params(labelsize=6.5)
            ax1.set_xlim(0, duracion)
            for b in (bloques or []):
                col = color_map.get(clasificar_veredicto(b['veredicto']), '#9CA3AF')
                ax1.axvline(b['inicio_seg'], color=col, lw=0.7, ls='--', alpha=0.7)

            f_max = SR / 2
            for b, t_attn, peso, ratio, col in puntos_atencion:
                circ = mpatches.Ellipse((t_attn, f_max / 2), width=duracion * 0.014,
                                         height=f_max * 0.92, fill=False,
                                         edgecolor=ATTN_HEX, linewidth=1.8, zorder=6)
                ax1.add_patch(circ)

            if hay_bloques:
                ax2 = fig.add_subplot(gs[2, 0])
                idx = 0
                xt, xtl = [], []
                barras_idx = {}
                for b in bloques:
                    scores = b.get('scores', [])
                    for local_i, s in enumerate(scores):
                        col = FAKE_HEX if s >= UMBRAL_DECISION else REAL_HEX
                        ax2.bar(idx, s, color=col, width=0.85)
                        barras_idx[(b['bloque_idx'], local_i)] = idx
                        idx += 1
                    if scores:
                        xt.append(idx - len(scores) / 2 - 0.5)
                        xtl.append(f"B{b['bloque_idx'] + 1}")
                    if b is not bloques[-1]:
                        ax2.axvline(idx - 0.5, color=GRID, lw=0.9, ls='-')

                for b in bloques:
                    if clasificar_veredicto(b['veredicto']) not in ('FAKE', 'INCONSISTENTE'):
                        continue
                    atipica = identificar_ventana_atipica(b)
                    if not atipica:
                        continue
                    idx_local, score_atip = atipica
                    x_pos = barras_idx.get((b['bloque_idx'], idx_local))
                    if x_pos is None:
                        continue
                    col = color_map.get(clasificar_veredicto(b['veredicto']), '#9CA3AF')
                    circ = mpatches.Ellipse((x_pos, min(score_atip, 0.9)), width=0.9,
                                             height=0.16, fill=False, edgecolor=ATTN_HEX,
                                             linewidth=1.8, zorder=6)
                    ax2.add_patch(circ)

                ax2.axhline(UMBRAL_DECISION, color='#111827', lw=1, ls='--', alpha=0.6)
                ax2.set_ylim(0, 1)
                ax2.set_xticks(xt)
                ax2.set_xticklabels(xtl, fontsize=7.5)
                ax_style(ax2, f'Score del Modelo por Ventana (5s, 50% de solape) — Umbral de Decisión = {UMBRAL_DECISION}')
                ax2.set_ylabel('Score (0 = REAL, 1 = FAKE)', fontsize=8)
                ax2.set_xlabel('Bloque de origen de cada ventana', fontsize=8)

            fig.savefig(img_out, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
            plt.close(fig)

            resultado['ok'] = True
            resultado['duracion'] = round(duracion, 2)

        except Exception as e:
            resultado['error'] = f"{e}\n{traceback.format_exc()}"

    hilo = threading.Thread(target=_run, daemon=True)
    hilo.start()
    hilo.join(timeout=180)

    if hilo.is_alive():
        resultado['error'] = 'Timeout: analisis tardo mas de 180 segundos'

    return resultado


def fmt_tiempo(seg):
    m = int(seg // 60)
    s = int(seg % 60)
    return f"{m:02d}:{s:02d}"


def formatear_veredicto_display(veredicto):
    
    return (veredicto or '').split('/')[0].strip()


def resumen_bloques(bloques):
    filas = []
    for b in bloques:
        veredicto = b['veredicto']
        cat = clasificar_veredicto(veredicto)
        col = FAKE_COL if cat == 'FAKE' else (WARN_COL if cat == 'INCONSISTENTE' else REAL_COL)
        nota = ' ⚠' if b.get('duracion_insuficiente') else ''
        scores = b.get('scores', [])
        # Conteo real de ventanas por lado del umbral dentro del bloque
        # Se calcula igual para las 3 categorías de veredicto (FAKE, REAL O INCONSISTENTE)
        n_fake = sum(1 for s in scores if s >= UMBRAL_DECISION)
        n_real = len(scores) - n_fake
        filas.append({
            'bloque':     f"Bloque {b['bloque_idx'] + 1}{nota}",
            'rango':      f"{fmt_tiempo(b['inicio_seg'])}–{fmt_tiempo(b['fin_seg'])}",
            'ventanas':   f"{b['ventanas_utiles']}/{b['ventanas_totales']}",
            'vent_real':  str(n_real),
            'vent_fake':  str(n_fake),
            'score_prom': f"{b['score_promedio']:.4f}",
            'desv_std':   f"{b['std_desviacion']:.4f}",
            'veredicto':  formatear_veredicto_display(veredicto),
            'confianza':  f"{b['confianza']:.1f}%",
            'color':      col,
        })
    return filas


def generar_explicacion(decision, prob_real, prob_fake, bloques):
    if not bloques:
        if decision == 'DEEPFAKE':
            return (
                f"El modelo GlowVox CNN-LSTM-Attention clasificó este audio como DEEPFAKE con una "
                f"confianza del {round(prob_fake * 100, 1)}%. Este reporte no cuenta con el detalle "
                f"por ventana del análisis original (por ejemplo, por tratarse de un reporte "
                f"regenerado posteriormente), por lo que solo se muestra el resultado agregado del modelo."
            )
        elif decision == 'REAL':
            return (
                f"El modelo GlowVox CNN-LSTM-Attention clasificó este audio como REAL con una "
                f"confianza del {round(prob_real * 100, 1)}%. Este reporte no cuenta con el detalle "
                f"por ventana del análisis original, por lo que solo se muestra el resultado agregado del modelo."
            )
        else:
            return (
                f"El modelo no fue consistente entre los distintos tramos evaluados del audio "
                f"(REAL: {round(prob_real * 100, 1)}% | DEEPFAKE: {round(prob_fake * 100, 1)}%). "
                f"Este reporte no cuenta con el detalle por ventana del análisis original."
            )

    total_ventanas = sum(len(b['scores']) for b in bloques)
    votos_fake = sum(1 for b in bloques for s in b['scores'] if s >= UMBRAL_DECISION)
    votos_real = total_ventanas - votos_fake
    n_bloques = len(bloques)

    bloques_fake    = [b for b in bloques if clasificar_veredicto(b['veredicto']) == 'FAKE']
    bloques_mixtos  = [b for b in bloques if clasificar_veredicto(b['veredicto']) == 'INCONSISTENTE']
    bloques_real    = [b for b in bloques if clasificar_veredicto(b['veredicto']) == 'REAL']
    bloques_baja_conf = [b for b in bloques if b.get('duracion_insuficiente')]

    def texto_atencion(b, idx_local):
        t_attn = tiempo_atencion_real(b, idx_local)
        if t_attn is None:
            return ''
        picos_peso = b.get('atencion_pico_peso') or []
        picos_ratio = b.get('atencion_ratio_concentracion') or []
        peso = picos_peso[idx_local] if idx_local < len(picos_peso) else None
        ratio = picos_ratio[idx_local] if idx_local < len(picos_ratio) else None
        frase = (
            f" Dentro de esa ventana, la capa de atención del modelo concentró la mayor parte de "
            f"su peso en el instante {fmt_tiempo(t_attn)} (marcado con un círculo violeta en la forma "
            f"de onda y en el mel-espectrograma)"
        )
        if ratio:
            frase += f", con un peso hasta {ratio:.1f} veces mayor que si hubiera atendido cada frame por igual"
        frase += "."
        return frase

    partes = []

    if decision == 'DEEPFAKE':
        partes.append(
            f"El modelo GlowVox CNN-LSTM-Attention clasificó este audio como DEEPFAKE. Sobre un total "
            f"de {total_ventanas} ventanas de 5 segundos (50% de solape) analizadas en {n_bloques} "
            f"bloque(s), {votos_fake} ventanas ({votos_fake / total_ventanas * 100:.1f}%) superaron el "
            f"umbral de decisión ({UMBRAL_DECISION}) frente a {votos_real} clasificadas como voz real."
        )
        for b in bloques_fake:
            vf = sum(1 for s in b['scores'] if s >= UMBRAL_DECISION)
            texto = (
                f"El bloque {b['bloque_idx'] + 1} ({fmt_tiempo(b['inicio_seg'])}–{fmt_tiempo(b['fin_seg'])}) "
                f"concentró la evidencia más fuerte de síntesis: {vf}/{len(b['scores'])} ventanas por "
                f"encima del umbral, con un score promedio de {b['score_promedio']:.3f} y una confianza "
                f"de veredicto de bloque del {b['confianza']:.1f}%."
            )
            atipica = identificar_ventana_atipica(b)
            if atipica:
                idx_local, score_atip = atipica
                texto += (
                    f" La ventana individual más determinante fue la que se ubica alrededor del "
                    f"segundo {fmt_tiempo(tiempo_ventana_en_audio(b, idx_local))}, con un score de "
                    f"{score_atip:.3f} — la más cercana a 1.0 de todo el bloque, es decir, la que el "
                    f"modelo interpretó con mayor seguridad como voz sintética."
                )
                texto += texto_atencion(b, idx_local)
            partes.append(texto)
        if bloques_mixtos:
            for b in bloques_mixtos:
                texto = (
                    f"El bloque {b['bloque_idx'] + 1} ({fmt_tiempo(b['inicio_seg'])}–{fmt_tiempo(b['fin_seg'])}) "
                    f"presentó además alta variabilidad entre sus ventanas (desviación estándar de "
                    f"{b['std_desviacion']:.3f}), lo que refleja una mezcla de tramos reales y sintéticos "
                    f"dentro del mismo bloque."
                )
                atipica = identificar_ventana_atipica(b)
                if atipica:
                    idx_local, score_atip = atipica
                    texto += (
                        f" La ventana alrededor del segundo {fmt_tiempo(tiempo_ventana_en_audio(b, idx_local))} "
                        f"(score de {score_atip:.3f}) fue la que más se apartó del comportamiento del resto "
                        f"de ventanas de ese bloque, y por tanto la que más aportó a la señal de inconsistencia."
                    )
                    texto += texto_atencion(b, idx_local)
                partes.append(texto)
        if bloques_real:
            partes.append(
                f"Los {len(bloques_real)} bloque(s) restante(s) no mostraron evidencia de síntesis por sí "
                f"solos (score promedio por debajo de {UMBRAL_DECISION} de forma consistente); sin embargo, "
                f"el criterio del sistema es conservador: basta con que un solo bloque resulte sintético "
                f"para que el archivo completo se marque como comprometido, ya que un empalme de audio real "
                f"con un fragmento generado sigue siendo, en su conjunto, un audio manipulado."
            )

    elif decision == 'REAL':
        score_prom_global = sum(b['score_promedio'] * len(b['scores']) for b in bloques) / total_ventanas
        bloque_max = max(bloques, key=lambda b: b['score_promedio'])
        partes.append(
            f"El modelo GlowVox CNN-LSTM-Attention clasificó este audio como REAL. Las {total_ventanas} "
            f"ventanas evaluadas en {n_bloques} bloque(s) mostraron un score promedio ponderado de "
            f"{score_prom_global:.3f} (donde 0 = real y 1 = sintético), de forma consistente por debajo "
            f"del umbral de decisión ({UMBRAL_DECISION}). {votos_fake} de {total_ventanas} ventanas "
            f"({votos_fake / total_ventanas * 100:.1f}%) superaron el umbral de forma aislada, sin "
            f"concentrarse en ningún bloque específico, precisamente por eso ningún bloque llegó a "
            f"marcarse como FAKE o INCONSISTENTE."
        )
        if bloque_max['score_promedio'] > 0.3:
            partes.append(
                f"El bloque más cercano al umbral fue el {bloque_max['bloque_idx'] + 1} "
                f"({fmt_tiempo(bloque_max['inicio_seg'])}–{fmt_tiempo(bloque_max['fin_seg'])}), con un "
                f"score promedio de {bloque_max['score_promedio']:.3f} y una desviación estándar de "
                f"{bloque_max['std_desviacion']:.3f} entre sus ventanas — aun así, ninguna ventana "
                f"individual llegó a superar el umbral de {UMBRAL_DECISION} de forma sostenida dentro de "
                f"ese bloque, por lo que se mantuvo clasificado como REAL."
            )
        else:
            partes.append(
                "Ningún bloque se acercó al umbral de decisión: los scores se mantuvieron bajos de forma "
                "homogénea a lo largo de todo el audio, sin picos aislados que sugieran algún tramo "
                "sintético puntual."
            )

    else:
        partes.append(
            f"El modelo no fue consistente entre los distintos tramos del audio (REAL global: "
            f"{round(prob_real * 100, 1)}% | DEEPFAKE global: {round(prob_fake * 100, 1)}%). Esto ocurre "
            f"cuando al menos un bloque presenta alta variabilidad entre sus ventanas sin que ninguna "
            f"ventana domine con claridad, lo que el sistema interpreta como evidencia de una posible "
            f"mezcla de segmentos reales y sintéticos dentro del mismo audio, en lugar de forzar una "
            f"clasificación binaria poco confiable."
        )
        bloques_relevantes = bloques_mixtos if bloques_mixtos else bloques_fake
        for b in bloques_relevantes:
            vf = sum(1 for s in b['scores'] if s >= UMBRAL_DECISION)
            texto = (
                f"El bloque {b['bloque_idx'] + 1} ({fmt_tiempo(b['inicio_seg'])}–{fmt_tiempo(b['fin_seg'])}) "
                f"fue el que generó la ambigüedad: {vf}/{len(b['scores'])} ventanas marcadas como "
                f"sintéticas y una desviación estándar de {b['std_desviacion']:.3f} entre sus ventanas "
                f"(por encima del umbral de variabilidad de 0.15 que el sistema usa para distinguir "
                f"'inconsistente' de un veredicto homogéneo)."
            )
            atipica = identificar_ventana_atipica(b)
            if atipica:
                idx_local, score_atip = atipica
                texto += (
                    f" La ventana alrededor del segundo {fmt_tiempo(tiempo_ventana_en_audio(b, idx_local))}, "
                    f"con score {score_atip:.3f}, fue la que más se apartó del resto — el punto exacto donde "
                    f"conviene enfocar una revisión manual si se requiere confirmar el resultado."
                )
                texto += texto_atencion(b, idx_local)
            partes.append(texto)

    if bloques_baja_conf:
        idxs = ', '.join(str(b['bloque_idx'] + 1) for b in bloques_baja_conf)
        partes.append(
            f"Los bloques {idxs} tuvieron menos de 2 ventanas evaluables (posiblemente por poca energía o "
            f"corta duración del tramo), por lo que su veredicto individual se considera de confianza reducida."
        )

    return '\n\n'.join(partes)


def xml_escape(s):
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def ps(**kw):
    return ParagraphStyle('_', **kw)

def hr():
    return HRFlowable(width='100%', thickness=0.5, color=GRAY_200, spaceAfter=4, spaceBefore=4)

def sp(n=6):
    return Spacer(1, n)

def tbl_style_base():
    return [
        ('FONTNAME',    (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 8.5),
        ('TEXTCOLOR',   (0,0), (-1,-1), GRAY_700),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',  (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING',(0,0), (-1,-1), 10),
        ('GRID',        (0,0), (-1,-1), 0.4, GRAY_200),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[WHITE, GRAY_100]),
    ]


def make_on_page(analisis_id, fecha, nombre):
    def on_page(canvas, doc):
        canvas.saveState()

        canvas.setStrokeColor(GRAY_300)
        canvas.setLineWidth(0.8)
        canvas.rect(0.8*cm, 0.8*cm, PAGE_W - 1.6*cm, PAGE_H - 1.6*cm, fill=0, stroke=1)

        canvas.setFillColor(PRIMARY)
        canvas.rect(0.8*cm, PAGE_H - 2.4*cm, PAGE_W - 1.6*cm, 1.6*cm, fill=1, stroke=0)

        canvas.setFont('Helvetica-Bold', 15)
        canvas.setFillColor(WHITE)
        canvas.drawString(1.4*cm, PAGE_H - 1.75*cm, 'GlowVox')
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(colors.HexColor('#93C5FD'))
        canvas.drawString(1.4*cm, PAGE_H - 2.1*cm, 'Audio Deepfake Detection Platform')

        canvas.setFont('Helvetica-Bold', 9)
        canvas.setFillColor(WHITE)
        canvas.drawRightString(PAGE_W - 1.4*cm, PAGE_H - 1.7*cm,
                               f'Reporte #{xml_escape(str(analisis_id))}')
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(colors.HexColor('#93C5FD'))
        canvas.drawRightString(PAGE_W - 1.4*cm, PAGE_H - 2.1*cm,
                               xml_escape(fecha))

        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(0.8*cm, PAGE_H - 2.4*cm, PAGE_W - 0.8*cm, PAGE_H - 2.4*cm)

        canvas.setFillColor(GRAY_100)
        canvas.rect(0.8*cm, 0.8*cm, PAGE_W - 1.6*cm, 0.85*cm, fill=1, stroke=0)

        canvas.setStrokeColor(GRAY_300)
        canvas.setLineWidth(0.4)
        canvas.line(0.8*cm, 1.65*cm, PAGE_W - 0.8*cm, 1.65*cm)

        canvas.setFont('Helvetica', 6.5)
        canvas.setFillColor(GRAY_500)
        canvas.drawString(1.4*cm, 1.1*cm,
            'GlowVox — Reporte generado automáticamente. '
            'Los resultados son orientativos y no constituyen prueba legal.')
        canvas.setFont('Helvetica-Bold', 7)
        canvas.setFillColor(PRIMARY)
        canvas.drawRightString(PAGE_W - 1.4*cm, 1.1*cm, f'Pág. {doc.page}')

        canvas.restoreState()

    return on_page


def generar_pdf(data, output_path):
    tmp_dir = tempfile.mkdtemp()
    img_out = os.path.join(tmp_dir, 'analisis_ventanas.png')

    audio_path  = data['audio_path']
    nombre      = data['nombre_audio']
    decision    = data['decision']
    prob_real   = float(data['prob_real'])
    prob_fake   = float(data['prob_fake'])
    fecha       = data.get('fecha', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    usuario     = data.get('usuario', 'Usuario')
    analisis_id = data.get('analisis_id', 'N/A')
    bloques     = data.get('bloques') or []

    visual = analizar_audio_visual(audio_path, img_out, bloques)
    tiene_img = os.path.isfile(img_out) and os.path.getsize(img_out) > 1000

    if visual.get('error'):
        print(f'[WARN] Error generando visualización: {visual["error"][:300]}', file=sys.stderr)

    duracion = data.get('duracion') or visual.get('duracion') or 0

    is_fake   = decision == 'DEEPFAKE'
    is_real   = decision == 'REAL'
    DEC_COL   = FAKE_COL if is_fake else (REAL_COL if is_real else WARN_COL)
    pct_real  = round(prob_real * 100, 1)
    pct_fake  = round(prob_fake * 100, 1)

    total_ventanas  = sum(len(b['scores']) for b in bloques) if bloques else 0
    votos_fake      = sum(1 for b in bloques for s in b['scores'] if s >= UMBRAL_DECISION) if bloques else 0
    votos_real      = total_ventanas - votos_fake
    n_bloques       = len(bloques)
    bloques_fake    = [b for b in bloques if clasificar_veredicto(b['veredicto']) == 'FAKE']
    bloques_mixtos  = [b for b in bloques if clasificar_veredicto(b['veredicto']) == 'INCONSISTENTE']

    on_page = make_on_page(analisis_id, fecha, nombre)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=3.0 * cm,
        bottomMargin=2.2 * cm,
    )

    story = []

    story.append(Paragraph(
        'REPORTE DE ANÁLISIS DE AUDIO',
        ps(fontName='Helvetica-Bold', fontSize=16, textColor=PRIMARY,
           alignment=TA_CENTER, leading=20)
    ))
    story.append(Paragraph(
        'Detección de Deepfakes mediante Inteligencia Artificial',
        ps(fontName='Helvetica', fontSize=9, textColor=GRAY_500,
           alignment=TA_CENTER, leading=14)
    ))
    story.append(sp(8))
    story.append(Paragraph(
        'Este reporte está diseñado para apoyar la interpretación del resultado entregado por el '
        'modelo: junto con el veredicto y su nivel de confianza, incorpora representaciones visuales '
        'de las regiones específicas del audio que el modelo identificó como más relevantes al momento '
        'de tomar su decisión.',
        ps(fontName='Helvetica', fontSize=8.5, textColor=GRAY_700, alignment=TA_JUSTIFY, leading=13)
    ))
    story.append(sp(10))
    story.append(hr())
    story.append(sp(10))

    story.append(SectionHeader('Resultado del Análisis'))
    story.append(sp(8))
    story.append(VerdictBox(decision, pct_fake if is_fake else pct_real))
    story.append(sp(10))

    story.append(ProgressBar('Probabilidad de voz REAL',  pct_real, REAL_COL, width=INNER_W))
    story.append(sp(6))
    story.append(ProgressBar('Probabilidad de DEEPFAKE',  pct_fake, FAKE_COL, width=INNER_W))
    story.append(sp(14))

    box_w = (INNER_W - 3 * 0.3 * cm) / 4
    if bloques:
        stats_row = Table(
            [[
                StatBox('Duración',           f"{duracion}s",          ACCENT,  width=box_w),
                StatBox('Bloques evaluados',  str(n_bloques),          PRIMARY, width=box_w),
                StatBox('Ventanas REAL',      f"{votos_real}/{total_ventanas}", REAL_COL, width=box_w),
                StatBox('Ventanas FAKE',      f"{votos_fake}/{total_ventanas}", FAKE_COL, width=box_w),
            ]],
            colWidths=[box_w, box_w, box_w, box_w],
            hAlign='LEFT',
        )
    else:
        stats_row = Table(
            [[
                StatBox('Duración',      f"{duracion}s",        ACCENT,  width=box_w),
                StatBox('Prob. REAL',    f"{pct_real}%",        REAL_COL, width=box_w),
                StatBox('Prob. DEEPFAKE',f"{pct_fake}%",        FAKE_COL, width=box_w),
                StatBox('Modelo',        'CNN-LSTM-Attn',       PRIMARY, width=box_w),
            ]],
            colWidths=[box_w, box_w, box_w, box_w],
            hAlign='LEFT',
        )
    stats_row.setStyle(TableStyle([
        ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
        ('VALIGN',       (0,0),(-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0),(-1,-1), 0),
        ('RIGHTPADDING', (0,0),(-1,-1), 5),
        ('TOPPADDING',   (0,0),(-1,-1), 0),
        ('BOTTOMPADDING',(0,0),(-1,-1), 0),
    ]))
    story.append(stats_row)
    story.append(sp(14))

    story.append(SectionHeader('Información del Archivo'))
    story.append(sp(8))

    def info_row(lbl, val):
        return [
            Paragraph(lbl, ps(fontName='Helvetica-Bold', fontSize=8,
                               textColor=PRIMARY, leading=13)),
            Paragraph(xml_escape(str(val)), ps(fontName='Helvetica', fontSize=8.5,
                                               textColor=GRAY_700, leading=13)),
        ]

    info_tbl = Table([
        info_row('Archivo analizado',  nombre),
        info_row('Usuario',            usuario),
        info_row('Fecha de análisis',  fecha),
        info_row('Duración del audio', f"{duracion} segundos"),
        info_row('Modelo utilizado',   'GlowVox CNN-LSTM-Attention'),
        info_row('Configuración',      '128 Mel-bands + Deltas temporales (16kHz), ventanas de 5s con 50% de solape'),
    ], colWidths=[INNER_W * 0.28, INNER_W * 0.72])
    info_tbl.setStyle(TableStyle(tbl_style_base() + [
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,0), (0,-1), PRIMARY),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#EFF6FF')),
    ]))
    story.append(info_tbl)
    story.append(sp(14))

    if bloques:
        story.append(SectionHeader('Detalle de Bloques Evaluados por el Modelo'))
        story.append(sp(8))

        bloq_rows = [[
            Paragraph('BLOQUE',        ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('RANGO (mm:ss)', ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('VENTANAS',      ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('VENT. REAL',    ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('VENT. FAKE',    ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('SCORE PROM.',   ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('DESV. EST.',    ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('VEREDICTO',     ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
            Paragraph('CONFIANZA',     ps(fontName='Helvetica-Bold', fontSize=7.5, textColor=WHITE)),
        ]]
        for fila in resumen_bloques(bloques):
            bloq_rows.append([
                Paragraph(fila['bloque'],     ps(fontName='Helvetica-Bold', fontSize=7.8, textColor=GRAY_900)),
                Paragraph(fila['rango'],      ps(fontName='Helvetica',      fontSize=7.8, textColor=GRAY_700)),
                Paragraph(fila['ventanas'],   ps(fontName='Helvetica',      fontSize=7.8, textColor=GRAY_700)),
                Paragraph(fila['vent_real'],  ps(fontName='Helvetica-Bold', fontSize=7.8, textColor=REAL_COL)),
                Paragraph(fila['vent_fake'],  ps(fontName='Helvetica-Bold', fontSize=7.8, textColor=FAKE_COL)),
                Paragraph(fila['score_prom'], ps(fontName='Helvetica',      fontSize=7.8, textColor=PRIMARY_L)),
                Paragraph(fila['desv_std'],   ps(fontName='Helvetica',      fontSize=7.8, textColor=GRAY_700)),
                Paragraph(fila['veredicto'],  ps(fontName='Helvetica-Bold', fontSize=7.8, textColor=fila['color'])),
                Paragraph(fila['confianza'],  ps(fontName='Helvetica-Bold', fontSize=7.8, textColor=fila['color'])),
            ])

        bloq_tbl = Table(bloq_rows, colWidths=[
            INNER_W*0.11, INNER_W*0.14, INNER_W*0.10, INNER_W*0.09, INNER_W*0.09,
            INNER_W*0.13, INNER_W*0.11, INNER_W*0.13, INNER_W*0.10
        ])
        bloq_tbl.setStyle(TableStyle(tbl_style_base() + [
            ('BACKGROUND',    (0,0), (-1,0),  PRIMARY),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, GRAY_100]),
            ('LINEBELOW',     (0,0), (-1,0),  1.5, ACCENT),
            ('FONTSIZE',      (0,0), (-1,-1), 7.8),
            ('ALIGN',         (2,0), (4,-1),  'CENTER'),
            ('ALIGN',         (8,0), (8,-1),  'CENTER'),
            ('LEFTPADDING',   (0,0), (-1,-1), 5),
            ('RIGHTPADDING',  (0,0), (-1,-1), 5),
        ]))
        story.append(bloq_tbl)
        story.append(sp(14))

    story.append(SectionHeader('Metodología de Evaluación del Modelo'))
    story.append(sp(8))
    metodologia = (
        f"El modelo GlowVox divide el audio en bloques de hasta 52.5 segundos, cada uno compuesto "
        f"por ventanas de 5 segundos con 50% de solape (hasta 20 ventanas por bloque). Cada ventana "
        f"se transforma en un mel-espectrograma de 128 bandas (más sus derivadas de primer y segundo "
        f"orden) y se evalúa de forma independiente con la red CNN-LSTM con atención, que produce un "
        f"score entre 0 (voz real) y 1 (voz sintética) para esa ventana.\n\n"
        f"Un bloque se marca FAKE si la mayoría de sus ventanas superan el umbral de decisión "
        f"({UMBRAL_DECISION}); se marca REAL si la mayoría no lo superan; y se marca INCONSISTENTE"
        f"si, aun sin ese predominio claro, la desviación estándar entre las ventanas del bloque "
        f"es alta (≥ 0.15) — lo que indica que dentro de ese bloque hay ventanas con comportamiento "
        f"marcadamente distinto entre sí, en lugar de un patrón homogéneo.\n\n"
        f"Dentro de cada ventana, antes de emitir el score, el modelo pasa la secuencia temporal por "
        f"una capa de atención (Attention) que aprende a ponderar más unos frames que otros al construir "
        f"el contexto final usado para clasificar. Ese peso de atención es real y se registra por "
        f"ventana: cuando una ventana resulta decisiva para el veredicto de un bloque, el reporte marca "
        f"con un círculo violeta el instante exacto donde ese peso se concentró — no es una estimación "
        f"visual, es el mismo frame que el modelo más ponderó internamente.\n\n"
        f"El veredicto final del archivo completo sigue un criterio conservador: si CUALQUIER bloque "
        f"resulta FAKE, todo el archivo se marca como DEEPFAKE, incluso si el resto del audio es real "
        f"— porque un solo tramo sintético compromete la autenticidad del archivo completo. Si ningún "
        f"bloque es FAKE pero al menos uno es INCONSISTENTE, el archivo se marca como SOSPECHOSO. Solo "
        f"si todos los bloques son REAL, el archivo se marca como REAL."
    )
    for parrafo in metodologia.split('\n\n'):
        story.append(Paragraph(
            xml_escape(parrafo),
            ps(fontName='Helvetica', fontSize=8.5, textColor=GRAY_700, leading=14, alignment=TA_JUSTIFY)
        ))
        story.append(sp(5))
    story.append(sp(9))

    story.append(SectionHeader('¿Por qué el modelo tomó esta decisión?'))
    story.append(sp(8))

    explicacion  = generar_explicacion(decision, prob_real, prob_fake, bloques)
    parrafos     = explicacion.split('\n\n')
    just_content = []
    for p in parrafos:
        p = p.strip()
        if not p:
            continue
        just_content.append(Paragraph(
            xml_escape(p),
            ps(fontName='Helvetica', fontSize=8.5, textColor=GRAY_700,
               leading=14, alignment=TA_JUSTIFY)
        ))
        just_content.append(sp(5))

    just_tbl = Table([[just_content]], colWidths=[INNER_W])
    just_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), WHITE),
        ('BOX',          (0,0),(-1,-1), 0.5, GRAY_200),
        ('LINEBEFORE',   (0,0),(0,-1),  4,   DEC_COL),
        ('LEFTPADDING',  (0,0),(-1,-1), 14),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 10),
        ('BOTTOMPADDING',(0,0),(-1,-1), 10),
    ]))
    story.append(just_tbl)
    story.append(sp(14))

    story.append(SectionHeader('Análisis Visual por Ventanas'))
    story.append(sp(6))
    if bloques:
        desc_visual = (
            'Las siguientes representaciones muestran únicamente lo que el modelo procesó para '
            'tomar su decisión: la forma de onda con los bloques resaltados según su veredicto, '
            'el mel-espectrograma de 128 bandas (la representación de entrada real del modelo) y '
            'el score individual que el modelo asignó a cada ventana de 5 segundos evaluada. El '
            'círculo violeta señala, en cada bloque con veredicto FAKE o INCONSISTENTE, el instante '
            'exacto donde la capa de atención del modelo concentró más peso dentro de su ventana '
            'más decisiva.'
        )
    else:
        desc_visual = (
            'Representación de la forma de onda y del mel-espectrograma (128 bandas), que es la '
            'representación de entrada real del modelo. No hay datos de bloques/ventanas '
            'disponibles para este reporte.'
        )
    story.append(Paragraph(
        desc_visual,
        ps(fontName='Helvetica', fontSize=8.5, textColor=GRAY_500,
           alignment=TA_JUSTIFY, leading=14)
    ))
    story.append(sp(8))

    if tiene_img:
        img_w = INNER_W
        img_h = img_w * (9.2 / 13) if bloques else img_w * (6.4 / 13)
        img_tbl = Table([[Image(img_out, width=img_w, height=img_h)]], colWidths=[INNER_W])
        img_tbl.setStyle(TableStyle([
            ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
            ('BOX',          (0,0),(-1,-1), 0.5, GRAY_200),
            ('TOPPADDING',   (0,0),(-1,-1), 4),
            ('BOTTOMPADDING',(0,0),(-1,-1), 4),
        ]))
        story.append(img_tbl)
    else:
        msg = 'Visualización no disponible.'
        if visual.get('error'):
            msg += f' (Error: {xml_escape(visual["error"][:200])})'
        story.append(Table([[Paragraph(msg,
            ps(fontSize=8.5, fontName='Helvetica', textColor=WARN_COL, alignment=TA_CENTER))]],
            colWidths=[INNER_W], style=TableStyle([
                ('BOX',          (0,0),(-1,-1), 0.5, WARN_BDR),
                ('BACKGROUND',   (0,0),(-1,-1), WARN_BG),
                ('TOPPADDING',   (0,0),(-1,-1), 18),
                ('BOTTOMPADDING',(0,0),(-1,-1), 18),
            ])))
    story.append(sp(14))

    story.append(SectionHeader('Conclusión y Recomendaciones'))
    story.append(sp(8))

    if is_fake:
        concl_main = (
            f'El análisis concluye con una confianza del {pct_fake}% que el archivo '
            f'"{xml_escape(nombre)}" corresponde a voz sintética generada artificialmente '
            f'mediante técnicas de inteligencia artificial (modelo TTS o clonación de voz). '
        )
        if bloques_fake:
            idxs = ', '.join(str(b['bloque_idx'] + 1) for b in bloques_fake)
            concl_main += (
                f'Esta conclusión se sostiene específicamente en el/los bloque(s) {idxs}, donde la '
                f'mayoría de las ventanas de 5 segundos superaron el umbral de decisión de forma '
                f'sostenida (ver sección "Detalle de Bloques Evaluados por el Modelo" y el círculo violeta '
                f'en la sección "Análisis Visual por Ventanas"). '
            )
        concl_main += (
            'El resto del audio, aunque no muestre evidencia de síntesis por sí solo, no revierte la '
            'clasificación: basta un tramo sintético para comprometer la autenticidad del archivo completo.'
        )
        recs = [
            'No utilizar como evidencia sin verificación independiente por expertos certificados.',
            'Complementar con herramientas forenses especializadas (Adobe Content Authenticity, FotoForensics).',
            'Verificar la cadena de custodia del archivo: metadatos, fecha de creación y software de origen.',
            'Considerar el contexto de obtención antes de emitir conclusiones definitivas.',
        ]
    elif is_real:
        concl_main = (
            f'El análisis concluye con una confianza del {pct_real}% que el archivo '
            f'"{xml_escape(nombre)}" corresponde a una grabación de voz humana auténtica. '
        )
        if bloques:
            concl_main += (
                f'Los {n_bloques} bloque(s) evaluados ({total_ventanas} ventanas en total) se '
                f'mantuvieron consistentemente por debajo del umbral de decisión a lo largo de todo el '
                f'audio, sin picos aislados ni bloques marcados como inconsistentes (ver "Detalle de '
                f'Bloques Evaluados por el Modelo").'
            )
        recs = [
            'Los resultados son orientativos; confirmación por experto es recomendable para usos legales.',
            'Conservar la cadena de custodia del archivo para garantizar su integridad.',
            'Mantener los metadatos originales del archivo como evidencia complementaria.',
        ]
    else:
        concl_main = (
            f'El análisis no fue concluyente (REAL: {pct_real}% | DEEPFAKE: {pct_fake}%). '
        )
        if bloques_mixtos:
            idxs = ', '.join(str(b['bloque_idx'] + 1) for b in bloques_mixtos)
            concl_main += (
                f'El/los bloque(s) {idxs} presentaron alta variabilidad entre sus ventanas sin que '
                f'ninguna clase dominara con claridad (ver círculo violeta en la sección de análisis visual), '
                f'lo que impide una clasificación definitiva. '
            )
        concl_main += 'Se requiere análisis adicional, idealmente con foco en los bloques y ventanas señalados en la parte superior.'
        recs = [
            'Realizar análisis con herramientas forenses especializadas antes de tomar decisiones.',
            'Solicitar una segunda opinión a un experto en análisis forense de audio.',
            'No utilizar este resultado como evidencia en ningún contexto legal o formal.',
        ]

    story.append(Paragraph(concl_main,
        ps(fontName='Helvetica', fontSize=9, textColor=GRAY_700,
           alignment=TA_JUSTIFY, leading=15)))
    story.append(sp(8))

    for i, rec in enumerate(recs, 1):
        story.append(Table([[
            Paragraph(str(i),
                ps(fontName='Helvetica-Bold', fontSize=8, textColor=WHITE, alignment=TA_CENTER)),
            Paragraph(xml_escape(rec),
                ps(fontName='Helvetica', fontSize=8.5, textColor=GRAY_700, leading=13)),
        ]], colWidths=[0.55*cm, INNER_W - 0.55*cm],
        style=TableStyle([
            ('BACKGROUND',   (0,0),(0,-1),  DEC_COL),
            ('BACKGROUND',   (1,0),(1,-1),  WHITE),
            ('BOX',          (0,0),(-1,-1), 0.4, GRAY_200),
            ('LINEBELOW',    (0,0),(-1,-1), 0.3, GRAY_200),
            ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
            ('ALIGN',        (0,0),(0,-1),  'CENTER'),
            ('TOPPADDING',   (0,0),(-1,-1), 7),
            ('BOTTOMPADDING',(0,0),(-1,-1), 7),
            ('LEFTPADDING',  (0,0),(0,-1),  0),
            ('LEFTPADDING',  (1,0),(1,-1),  10),
        ])))
        story.append(sp(3))

    story.append(sp(12))

    aviso_tbl = Table([[Paragraph(
        '⚠ AVISO LEGAL — Este reporte fue generado automáticamente por la plataforma GlowVox. '
        'Los resultados tienen carácter orientativo y NO constituyen prueba legal por sí solos. '
        'GlowVox no se responsabiliza por decisiones tomadas exclusivamente con base en este documento.',
        ps(fontName='Helvetica', fontSize=8, textColor=WARN_COL,
           alignment=TA_JUSTIFY, leading=13)
    )]], colWidths=[INNER_W])
    aviso_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), WARN_BG),
        ('BOX',          (0,0),(-1,-1), 1,   WARN_BDR),
        ('LINEBEFORE',   (0,0),(0,-1),  4,   WARN_COL),
        ('TOPPADDING',   (0,0),(-1,-1), 10),
        ('BOTTOMPADDING',(0,0),(-1,-1), 10),
        ('LEFTPADDING',  (0,0),(-1,-1), 14),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
    ]))
    story.append(aviso_tbl)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': "Uso: script.py '<json>' <output_path>"}))
        sys.exit(1)
    try:
        data        = json.loads(sys.argv[1])
        output_path = sys.argv[2]
        generar_pdf(data, output_path)
        print(json.dumps({'ok': True, 'path': output_path}))
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()