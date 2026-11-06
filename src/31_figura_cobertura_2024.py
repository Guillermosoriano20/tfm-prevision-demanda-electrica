"""Figura de la cobertura por franja horaria y por mes en 2024 (seccion 5.5 de la memoria).

Solo lee los CSV que dejo el script 29 (data/resultados/final/) y no calcula nada
nuevo: dibuja las columnas cob_B1..cob_B4 y cob_mes01..cob_mes12 del intervalo propio
y del intervalo de ACI (gamma = 0,001). Sustituye a cobertura_bloque_mes.png, cuyo
titulo hablaba de "bloque de anticipacion": con un origen diario, el bloque es una
franja horaria del dia objetivo (C41), no un efecto de la anticipacion.
Uso (desde la raiz del repositorio): python src/31_figura_cobertura_2024.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FINAL = RAIZ / "data" / "resultados" / "final"
SALIDA = RAIZ / "memoria" / "figuras" / "calibracion_2024_bloque_mes.png"

MODELOS = ["ingenuo", "sarimax", "lightgbm", "nhits"]
ETIQUETA = {"ingenuo": "Ingenuo", "sarimax": "SARIMAX", "lightgbm": "LightGBM", "nhits": "N-HiTS"}
COLOR = {"ingenuo": "#4c78a8", "sarimax": "#f58518", "lightgbm": "#54a24b", "nhits": "#b279a2"}
FRANJAS = ["B1\n00-05 h", "B2\n06-11 h", "B3\n12-17 h", "B4\n18-23 h"]
MESES = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

propio = pd.read_csv(FINAL / "propios_metricas.csv", index_col=0)
aci = pd.read_csv(FINAL / "aci_metricas.csv", index_col=0)

fig, ejes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 2.2]})
for m in MODELOS:
    for tabla, estilo, marca, nombre in [(propio, "-", "o", "propio"), (aci, "--", "s", "ACI")]:
        franja = [100 * tabla.loc[m, f"cob_B{i}"] for i in range(1, 5)]
        mes = [100 * tabla.loc[m, f"cob_mes{i:02d}"] for i in range(1, 13)]
        etiqueta = f"{ETIQUETA[m]}, {nombre}"
        ejes[0].plot(range(4), franja, estilo, marker=marca, ms=4, color=COLOR[m], label=etiqueta)
        ejes[1].plot(range(12), mes, estilo, marker=marca, ms=4, color=COLOR[m], label=etiqueta)
for eje in ejes:
    eje.axhline(90, color="gray", lw=1, ls=":")
    eje.axhspan(88, 92, color="gray", alpha=0.12, lw=0)
    eje.set_ylabel("cobertura del intervalo del 90 % (%)")
    eje.grid(alpha=0.3)
ejes[0].set_xticks(range(4), FRANJAS)
ejes[0].set_xlabel("franja horaria del día objetivo (horas locales)")
ejes[0].set_title("Por franja horaria")
ejes[1].set_xticks(range(12), MESES)
ejes[1].set_xlabel("mes de 2024")
ejes[1].set_title("Por mes")
ejes[1].legend(fontsize=7, ncol=2, loc="lower right")
fig.tight_layout()
SALIDA.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(SALIDA, dpi=150)
print("escrita", SALIDA.relative_to(RAIZ))
