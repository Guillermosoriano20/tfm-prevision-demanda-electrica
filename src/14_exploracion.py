"""OE2: análisis exploratorio de la demanda (solo 2017-2022)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las figuras en archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import MSTL

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "processed" / "dataset.parquet"
FIGURAS = RAIZ / "figuras" / "exploracion"
FIGURAS.mkdir(parents=True, exist_ok=True)

HUELLA_OFICIAL = 5794786326140387865
FIN_EXPLORACION = pd.Timestamp("2023-01-01")  # 2023 validación, 2024 prueba: no se miran

datos = pd.read_parquet(DATASET)
assert pd.util.hash_pandas_object(datos, index=False).sum() == HUELLA_OFICIAL

explo = datos[datos["fecha_local"] < FIN_EXPLORACION].copy()
del datos  # a partir de aquí, 2023 y 2024 no existen en este script

print("Horas en exploración:", len(explo))
print("Desde", explo["hora_local"].min(), "hasta", explo["hora_local"].max())
print("Horas COVID dentro:", int(explo["excluir_covid"].sum()))

# --- 1b. Periodograma ---
y = explo["demanda_mw"].to_numpy()
frec, potencia = periodogram(y - y.mean(), fs=1.0)  # fs=1: frecuencia en ciclos por hora
frec, potencia = frec[1:], potencia[1:]             # se descarta la frecuencia 0
periodo = 1 / frec                                   # periodo en horas

picos = pd.Series(potencia, index=periodo).nlargest(8)
print("Periodos con más potencia (h):", [round(p, 1) for p in picos.index])

fig, ax = plt.subplots(figsize=(9, 4))
ax.loglog(periodo, potencia, linewidth=0.5)
for p in (24, 168, 8766):
    ax.axvline(p, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("Periodo (horas)")
ax.set_ylabel("Potencia espectral")
ax.set_title("Periodograma de la demanda horaria, 2017-2022")
fig.tight_layout()
fig.savefig(FIGURAS / "periodograma.png", dpi=150)
plt.close(fig)

# --- 1b bis. Peso del ciclo anual frente al semestral ---
pot = pd.Series(potencia, index=periodo)
p_anual = min(pot.index, key=lambda p: abs(p - 8766))
p_semestral = min(pot.index, key=lambda p: abs(p - 4383))
puesto = int((pot > pot[p_anual]).sum()) + 1
print(f"Anual: periodo {p_anual:.1f} h, puesto {puesto} de {len(pot)}")
print(f"Potencia anual / semestral: {pot[p_anual] / pot[p_semestral]:.3f}")

# --- 1c. Descomposición MSTL (24, 168 y 8766 h) ---
serie = explo.set_index("hora_utc")["demanda_mw"]
mstl = MSTL(serie, periods=(24, 168, 8766)).fit()

comp = mstl.seasonal.copy()
comp["tendencia"] = mstl.trend
comp["residuo"] = mstl.resid
print("Desviación típica de cada componente (MW):")
print(comp.std().round(0))

fig = mstl.plot()
fig.set_size_inches(10, 10)
fig.tight_layout()
fig.savefig(FIGURAS / "descomposicion_mstl.png", dpi=150)
plt.close(fig)