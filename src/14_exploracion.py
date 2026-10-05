"""OE2: análisis exploratorio de la demanda (solo 2017-2022)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las figuras en archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

# --- 1c. Descomposición MSTL (24, 168 y 8766 h), guardada en disco ---
PERIODOS = (24, 168, 8766)
CACHE_MSTL = RAIZ / "data" / "interim" / f"mstl_{'_'.join(map(str, PERIODOS))}.parquet"

serie = explo.set_index("hora_utc")["demanda_mw"]
if CACHE_MSTL.exists():
    comp = pd.read_parquet(CACHE_MSTL)
else:
    mstl = MSTL(serie, periods=PERIODOS).fit()
    comp = mstl.seasonal.copy()
    comp["tendencia"] = mstl.trend
    comp["residuo"] = mstl.resid
    comp.to_parquet(CACHE_MSTL)

print("Desviación típica de cada componente (MW):")
print(comp.std().round(0))

paneles = pd.concat([serie.rename("observada"), comp[["tendencia", "seasonal_24", "seasonal_168", "seasonal_8766", "residuo"]]], axis=1)
axes = paneles.plot(subplots=True, figsize=(10, 12), linewidth=0.5, legend=False)
for ax, nombre in zip(axes, paneles.columns):
    ax.set_ylabel(f"{nombre} (MW)")
fig = axes[0].get_figure()
fig.tight_layout()
fig.savefig(FIGURAS / "descomposicion_mstl.png", dpi=150)
plt.close(fig)

# --- 2a. Contrastes de estacionariedad: ADF y KPSS ---
y_serie = explo["demanda_mw"].to_numpy()
adf_est, adf_p, adf_lags, *_ = adfuller(y_serie, maxlag=200, autolag="AIC", result_object=False)
kpss_est, kpss_p, kpss_lags, _ = kpss(y_serie, regression="c", nlags="auto", result_object=False)
print(f"ADF:  estadístico {adf_est:.2f}, p-valor {adf_p:.4f}, retardos {adf_lags}")
print(f"KPSS: estadístico {kpss_est:.2f}, p-valor {kpss_p:.4f}, retardos {kpss_lags}")
adf336 = adfuller(y_serie, maxlag=336, autolag=None, result_object=False)
print(f"ADF (336 retardos fijos): estadístico {adf336[0]:.2f}, p-valor {adf336[1]:.4f}")

# --- 2b. ACF y PACF: demanda original y residuo de MSTL ---
for nombre, s in [("demanda", serie), ("residuo", comp["residuo"])]:
    r = acf(s, nlags=336)
    print(f"ACF {nombre}: k=1 {r[1]:.2f}, k=24 {r[24]:.2f}, k=168 {r[168]:.2f}, k=336 {r[336]:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for fila, (nombre, s) in enumerate([("demanda", serie), ("residuo MSTL", comp["residuo"])]):
    plot_acf(s, lags=400, ax=axes[fila, 0], markersize=2, title=f"ACF {nombre}")
    plot_pacf(s, lags=400, ax=axes[fila, 1], markersize=2, method="ywm", title=f"PACF {nombre}")
for ax in axes.flat:
    for p in (24, 168, 336):
        ax.axvline(p, color="red", linestyle="--", linewidth=0.6)
fig.tight_layout()
fig.savefig(FIGURAS / "acf_pacf.png", dpi=150)
plt.close(fig)