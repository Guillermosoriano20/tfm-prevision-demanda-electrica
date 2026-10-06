"""OE2: análisis exploratorio de la demanda (solo 2017-2022)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # guarda las figuras en archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


# --- 3a. Curva demanda-temperatura (días laborables, sin COVID) ---
diario = explo.groupby("fecha_local").agg(
    demanda=("demanda_mw", "mean"),
    temp=("t_ponderada", "mean"),
    tipo=("tipo_dia", "first"),
    covid=("excluir_covid", "any"),
)
lab = diario[(diario["tipo"] == "laborable") & ~diario["covid"]]
print("Días laborables sin COVID:", len(lab))

curva = lab.groupby(lab["temp"].round())["demanda"].agg(["mean", "count"])
print(curva.round(0).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(lab["temp"], lab["demanda"], s=4, alpha=0.4)
ax.plot(curva.index, curva["mean"], color="red", linewidth=1.5)
ax.set_xlabel("Temperatura media diaria ponderada (°C)")
ax.set_ylabel("Demanda media diaria (MW)")
ax.set_title("Demanda frente a temperatura, laborables 2017-2022 sin COVID")
fig.tight_layout()
fig.savefig(FIGURAS / "demanda_temperatura.png", dpi=150)
plt.close(fig)


# --- 3b. Umbrales de grados-día (por provincia, ponderados después) ---
pesos = pd.read_csv(RAIZ / "referencia" / "pesos_poblacion_2021.csv", dtype={"cod_provincia": str})
cols_t = ["t_" + c for c in pesos["cod_provincia"]]
w = pesos["peso"].to_numpy()
assert set(cols_t) <= set(explo.columns) and abs(w.sum() - 1) < 1e-9

t_prov = explo.groupby("fecha_local")[cols_t].mean().loc[lab.index].to_numpy()  # días × provincias
anios = pd.get_dummies(lab.index.year, dtype=float).to_numpy()  # una columna por año
yd = lab["demanda"].to_numpy()

resultados = []
for tc in np.arange(12, 20.01, 0.5):
    hdd = np.maximum(tc - t_prov, 0) @ w  # grados-día de calefacción, ponderados
    for tr in np.arange(max(tc, 16), 24.01, 0.5):
        cdd = np.maximum(t_prov - tr, 0) @ w  # grados-día de refrigeración, ponderados
        X = np.column_stack([anios, hdd, cdd])
        coef, sse, *_ = np.linalg.lstsq(X, yd, rcond=None)
        resultados.append((tc, tr, sse[0], coef[-2], coef[-1]))

res = pd.DataFrame(resultados, columns=["tc", "tr", "sse", "b_cal", "b_ref"]).sort_values("sse")
res["dif_%"] = 100 * (res["sse"] / res["sse"].min() - 1)
print(res.head(10).drop(columns="sse").round(2).to_string(index=False))


# --- 3c. Ajuste con los umbrales elegidos ---
TC, TR = res.iloc[0]["tc"], res.iloc[0]["tr"]
hdd = np.maximum(TC - t_prov, 0) @ w
cdd = np.maximum(t_prov - TR, 0) @ w
X = np.column_stack([anios, hdd, cdd])
coef, *_ = np.linalg.lstsq(X, yd, rcond=None)
ajuste = X @ coef
r2 = 1 - ((yd - ajuste) ** 2).sum() / ((yd - yd.mean()) ** 2).sum()
print(f"Umbrales: Tc={TC} °C, Tr={TR} °C; R² = {r2:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(lab["temp"], yd, s=4, alpha=0.3, label="observada")
ax.scatter(lab["temp"], ajuste, s=4, alpha=0.6, color="red", label="ajustada (grados-día + año)")
ax.axvline(TC, color="black", linestyle="--", linewidth=0.8)
ax.axvline(TR, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("Temperatura media diaria ponderada (°C)")
ax.set_ylabel("Demanda media diaria (MW)")
ax.set_title(f"Ajuste con grados-día: Tc = {TC} °C, Tr = {TR} °C")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "demanda_temperatura_ajuste.png", dpi=150)
plt.close(fig)


# --- 4a. Fiestas nacionales de fecha fija que caen en domingo (solo calendario) ---
FIJAS = ["01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25"]
fechas = pd.to_datetime([f"{a}-{md}" for a in range(2017, 2025) for md in FIJAS])
domingos = fechas[fechas.dayofweek == 6]
print(f"Fiestas nacionales fijas en domingo ({len(domingos)}):", [f.strftime("%d/%m/%Y") for f in domingos])

frac = explo.groupby("fecha_local")["fraccion_festivo"].first()
print(frac.reindex(domingos[domingos < FIN_EXPLORACION]).to_string())


# --- 4b. Laborables con demanda muy por debajo de la que predice la temperatura ---
resid_lab = pd.Series(yd - ajuste, index=lab.index)
peores = resid_lab.nsmallest(20)
print(pd.DataFrame({"residuo_MW": peores.round(0), "dia": peores.index.day_name()}).to_string())

# --- 4c. Filomena (enero de 2021) y extremos horarios ---
print(diario.loc["2021-01-04":"2021-01-17", ["demanda", "temp", "tipo"]].round(1).to_string())
for nombre, idx in [("máximo", explo["demanda_mw"].idxmax()), ("mínimo", explo["demanda_mw"].idxmin())]:
    fila = explo.loc[idx]
    print(f"{nombre}: {fila['hora_local']}, {fila['demanda_mw']:.0f} MW, {fila['t_ponderada']:.1f} °C, {fila['tipo_dia']}")