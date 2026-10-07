"""SARIMAX: prueba de tiempos con la configuración más pesada prevista.

Criterio fijado antes de medir: docs/protocolo_oe5.md, apartado 8 (commit 76c732b);
método de estimación FGLS iterado, apartado 8.6 (commit 8ec9995).
Solo mide tiempos: no calcula errores ni muestra previsiones de 2023.
"""
import time

import numpy as np

import validacion as v
from exogenas import exogenas_sarimax
from sarimax_fgls import ajustar_fgls, con_constante

datos = v.cargar_datos()                      # huella comprobada; 2024 cortado
X = exogenas_sarimax(datos, k_dia=10, k_semana=10, k_anio=4, k_interaccion=3)
y = datos["demanda_mw"].where(~datos["excluir_covid"])   # COVID como vacío
y.index = X.index = y.index.tz_localize(None)            # statsmodels: índice sin zona
y, X = y.asfreq("h"), X.asfreq("h")
print(f"Columnas exógenas: {X.shape[1]} (más la constante)")

origenes = v.generar_origenes("validacion")
fin_ajuste = origenes["origen_utc"].iloc[0].tz_localize(None)   # 31/12/2022 12:00 local
y_aj, X_aj = y[y.index < fin_ajuste], X[X.index < fin_ajuste]
print(f"Ventana de ajuste: {len(y_aj)} horas ({y_aj.isna().sum()} vacías por COVID),"
      f" hasta {y_aj.index[-1]} UTC")

# --- t_ajuste: FGLS completo hasta la regla de parada ---
inicio = time.perf_counter()
res, informe = ajustar_fgls(y_aj, X_aj, order=(2, 0, 1))
t_ajuste = time.perf_counter() - inicio
print(f"t_ajuste: {t_ajuste / 60:.1f} min; vueltas {informe['vueltas']},"
      f" convergió: {informe['convergio']}, log-verosimilitud {informe['llf']:.1f}")

# --- t_origen: previsión con parámetros fijos en 10 orígenes (sin mirar errores) ---
Xc = con_constante(X)
tiempos = []
for origen in origenes.head(10).itertuples():
    o = origen.origen_utc.tz_localize(None)
    inicio = time.perf_counter()
    res_o = res.apply(y[y.index < o], exog=Xc[Xc.index < o])        # solo horas terminadas
    futuro = Xc[(Xc.index >= o) & (Xc.index < o + 37 * v.UNA_HORA)]
    prevision = res_o.forecast(steps=37, exog=futuro)
    tiempos.append(time.perf_counter() - inicio)
    assert len(prevision) == 37
t_origen = float(np.mean(tiempos))
print(f"t_origen: {t_origen:.2f} s (media de 10 orígenes)")

# --- Criterio del protocolo (apartado 8) ---
horas_sel = 20 * (t_ajuste + 365 * t_origen) * 1.2 / 3600
horas_fin = (12 * t_ajuste * 1.3 + 366 * t_origen) / 3600
print(f"Selección: 20 x (t_ajuste + 365 x t_origen) x 1,2 = {horas_sel:.1f} h"
      f" -> {'CABE' if horas_sel <= 10 else 'NO CABE'} (límite 10 h)")
print(f"Evaluación final: 12 x t_ajuste x 1,3 + 366 x t_origen = {horas_fin:.1f} h"
      f" -> {'reajuste mensual' if horas_fin <= 8 else 'reajuste TRIMESTRAL para los cuatro'}"
      f" (límite 8 h)")
