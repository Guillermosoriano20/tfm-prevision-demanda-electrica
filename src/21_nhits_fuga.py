"""N-HiTS: prueba de fuga y comprobaciones del escalado por ventana (paso N3).

Protocolo: docs/protocolo_oe5.md, apartado 12.3 (commit 2013525), con el diseño de 12.1 y
12.2. Modelo de referencia: arquitectura por defecto de N-HiTS, L = 336, 100 pasos,
semilla 42, ajustado con los datos hasta el 31/12/2022 12:00. No se calcula ningún error
de 2023: solo se comparan previsiones entre sí. Las letras (a)-(k) son las de la tabla
del apartado 12.3.3; si una falla, el script se detiene (12.3.4).
"""
import os
import time

import numpy as np
import pandas as pd

import validacion as v
import nhits_modelo as nm

L_REF, PASOS_REF = 336, 100                  # modelo de referencia (12.3.2)
HORAS_CONTROL, L_MAXIMA = 8 * 168, 672       # parada temprana (2.5) y mayor L de 12.1.2


def cronometro(inicio):
    return f"({time.perf_counter() - inicio:.0f} s)"


nucleos = os.cpu_count() or 0
print(f"Núcleos lógicos del equipo: {nucleos}; hilos de torch: {nm.HILOS}")
if nucleos < nm.HILOS:
    print("  AVISO: menos núcleos que hilos; el ajuste puede ser mucho más lento")

datos = v.cargar_datos()                                   # huella comprobada; 2024 cortado
origenes = v.generar_origenes("validacion")
fin = origenes["origen_utc"].iloc[0]                       # 31/12/2022 12:00 local

# --- (a) Pesos del horizonte (12.1.11) ---
w = nm.PESOS_HORIZONTE
assert len(w) == nm.H == 37 and (w[:12] == 0).all() and (w[12:] == 1).all()
print(f"(a) Pesos del horizonte: {len(w)} valores, {int((w == 0).sum())} ceros y"
      f" {int((w == 1).sum())} unos")

# --- Ajuste del modelo de referencia; (b) se comprueba dentro de nm.ajustar ---
inicio = time.perf_counter()
nf = nm.ajustar(datos, fin, nm.crear_modelo(L_REF, PASOS_REF, val_check_steps=PASOS_REF))
print(f"Ajuste del modelo de referencia ({PASOS_REF} pasos) {cronometro(inicio)}")
print("(b) Sin escalado de la serie completa (local_scaler_type=None, sin escaladores);"
      " demanda por ventana (robust); exógenas sin escalar: correcto")

# --- (c) Datos del ajuste ---
horas = nm.horas_ajuste(datos, fin)
ultima = horas[-1].tz_convert(v.ZONA)
assert horas[-1] + v.UNA_HORA == fin and len(nf.dataset.temporal) == len(horas)
print(f"(c) Datos del ajuste: {len(horas)} horas; la última, de {ultima:%d/%m/%Y %H:%M}"
      f" a {ultima + v.UNA_HORA:%H:%M} (local)")

# --- (d) Ventanas válidas: a mano frente a la librería (12.1.4) ---
mascara = 1 - datos.loc[horas, "excluir_covid"].astype(int).to_numpy()
largo = L_REF + nm.H
a_mano = int((np.convolve(mascara, np.ones(largo, int), "valid") == largo).sum())
posibles = len(horas) - largo + 1
modelo = nf.models[0]
lote = {"temporal": nf.dataset.temporal.T.unsqueeze(0), "temporal_cols": nf.dataset.temporal_cols,
        "static": None, "static_cols": None}
de_libreria = len(modelo._create_windows(lote, step="train")[3])
print(f"(d) Ventanas de {largo} h: {posibles} posibles; válidas a mano {a_mano},"
      f" según la librería {de_libreria}; descartadas por el COVID {posibles - a_mano}")
assert a_mano == de_libreria

# --- (e) Las ventanas de la parada temprana no tocan el COVID ---
desde = fin - (HORAS_CONTROL + L_MAXIMA) * v.UNA_HORA
tramo = datos[(datos.index >= desde) & (datos.index < fin)]
assert not tramo["excluir_covid"].any()
print(f"(e) Horas que pueden usar las ventanas de control (desde"
      f" {desde.tz_convert(v.ZONA):%d/%m/%Y %H:%M}): ninguna en el periodo COVID")

# --- (f), (g), (h) Prueba de fuga en las tres salidas, 365 orígenes de 2023 ---
pruebas = [("(f) Prueba de fuga (demanda futura +10.000 MW)", nm.hacer_prever(nf), "desplazar"),
           ("(g) Prueba de fuga reforzada (demanda futura borrada)", nm.hacer_prever(nf), "borrar"),
           ("(h) Control positivo (ventana que termina 24 h después del origen)",
            nm.hacer_prever(nf, desfase=24), "desplazar")]
for texto, prever, modo in pruebas:
    inicio = time.perf_counter()
    fugas = v.comprobar_fuga(prever, datos, origenes, modo=modo)
    print(f"{texto}: {len(fugas)} días con fuga {fugas[:3]} {cronometro(inicio)}", flush=True)
    esperado = len(origenes) if texto.startswith("(h)") else 0
    assert len(fugas) == esperado, "RESULTADO DISTINTO DEL CRITERIO: se detiene todo"

# --- (i) Demanda de 2017-2021 multiplicada por 2: previsiones idénticas bit a bit ---
inicio = time.perf_counter()
prever = nm.hacer_prever(nf)
dobles = datos.copy()
antiguo = dobles["fecha_local"] < pd.Timestamp("2022-01-01")
dobles.loc[antiguo, "demanda_mw"] *= 2
distintos = [str(o.fecha_objetivo) for o in origenes.itertuples()
             if not prever(datos, o).equals(prever(dobles, o))]
print(f"(i) Demanda de 2017-2021 x 2: {len(distintos)} orígenes con previsiones distintas"
      f" de {len(origenes)} {cronometro(inicio)}")
assert not distintos

# --- (j) y (k): 10 orígenes repartidos por el año ---
muestra = origenes.iloc[::37].head(10)
subida = datos.copy()
subida["demanda_mw"] += 1000
calor = datos.copy()
calor["t_ponderada"] += 5
desvio_j, cambio_k = 0.0, 0.0
for o in muestra.itertuples():
    base = prever(datos, o)
    desvio_j = max(desvio_j, float(((prever(subida, o) - base) - 1000).abs().max().max()))
    cambio_k = max(cambio_k, float((prever(calor, o) - base).abs().max().max()))
print(f"(j) +1.000 MW en la ventana de entrada: las tres salidas suben 1.000 MW con un"
      f" desvío máximo de {desvio_j:.4f} MW (tolerancia 1 MW)")
print(f"(k) +5 °C en t_ponderada: cambio máximo de la previsión {cambio_k:.1f} MW"
      f" (debe ser distinto de 0)")
assert desvio_j <= 1 and cambio_k > 0
print("\nTodas las comprobaciones (a)-(k) cumplen el criterio del apartado 12.3.")
