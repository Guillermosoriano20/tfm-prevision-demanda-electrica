"""N-HiTS: prueba de tiempos en CPU (paso N4).

Protocolo: docs/protocolo_oe5.md, apartado 12.4, fijado antes de medir. Mide el tiempo de
un paso de entrenamiento con la configuración más pesada prevista para la búsqueda (12.5)
y el de una previsión, y aplica el criterio de "cabe" y el orden de recortes. Solo mide
tiempos: no calcula ningún error ni mira ninguna previsión de 2023.
"""
import os
import time

import validacion as v
import nhits_modelo as nm

PASOS_MEDIDA, CONTROL_CADA = 200, 100        # 200 pasos, control cada 100 (12.4.2)
HORAS_CONTROL = 8 * 168                      # parada temprana: últimas 8 semanas (2.5)
N_ORIGENES = 10                              # previsiones medidas (solo tiempo)
CONFIGURACIONES = {                          # L y arquitectura (12.4.2 y 12.4.4, paso 2)
    "más pesada prevista": (672, {"mlp_units": 3 * [[1024, 1024]], "n_blocks": [2, 2, 2],
                                  "windows_batch_size": 1024}),
    "reducida (paso 2)": (504, {"mlp_units": 3 * [[512, 512]], "n_blocks": [1, 1, 1],
                                "windows_batch_size": 1024}),
}
P, P_MINIMO = 1000, 500                      # tope de pasos (provisional) y mínimo (12.4.4)
N_CONF = 22                                  # 20 de la búsqueda + 2 semillas (3.5)
LIMITE_BUSQUEDA, LIMITE_FINAL = 10 * 3600, 8 * 3600     # segundos


def busqueda(p, t_paso, t_prever):
    return N_CONF * (2 * p * t_paso + 365 * t_prever) * 1.2


def final(p, t_paso, t_prever):
    return 12 * 2 * p * t_paso * 1.3 + 366 * t_prever


def medir(nombre, L, arquitectura):
    """t_paso (200 pasos con el control activo, arranque incluido) y t_prever."""
    modelo = nm.crear_modelo(L, PASOS_MEDIDA, val_check_steps=CONTROL_CADA, **arquitectura)
    n_par = f"{sum(p.numel() for p in modelo.parameters()):,}".replace(",", ".")
    print(f"\n=== Configuración {nombre}: L = {L}, capas {arquitectura['mlp_units'][0]},"
          f" bloques {arquitectura['n_blocks']},"
          f" {arquitectura['windows_batch_size']} ventanas por lote;"
          f" {n_par} parámetros ===", flush=True)
    inicio = time.perf_counter()
    nf = nm.ajustar(datos, fin, modelo, val_size=HORAS_CONTROL)
    t_ajuste = time.perf_counter() - inicio
    t_paso = t_ajuste / PASOS_MEDIDA
    print(f"Entrenamiento de {PASOS_MEDIDA} pasos: {t_ajuste:.1f} s -> t_paso = {t_paso:.3f} s")
    prever = nm.hacer_prever(nf)
    prever(datos, next(origenes.itertuples()))              # la primera calienta la librería
    inicio = time.perf_counter()
    for origen in origenes.head(N_ORIGENES).itertuples():
        prever(datos, origen)
    t_prever = (time.perf_counter() - inicio) / N_ORIGENES
    print(f"Previsión: t_prever = {t_prever:.3f} s (media de {N_ORIGENES} orígenes)")
    return t_paso, t_prever


def criterio(t_paso, t_prever):
    """Devuelve el P que cabe (P = 1000 o el paso 1 del orden de recortes) o None."""
    print(f"Con P = {P}: T_conf = {(2 * P * t_paso + 365 * t_prever) / 60:.1f} min por"
          f" configuración (peor caso)")
    print(f"  Búsqueda + semillas: {N_CONF} x T_conf x 1,2 ="
          f" {busqueda(P, t_paso, t_prever) / 3600:.2f} h (límite 10 h)")
    print(f"  Evaluación final de 2024: {final(P, t_paso, t_prever) / 3600:.2f} h (límite 8 h)")
    for p in range(P, P_MINIMO - 1, -100):           # paso 1: mayor P múltiplo de 100
        if (busqueda(p, t_paso, t_prever) <= LIMITE_BUSQUEDA
                and final(p, t_paso, t_prever) <= LIMITE_FINAL):
            return p
    return None


print(f"Núcleos lógicos del equipo: {os.cpu_count()}; hilos de torch: {nm.HILOS}")
datos = v.cargar_datos()
origenes = v.generar_origenes("validacion")
fin = origenes["origen_utc"].iloc[0]                       # 31/12/2022 12:00 local

for nombre, (L, arquitectura) in CONFIGURACIONES.items():
    t_paso, t_prever = medir(nombre, L, arquitectura)
    p = criterio(t_paso, t_prever)
    if p is not None:
        print(f"CABE con la configuración {nombre} y P = {p}"
              f" (búsqueda {busqueda(p, t_paso, t_prever) / 3600:.2f} h,"
              f" final {final(p, t_paso, t_prever) / 3600:.2f} h).")
        break
    print(f"NO CABE ni con P = {P_MINIMO}.")
else:
    print("\nNinguna de las dos configuraciones cabe: paso 3 (12 configuraciones),"
          " solo por decisión del autor.")
