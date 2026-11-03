"""Previsiones de la evaluación final en 2024 (protocolo 15): reajuste mensual de los cuatro modelos.

Genera, por componente y mes, las previsiones puntuales, los cuantiles y los intervalos propios
de 2024 y los guarda en data/interim/final_2024/. Cuando están los 12 meses de los cinco
componentes, escribe data/predicciones/{modelo}_prueba.parquet, {modelo}_cuantiles_prueba.parquet
y {lightgbm,nhits}_intervalos_prueba.parquet.

NO imprime ninguna métrica de error, de cobertura ni de sesgo (15.1.4): solo tiempos, número de
árboles o de pasos, convergencia, pruebas de fuga y recuentos de control. La primera cifra de
error de 2024 que se ve es la de 29_evaluacion_final_2024.py, que se ejecuta una sola vez.
Reanudable: un (componente, mes) ya guardado no se repite; los reajustes son independientes y
deterministas, así que reanudar no cambia ninguna cifra.

Uso (ejecución completa, un terminal):
    python src/28_prevision_final_2024.py
En dos terminales (15.10.3), sin cambiar ningún cálculo:
    terminal A:  python src/28_prevision_final_2024.py --componentes ingenuo,sarimax,lightgbm
    terminal B:  python src/28_prevision_final_2024.py --componentes nhits
Recortes (15.10.4), solo si la proyección supera las 8 h y con la confirmación del autor:
    --recorte 1   segunda red de N-HiTS con reajuste trimestral
    --recorte 2   además, los 17 cuantiles de LightGBM distintos de 0,05 y 0,95, trimestrales
"""
import argparse
import time

import numpy as np

import finales as F
import validacion as v

PRESUPUESTO_H = 8.0
FACTOR = 1.3                                     # la serie crece con cada reajuste (12.4.3)


def lista_componentes(texto):
    pedidos = []
    for c in texto.split(","):
        c = c.strip()
        if c == "nhits":
            pedidos += ["nhits", "nhits19"]
        elif c in F.COMPONENTES:
            pedidos.append(c)
        else:
            raise SystemExit(f"componente desconocido: {c}")
    return list(dict.fromkeys(pedidos))


def proyeccion(comp, minutos_primer_mes, n_trabajos):
    horas = minutos_primer_mes * n_trabajos * FACTOR / 60
    print(f"  PROYECCIÓN {comp}: {minutos_primer_mes:.1f} min el primer reajuste → "
          f"{horas:.1f} h con los {n_trabajos} entrenamientos y el factor {FACTOR}", flush=True)
    return horas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--componentes", default="ingenuo,sarimax,lightgbm,nhits")
    ap.add_argument("--meses", default="1-12", help="solo para pruebas: p. ej. 1,7 o 1-3")
    ap.add_argument("--recorte", type=int, default=0, choices=(0, 1, 2))
    a = ap.parse_args()
    if F.ARNES:
        F.aplicar_arnes()
        print("*** ARNÉS DE PRUEBAS: presupuestos reducidos y demanda de 2024 inventada ***")
    meses = sorted({int(x) for p in a.meses.split(",")
                    for x in (range(int(p.split("-")[0]), int(p.split("-")[1]) + 1) if "-" in p else [p])})
    comps = lista_componentes(a.componentes)
    datos = v.cargar_datos(incluir_prueba=True)          # carga completa, con la huella comprobada (2.7)
    cal, origenes = F.calendario()
    por_k = {k: F.origenes_mes(origenes, k) for k in range(1, 13)}
    fin = {int(r.k): r.fin for r in cal.itertuples()}
    print(f"Calendario: 12 reajustes, 366 orígenes, 8.784 horas; recorte {a.recorte}; meses {meses}")
    ctx = {"recorte": a.recorte}
    if "ingenuo" in comps:
        print("Ingenuo: errores 2017-2024 con su instante de cierre (solo se usan los anteriores a cada reajuste)", flush=True)
        ctx["errores"] = F.errores_ingenuo_fechados(datos)
    funciones = {"ingenuo": F.componente_ingenuo, "sarimax": F.componente_sarimax,
                 "lightgbm": F.componente_lightgbm, "nhits": F.componente_nhits}
    proyectadas = {}
    t_total = time.perf_counter()
    for comp in comps:
        print(f"\n{comp}:", flush=True)
        # Trabajos: (mes del reajuste, meses que prevé). Salvo el recorte 1 de la segunda red, uno por mes.
        if comp == "nhits19" and a.recorte >= 1:
            trabajos = [(q, [m for m in range(q, q + 3) if m in meses and not F.hecho(comp, m)])
                        for q in F.TRIMESTRALES]
        else:
            trabajos = [(m, [m] if not F.hecho(comp, m) else []) for m in meses]
        n_trab = 4 if (comp == "nhits19" and a.recorte >= 1) else 12
        primero = True
        for mes, grupo in trabajos:
            if not grupo:
                print(f"  reajuste {mes:02d}: ya guardado (o fuera de la selección)", flush=True)
                continue
            print(f"  reajuste {mes:02d} (origen {fin[mes].tz_convert(v.ZONA):%d/%m/%Y %H:%M} local)"
                  + (f" → meses {grupo}" if len(grupo) > 1 else ""), flush=True)
            t0 = time.perf_counter()
            if comp == "nhits19":
                for m, (tabla, meta) in F.componente_nhits19(datos, grupo, fin[mes], por_k, ctx).items():
                    F.guardar_mes(comp, m, tabla, meta)
            else:
                tabla, meta = funciones[comp](datos, mes, fin[mes], por_k[mes], ctx)
                F.guardar_mes(comp, mes, tabla, meta)
            dur = (time.perf_counter() - t0) / 60 - (meta.get("minutos_prueba_reajuste", 0.0) if comp != "nhits19" else 0.0)
            print(f"  guardado ({dur:.1f} min)", flush=True)
            if primero and comp != "ingenuo":
                proyectadas[comp] = proyeccion(comp, dur, n_trab)
                primero = False
    if proyectadas:
        total = sum(proyectadas.values())
        print(f"\nSuma de las proyecciones de esta ejecución: {total:.1f} h (presupuesto {PRESUPUESTO_H:.0f} h)")
        if total > PRESUPUESTO_H:
            print("  SUPERA el presupuesto: interrumpe (Ctrl+C) y relanza con --recorte 1 (15.10.4); "
                  "pide antes la confirmación del autor.")
    print(f"\nTiempo de esta ejecución: {(time.perf_counter() - t_total) / 3600:.2f} h")

    faltan = [(c, k) for c in F.COMPONENTES for k in range(1, 13) if not F.hecho(c, k)]
    if faltan:
        print(f"Faltan {len(faltan)} de 60 (componente, mes); cuando estén todos, repite este comando para ensamblar.")
        return
    resumen = F.ensamblar(datos)
    print("\nEnsamblado: archivos *_prueba.parquet, *_cuantiles_prueba.parquet e *_intervalos_prueba.parquet")
    print("Recuentos de control (sin interpretar todavía):")
    print(resumen.to_string(index=False))
    print("\nListo. Siguiente: python src/29_evaluacion_final_2024.py (una sola vez). No abras antes data/predicciones/*_prueba*.")


if __name__ == "__main__":
    main()
