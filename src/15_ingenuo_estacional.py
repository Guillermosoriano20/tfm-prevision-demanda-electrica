"""Ingenuo estacional: misma hora local de la semana anterior. Validación 2023.

Definición del protocolo (docs/protocolo_oe5.md, apartado 5.2).
"""
import pandas as pd

import validacion as v


def prever_ingenuo(datos, origen):
    """Copia la misma hora local del mismo día de la semana anterior.

    Día de 25 h: las dos 02:00 copian la única 02:00 de la semana anterior.
    Semana anterior de 23 h (sin 02:00): media de 01:00 y 03:00.
    Semana anterior de 25 h (dos 02:00): media de las dos.
    """
    horas = pd.date_range(origen.inicio_utc, origen.fin_utc, freq="h",
                          inclusive="left").as_unit(datos.index.unit)
    dia_fuente = pd.Timestamp(origen.fecha_objetivo) - 7 * v.UN_DIA
    fuente = datos[datos["fecha_local"] == dia_fuente]
    perfil = fuente.groupby("hora_del_dia")["demanda_mw"].mean()  # dos 02:00 -> media
    perfil = perfil.reindex(range(24)).interpolate()  # falta la 02:00 -> media de 01:00 y 03:00
    hora_local = horas.tz_convert(v.ZONA).hour
    return pd.Series(perfil.loc[hora_local].to_numpy(), index=horas)


def demanda_local(datos, fecha, hora):
    """Demanda real de una fecha y hora locales (una o dos filas si es la 02:00 de octubre)."""
    filas = datos[(datos["fecha_local"] == pd.Timestamp(fecha)) & (datos["hora_del_dia"] == hora)]
    return filas["demanda_mw"]


if __name__ == "__main__":
    datos = v.cargar_datos()
    origenes = v.generar_origenes("validacion")

    fugas = v.comprobar_fuga(prever_ingenuo, datos, origenes)
    print("Días con fuga:", fugas)
    assert not fugas

    # Reglas del cambio de hora, sobre los cuatro casos reales de 2023
    por_fecha = {str(o.fecha_objetivo): o for o in origenes.itertuples()}
    def a_las_dos(fecha):
        p = prever_ingenuo(datos, por_fecha[fecha])
        return p[p.index.tz_convert(v.ZONA).hour == 2]
    print("26/03 (día de 23 h), horas previstas:", len(prever_ingenuo(datos, por_fecha["2023-03-26"])))
    print("29/10 (día de 25 h), las dos 02:00 copian la del 22/10:",
          (a_las_dos("2023-10-29") == demanda_local(datos, "2023-10-22", 2).iloc[0]).all())
    print("02/04, la 02:00 es la media de 01:00 y 03:00 del 26/03:",
          a_las_dos("2023-04-02").iloc[0] == (demanda_local(datos, "2023-03-26", 1).iloc[0]
                                              + demanda_local(datos, "2023-03-26", 3).iloc[0]) / 2)
    print("05/11, la 02:00 es la media de las dos 02:00 del 29/10:",
          a_las_dos("2023-11-05").iloc[0] == demanda_local(datos, "2023-10-29", 2).mean())

    tabla = v.evaluar(prever_ingenuo, datos, "validacion", "ingenuo")
    print(f"Guardado: {len(tabla)} horas en data/predicciones/ingenuo_validacion.parquet")
    print(v.metricas(tabla, v.escala_mase(datos)).round(3).to_string())
