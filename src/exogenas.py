"""Variables exógenas comunes (docs/protocolo_oe5.md, apartado 4) y su codificación
para SARIMAX: términos de Fourier sobre la hora local e interacciones.

Módulo sin número porque se importa, como validacion.py. Ninguna variable usa la
demanda: todas se conocen de antemano (calendario) o pertenecen al escenario de
previsión meteorológica perfecta (temperatura), así que no introducen fuga.
"""
import numpy as np
import pandas as pd

from validacion import RAIZ

TC, TR = 15.5, 19.5  # umbrales de grados-día fijados con 2017-2022 (no se reestiman)


def grados_dia(datos):
    """Grados-día diarios por provincia, ponderados por población, repetidos en cada hora."""
    pesos = pd.read_csv(RAIZ / "referencia" / "pesos_poblacion_2021.csv",
                        dtype={"cod_provincia": str})
    cols = ["t_" + c for c in pesos["cod_provincia"]]
    w = pesos["peso"].to_numpy()
    t_dia = datos.groupby("fecha_local")[cols].mean()          # días × provincias
    gd = pd.DataFrame({
        "gd_calefaccion": np.maximum(TC - t_dia.to_numpy(), 0) @ w,
        "gd_refrigeracion": np.maximum(t_dia.to_numpy() - TR, 0) @ w,
    }, index=t_dia.index)
    return gd.reindex(datos["fecha_local"]).set_axis(datos.index)


def fourier(posicion, periodo, armonicos, prefijo):
    """Pares seno/coseno de los armónicos indicados para una posición dentro del ciclo."""
    columnas = {}
    for k in armonicos:
        angulo = 2 * np.pi * k * posicion / periodo
        if 2 * k != periodo:  # en k = periodo/2 el seno vale siempre 0: solo el coseno
            columnas[f"{prefijo}_sen{k}"] = np.sin(angulo)
        columnas[f"{prefijo}_cos{k}"] = np.cos(angulo)
    return pd.DataFrame(columnas, index=posicion.index)


def exogenas_sarimax(datos, k_dia, k_semana, k_anio, k_interaccion):
    """Matriz exógena de SARIMAX, una fila por hora UTC del índice de `datos`.

    Fourier sobre la hora LOCAL (protocolo): día (24 h), semana (168 h) y año.
    En la semana se omiten los armónicos múltiplos de 7: tienen periodo de 24 h
    o divisores y repetirían exactamente los diarios (columnas duplicadas).
    """
    hora = datos["hora_del_dia"].astype(float)
    hora_semana = datos["dia_semana"] * 24 + hora                 # lunes 00:00 = 0
    dia_anio = datos["fecha_local"].dt.dayofyear - 1 + hora / 24

    semana = [k for k in range(1, 200) if k % 7 != 0][:k_semana]
    partes = [
        fourier(hora, 24, range(1, k_dia + 1), "dia"),
        fourier(hora_semana, 168, semana, "sem"),
        fourier(dia_anio, 365.25, range(1, k_anio + 1), "anio"),
        grados_dia(datos),
    ]
    otras = pd.DataFrame({
        "t_ponderada": datos["t_ponderada"],
        "fraccion_festivo": datos["fraccion_festivo"],
        "puente": datos["puente"].astype(float),
        "navidad": (((datos["fecha_local"].dt.month == 12) & (datos["fecha_local"].dt.day >= 23))
                    | ((datos["fecha_local"].dt.month == 1) & (datos["fecha_local"].dt.day <= 6))
                    ).astype(float),
    }, index=datos.index)
    partes.append(otras)
    if k_interaccion:
        base = fourier(hora, 24, range(1, k_interaccion + 1), "dia")
        partes.append(base.mul(datos["t_ponderada"], axis=0).add_prefix("t_x_"))
    return pd.concat(partes, axis=1)
