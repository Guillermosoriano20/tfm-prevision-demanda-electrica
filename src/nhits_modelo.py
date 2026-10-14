"""N-HiTS: tabla de entrada de NeuralForecast, modelo, ajuste y previsión.

Implementa docs/protocolo_oe5.md, apartados 12.1 (diseño, commits 258a19a y 4dc9142),
12.2 (intervalos, commit 44a9ab5) y las comprobaciones de 12.3 (commit 2013525). Módulo
sin número porque se importa, como validacion.py y lightgbm_modelo.py.

Modelo de trayectoria: la ventana de entrada son las últimas L horas de demanda hasta el
origen y la red dibuja de una vez los 37 pasos UTC siguientes; validacion.recortar se
queda con el día local D+1. Una sola red da los cuantiles 0,05, 0,5 y 0,95; la mediana es
la previsión puntual (12.2).
"""
import logging
import warnings

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

import validacion as v
from exogenas import grados_dia, navidad

H = 37                                               # pasos UTC desde el origen (12.1.1)
PESOS_HORIZONTE = np.r_[np.zeros(12), np.ones(H - 12)]   # anticipación 12-36 h (12.1.11)
CUANTILES = [0.05, 0.5, 0.95]                        # 12.2.1
SALIDAS = {"NHITS-lo-90.0": "q05", "NHITS-median": "mediana", "NHITS-hi-90.0": "q95"}
NUMERICAS = ["gd_calefaccion", "gd_refrigeracion", "t_ponderada", "hora_sen", "hora_cos",
             "anio_sen", "anio_cos", "fraccion_festivo", "puente", "navidad"]
CATEGORICAS = ["dia_semana"]                         # embedding de la librería (12.1.5)
EXOGENAS = NUMERICAS + CATEGORICAS
SERIE = "peninsula"
HILOS = 4                                            # como LightGBM (12.1.9)
SEMILLA = 42

# Lightning escribe varias líneas en cada ajuste y en cada previsión; solo errores. El
# aviso interno de LeafSpec (pytorch-lightning 2.5.6 con torch 2.14.1) sale en cada
# previsión y no afecta a los cálculos.
for nombre in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric", "lightning"):
    logging.getLogger(nombre).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*LeafSpec.*")


class NHITSExogSinEscalar(NHITS):
    """N-HiTS con la demanda normalizada por ventana y las exógenas sin escalar (12.1.6).

    La librería normaliza por ventana la demanda y todas las exógenas numéricas que
    devuelve este método (su único uso es elegir esas columnas). Devolviendo una lista
    vacía, solo la demanda pasa por el normalizador.
    """

    def _get_temporal_exogenous_cols(self, temporal_cols):
        return []


def crear_modelo(L, max_steps, semilla=SEMILLA, **hiper):
    """Modelo con todo lo fijado en 12.1 y 12.2; `hiper` para la arquitectura (12.5)."""
    return NHITSExogSinEscalar(
        h=H, input_size=L, futr_exog_list=EXOGENAS, cat_exog_list=CATEGORICAS,
        categorical_cardinalities={"dia_semana": 7},
        loss=MQLoss(quantiles=CUANTILES, horizon_weight=PESOS_HORIZONTE),
        scaler_type="robust",                          # por ventana (12.1.6)
        training_data_availability_threshold=[1.0, 1.0],   # COVID estricto (12.1.4)
        max_steps=max_steps, random_seed=semilla, alias="NHITS",
        accelerator="cpu", devices=1, deterministic=True, logger=False,
        enable_progress_bar=False, enable_model_summary=False, **hiper)


def tabla(datos, horas, con_demanda=True):
    """Tabla de NeuralForecast para las horas dadas (índice UTC): demanda, exógenas y máscara.

    Las exógenas no usan la demanda. Los grados-día se calculan con los días locales
    completos que tocan las horas pedidas (media diaria por provincia, exogenas.grados_dia).
    """
    horas = pd.DatetimeIndex(horas)
    filas = datos.loc[horas]
    dias = datos[datos["fecha_local"].isin(filas["fecha_local"].unique())]
    gd = grados_dia(dias).loc[horas]
    hora = filas["hora_del_dia"].to_numpy(float)
    dia_anio = filas["fecha_local"].dt.dayofyear.to_numpy(float) - 1
    t = pd.DataFrame({
        "unique_id": SERIE,
        "ds": horas.tz_convert(None),                  # UTC sin zona (NeuralForecast)
        "gd_calefaccion": gd["gd_calefaccion"].to_numpy(),
        "gd_refrigeracion": gd["gd_refrigeracion"].to_numpy(),
        "t_ponderada": filas["t_ponderada"].to_numpy(),
        "hora_sen": np.sin(2 * np.pi * hora / 24),
        "hora_cos": np.cos(2 * np.pi * hora / 24),
        "anio_sen": np.sin(2 * np.pi * dia_anio / 365.25),
        "anio_cos": np.cos(2 * np.pi * dia_anio / 365.25),
        "fraccion_festivo": filas["fraccion_festivo"].to_numpy(),
        "puente": filas["puente"].astype(float).to_numpy(),
        "navidad": navidad(filas).to_numpy(),
        "dia_semana": filas["dia_semana"].to_numpy(),
    })
    if con_demanda:
        t.insert(2, "y", filas["demanda_mw"].to_numpy())
        t["available_mask"] = 1.0 - filas["excluir_covid"].astype(float).to_numpy()
    return t


def horas_ajuste(datos, fin):
    """Horas ya terminadas en el origen del ajuste: hora_utc + 1 h <= fin (12.1.3)."""
    return datos.index[datos.index + v.UNA_HORA <= fin]


def comprobar_escalado(nf):
    """Apartado 12.1.7 (a) y (b): ningún escalado de la serie completa; demanda por ventana."""
    modelo = nf.models[0]
    assert nf.local_scaler_type is None, "escalado de la serie completa activado"
    assert not nf.scalers_, "hay escaladores ajustados sobre la serie completa"
    assert modelo.scaler.scaler_type == "robust"
    assert modelo._get_temporal_exogenous_cols(pd.Index(EXOGENAS)) == []
    assert np.array_equal(modelo.loss.horizon_weight.numpy(), PESOS_HORIZONTE)


def ajustar(datos, fin, modelo, val_size=0):
    """Ajusta el modelo con las horas terminadas en `fin` y comprueba el escalado."""
    torch.set_num_threads(HILOS)
    df = tabla(datos, horas_ajuste(datos, fin))
    assert (df["ds"] + v.UNA_HORA <= fin.tz_convert(None)).all()
    nf = NeuralForecast(models=[modelo], freq="h", local_scaler_type=None)
    nf.fit(df=df, val_size=val_size)
    comprobar_escalado(nf)
    return nf


def exogenas_futuras(datos, futuras, origen):
    """Exógenas de los 37 pasos (protocolo 12.1.12).

    Las horas que no están en los datos (paso 37 del último origen de cada conjunto: los
    datos acaban con su día objetivo) toman las exógenas de la última hora disponible;
    nunca es una hora evaluada (se comprueba).
    """
    hay = futuras[futuras.isin(datos.index)]
    futr = tabla(datos, hay, con_demanda=False)
    faltan = futuras[len(hay):]
    if len(faltan):
        assert (futuras[:len(hay)] == hay).all() and (faltan >= origen.fin_utc).all()
        relleno = futr.iloc[[-1] * len(faltan)].assign(ds=faltan.tz_convert(None))
        futr = pd.concat([futr, relleno], ignore_index=True)
    return futr


def hacer_prever(nf, salida=None, desfase=0):
    """Interfaz común prever(datos, origen).

    Ventana de entrada: las últimas L horas con hora_utc + 1 h <= origen (no lee demanda
    posterior). Devuelve los 37 pasos desde el origen: una tabla con las tres salidas
    (q05, mediana, q95) o, con `salida`, solo esa columna como serie.
    `desfase` (horas) solo para el control positivo de la prueba de fuga (12.3, h): la
    ventana termina `desfase` horas después del origen, lo que es una fuga real.
    """
    L = nf.models[0].input_size

    def prever(datos, origen):
        torch.set_num_threads(HILOS)
        fin = origen.origen_utc + desfase * v.UNA_HORA
        entrada = horas_ajuste(datos, fin)[-L:]
        futuras = pd.date_range(fin, periods=H, freq="h").as_unit(datos.index.unit)
        p = nf.predict(df=tabla(datos, entrada),
                       futr_df=exogenas_futuras(datos, futuras, origen))
        assert len(p) == H and (p["ds"].to_numpy() == futuras.tz_convert(None).to_numpy()).all()
        indice = pd.date_range(origen.origen_utc, periods=H, freq="h").as_unit(datos.index.unit)
        tabla_p = p[list(SALIDAS)].rename(columns=SALIDAS).set_axis(indice)
        return tabla_p if salida is None else tabla_p[salida]
    return prever
