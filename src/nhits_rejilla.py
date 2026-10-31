"""N-HiTS con la rejilla de 19 cuantiles: segunda red para el CRPS (protocolo 14.4.4, D11).

Reutiliza de nhits_modelo.py todo lo que no depende de la lista de cuantiles (tabla de
entrada, ajuste con comprobación de escalado, exógenas futuras) y duplica solo lo que sí:
la pérdida, el procedimiento de 12.5.1 y la lectura de las salidas. No toca el módulo
cerrado y validado. La red oficial (c13, tres cuantiles) no cambia: esta es otra red.
"""
import pandas as pd
import torch
from neuralforecast.losses.pytorch import MQLoss
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler

import nhits_modelo as nm
import validacion as v
from probabilistica import COLUMNAS_Q, NIVELES

TAUS = [float(t) for t in NIVELES]


def config_c13():
    """Hiperparámetros de c13 (12.5.3), reconstruidos con el mismo sorteo y comprobados."""
    espacio = {"input_size": [168, 336, 504], "learning_rate": loguniform(1e-4, 3e-3),
               "ancho": [128, 256, 512], "windows_batch_size": [256, 512, 1024],
               "escalas": ["A", "B", "C"], "dropout_prob_theta": [0.0, 0.1]}
    c = list(ParameterSampler(espacio, n_iter=19, random_state=42))[12]            # c13
    c = {k: (x if isinstance(x, str) else float(x) if k in ("learning_rate", "dropout_prob_theta")
             else int(x)) for k, x in c.items()}
    assert (c["input_size"], c["ancho"], c["windows_batch_size"], c["escalas"],
            c["dropout_prob_theta"], round(c["learning_rate"], 5)) == (336, 256, 1024, "B", 0.1, 0.00119)
    return c


def crear_modelo(L, max_steps, semilla=nm.SEMILLA, **hiper):
    """Como nhits_modelo.crear_modelo, con la pérdida multicuantil de los 19 niveles."""
    return nm.NHITSExogSinEscalar(
        h=nm.H, input_size=L, futr_exog_list=nm.EXOGENAS, cat_exog_list=nm.CATEGORICAS,
        categorical_cardinalities={"dia_semana": 7},
        loss=MQLoss(quantiles=TAUS, horizon_weight=nm.PESOS_HORIZONTE),
        scaler_type="robust", training_data_availability_threshold=[1.0, 1.0],
        max_steps=max_steps, random_seed=semilla, alias="NHITS",
        accelerator="cpu", devices=1, deterministic=True, logger=False,
        enable_progress_bar=False, enable_model_summary=False, **hiper)


def entrenar(datos, fin, L, semilla=nm.SEMILLA, **hiper):
    """Procedimiento de 12.5.1 (parada temprana con 8 semanas de control y reentrenamiento)."""
    previo = crear_modelo(L, nm.P_MAX, semilla, val_check_steps=nm.CONTROL_CADA,
                          early_stop_patience_steps=nm.PACIENCIA, num_sanity_val_steps=0, **hiper)
    ajustado = nm.ajustar(datos, fin, previo, val_size=nm.HORAS_CONTROL).models[0]
    historial = [(int(p), float(x)) for p, x in ajustado.valid_trajectories]
    assert historial and all(p % nm.CONTROL_CADA == 0 for p, _ in historial)
    paso, perdida = min(historial, key=lambda x: x[1])
    nf = nm.ajustar(datos, fin, crear_modelo(L, paso, semilla, val_check_steps=paso, **hiper))
    return nf, paso, perdida


def hacer_prever(nf):
    """prever(datos, origen): los 37 pasos con las 19 salidas, columnas q05 … q95."""
    L = nf.models[0].input_size
    salidas = ["NHITS" + s for s in nf.models[0].loss.output_names]
    assert len(salidas) == len(COLUMNAS_Q)                      # orden ascendente de τ

    def prever(datos, origen):
        torch.set_num_threads(nm.HILOS)
        entrada = nm.horas_ajuste(datos, origen.origen_utc)[-L:]
        futuras = pd.date_range(origen.origen_utc, periods=nm.H, freq="h").as_unit(datos.index.unit)
        p = nf.predict(df=nm.tabla(datos, entrada),
                       futr_df=nm.exogenas_futuras(datos, futuras, origen))
        assert len(p) == nm.H and (p["ds"].to_numpy() == futuras.tz_convert(None).to_numpy()).all()
        indice = pd.date_range(origen.origen_utc, periods=nm.H, freq="h").as_unit(datos.index.unit)
        return p[salidas].set_axis(COLUMNAS_Q, axis=1).set_axis(indice)
    return prever
