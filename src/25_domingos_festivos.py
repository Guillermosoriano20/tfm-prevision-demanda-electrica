"""Fiestas nacionales que caen en domingo (2017-2022): efecto en la demanda diaria.

Apoya la sección de calendario laboral de la memoria (4.1.6). Criterio fijado antes de
ejecutarlo y registrado con el commit de este script (15/10/2026):

- Domingos festivos: las 8 fiestas nacionales de fecha fija que cayeron en domingo en
  2017-2022, que en el conjunto de datos figuran como domingos corrientes (lista
  verificada en 14_exploracion.py, paso 4a).
- Domingo normal: fraccion_festivo = 0, fuera del periodo `navidad` (23/12 a 6/1) y fuera
  del periodo de la pandemia.
- Referencia de cada domingo: media de la demanda media diaria de los 4 domingos normales
  más cercanos en el tiempo (normalmente 2 antes y 2 después; en los extremos del
  periodo, los del lado disponible), sin contarse a sí mismo.
- Diferencia relativa = demanda del domingo / referencia - 1.
- Vara de medir: la misma diferencia calculada para cada domingo normal de 2017-2022.
- Lectura: efecto APRECIABLE si la mediana de las diferencias de los 8 domingos festivos
  es menor que el percentil 10 de las de los domingos normales; si no, DENTRO DE LA
  VARIABILIDAD HABITUAL.

Solo usa 2017-2022: 2023 y 2024 no intervienen. No cambia el conjunto de datos ni ningún
modelo; el resultado es descriptivo.
"""
import numpy as np
import pandas as pd

import validacion as v
from exogenas import navidad

FESTIVOS_EN_DOMINGO = ["2017-01-01", "2019-01-06", "2019-12-08", "2020-11-01",
                       "2020-12-06", "2021-08-15", "2022-05-01", "2022-12-25"]
N_VECINOS = 4
SALIDA = v.RAIZ / "data" / "resultados" / "domingos_festivos.csv"

datos = v.cargar_datos()                                    # huella comprobada
datos = datos[datos["fecha_local"] < v.FIN_ENTRENAMIENTO]   # solo 2017-2022
dias = datos.groupby("fecha_local").agg(
    demanda=("demanda_mw", "mean"), dia_semana=("dia_semana", "first"),
    fraccion_festivo=("fraccion_festivo", "first"), covid=("excluir_covid", "max"))
dias["navidad"] = navidad(dias.index.to_frame(name="fecha_local")).astype(bool).to_numpy()

domingos = dias[dias["dia_semana"] == 6].copy()             # 0 = lunes, 6 = domingo
festivos = pd.to_datetime(FESTIVOS_EN_DOMINGO)
assert festivos.isin(domingos.index).all(), "alguna fecha no es domingo"
assert (domingos.loc[festivos, "fraccion_festivo"] == 0).all()
domingos["festivo_en_domingo"] = domingos.index.isin(festivos)
domingos["normal"] = ((domingos["fraccion_festivo"] == 0) & ~domingos["navidad"]
                      & ~domingos["covid"] & ~domingos["festivo_en_domingo"])
normales = domingos[domingos["normal"]]


def diferencia(fecha):
    """Diferencia relativa frente a los 4 domingos normales más cercanos (sin él mismo)."""
    candidatos = normales.drop(index=fecha, errors="ignore")
    distancia = np.abs((candidatos.index - fecha).days)
    vecinos = candidatos.iloc[np.argsort(distancia, kind="stable")[:N_VECINOS]]
    referencia = vecinos["demanda"].mean()
    return domingos.loc[fecha, "demanda"], referencia, vecinos.index


filas = []
for fecha in domingos.index[domingos["festivo_en_domingo"] | domingos["normal"]]:
    demanda, referencia, vecinos = diferencia(fecha)
    filas.append({"fecha": fecha.date(), "tipo": ("festivo en domingo"
                  if domingos.loc[fecha, "festivo_en_domingo"] else "normal"),
                  "navidad": bool(domingos.loc[fecha, "navidad"]),
                  "demanda": demanda, "referencia": referencia,
                  "diferencia_mw": demanda - referencia,
                  "diferencia_pct": 100 * (demanda / referencia - 1),
                  "vecinos": ", ".join(str(d.date()) for d in vecinos)})
tabla = pd.DataFrame(filas)
SALIDA.parent.mkdir(parents=True, exist_ok=True)
tabla.to_csv(SALIDA, index=False)

fest = tabla[tabla["tipo"] == "festivo en domingo"]
norm = tabla[tabla["tipo"] == "normal"]
p10 = np.percentile(norm["diferencia_pct"], 10)
mediana = fest["diferencia_pct"].median()
print(f"Domingos normales 2017-2022: {len(norm)}; festivos en domingo: {len(fest)}\n")
print("=== Fiestas nacionales en domingo frente a sus 4 domingos normales más cercanos ===")
print(fest[["fecha", "navidad", "demanda", "referencia", "diferencia_mw", "diferencia_pct"]]
      .round({"demanda": 0, "referencia": 0, "diferencia_mw": 0, "diferencia_pct": 1})
      .to_string(index=False), "\n")
print("=== Vara de medir: la misma diferencia en los domingos normales (%) ===")
print(norm["diferencia_pct"].describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_string(), "\n")
fuera_navidad = fest.loc[~fest["navidad"], "diferencia_pct"].median()
print(f"Mediana de los 8 festivos: {mediana:.1f} % (fuera de navidad, 5 días: "
      f"{fuera_navidad:.1f} %); percentil 10 de los normales: {p10:.1f} %")
print("Lectura:", "EFECTO APRECIABLE" if mediana < p10 else "DENTRO DE LA VARIABILIDAD HABITUAL")
print(f"Tabla completa en {SALIDA}")
