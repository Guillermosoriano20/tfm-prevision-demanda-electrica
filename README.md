# Previsión a corto plazo de la demanda eléctrica peninsular

Código y resultados del Trabajo Fin de Máster del Máster Universitario en Inteligencia
Artificial de la Universidad Alfonso X el Sabio (UAX), de Guillermo Soriano Pérez
(director: Ángel Manuel Rayo Acevedo, noviembre de 2026).

El trabajo compara, para la previsión del día siguiente de la demanda horaria peninsular
española (2017-2024), cuatro modelos: un **ingenuo estacional**, un **SARIMAX** con términos de
Fourier para la estacionalidad múltiple, **LightGBM** y la red **N-HiTS**. Evalúa la precisión
puntual y la calibración de los intervalos de predicción (propios y con inferencia conforme
adaptativa, ACI). 2023 se usa para elegir las configuraciones y 2024, una sola vez, para evaluar.

La memoria está en `memoria/` (LaTeX) y el protocolo, con las decisiones fijadas antes de
medir, en `docs/protocolo_oe5.md`.

## Entorno

- Python 3.13 y las versiones fijadas en `requirements.txt` (entorno completo).
  pandas debe ser la 2.3.3, que es la que exige NeuralForecast.
- Todo se ejecuta en CPU (sin tarjeta gráfica). Se desarrolló en Windows.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Los scripts se lanzan desde la raíz del repositorio (las rutas son relativas a ella), por
ejemplo `python src/13_construir_dataset.py`.

## Datos

Los datos de entrada **no se publican**: se regeneran con los scripts.

- **Demanda:** API de ESIOS (indicador 1293). Hace falta un token personal, que se guarda en un
  archivo `.env` en la raíz (excluido de git) con esta forma: `ESIOS_TOKEN=...`.
- **Temperatura:** reanálisis ERA5-Land a través de Open-Meteo, por provincia, ponderada por
  población (tabla 2852 del INE).
- **Calendario:** festivos y variables de calendario construidos por los scripts `11` y `12` (contrastados con la librería `holidays`).

La reutilización de los datos de ESIOS se rige por las condiciones de ESIOS; este repositorio no
los redistribuye.

Las carpetas `data/raw`, `data/interim`, `data/processed` y `data/predicciones` no están en git
y se regeneran. `data/resultados/` sí está versionada: contiene las tablas y figuras de la
memoria, con los CSV de cada análisis.

## Scripts (`src/`)

Se ejecutan en orden numérico. Los de prueba y verificación son opcionales para reproducir.

| Scripts | Contenido |
|---|---|
| `00`-`04` | Comprobación del entorno y de la API de ESIOS, descarga y verificación de la demanda |
| `05`-`10` | Población del INE, temperatura de Open-Meteo, temperatura ponderada por población y sus verificaciones |
| `11`-`13` | Verificación de festivos, calendario y construcción del conjunto de datos (`data/processed/dataset.parquet`) |
| `14` | Análisis exploratorio (OE2) |
| `15`-`18` | Ingenuo estacional y SARIMAX: tiempos, búsqueda de armónicos y validación en 2023 |
| `19`-`20` | LightGBM: prueba de fuga y validación en 2023 |
| `21`-`23` | N-HiTS: prueba de fuga, tiempos y validación en 2023 |
| `24`-`25` | Tabla de validación y desgloses; efecto de las fiestas en domingo |
| `26`-`27` | Cuantiles y capa probabilística en 2023 (`26` es reanudable: omite los modelos cuyo archivo ya existe) |
| `28` | Previsión final de 2024 con reajuste mensual de los cuatro modelos (unas 4,3 h en CPU) |
| `29` | Evaluación final de 2024: métricas, Diebold-Mariano, intervalos y ACI |
| `30` | Desglose de OE8 por hora, tipo de día y régimen |
| `31` | Figura de cobertura de 2024 |

Los módulos sin número se importan, no se ejecutan: `validacion.py` (orígenes, prueba de
fuga y métricas), `exogenas.py`, `sarimax_fgls.py`, `lightgbm_modelo.py`, `nhits_modelo.py`,
`nhits_rejilla.py`, `probabilistica.py` y `finales.py`.

### Garantías de la evaluación final

- **Candado y manifiesto (`29` y `30`).** Se niegan a ejecutarse si ya existe su carpeta de
  resultados, salvo con `--repetir-por-error "motivo"`, que queda anotado en el manifiesto
  (`manifiesto.json`). La evaluación de 2024 se ejecutó una sola vez.
- **Arnés de pruebas (`TFM_ARNES`).** Variable de entorno que sustituye la demanda de 2024 por
  datos inventados para probar el código sin leer la demanda real. Solo sirve para pruebas y no
  debe activarse para producir resultados.
- **Semillas.** 42 en todas las búsquedas y en el *bootstrap*; 43 y 44 para medir el ruido entre
  semillas de N-HiTS.

Las salidas `data/resultados/salida_29.txt` y `salida_30.txt` se guardaron desde la consola de
Windows y las tildes aparecen alteradas; las cifras son correctas.

## Memoria

```powershell
cd memoria
latexmk -pdf main.tex
```

(o `pdflatex` y `biber` desde TeXstudio). La bibliografía se gestiona con `biber`.

## Licencia y cita

Licencia MIT (archivo `LICENSE`). Si usas este trabajo, cita:

Soriano Pérez, G. (2026). *Previsión a corto plazo de la demanda eléctrica peninsular:
comparación de modelos clásicos y de aprendizaje automático con evaluación de intervalos de
predicción* [Trabajo Fin de Máster]. Universidad Alfonso X el Sabio.
