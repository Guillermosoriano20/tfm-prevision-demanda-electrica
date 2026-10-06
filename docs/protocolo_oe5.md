# Protocolo de comparación equitativa (OE5)

**Fijado el 6 de octubre de 2026, antes de entrenar ningún modelo y antes de ver ningún
resultado de validación o de prueba.** Este archivo se versiona en git: el commit que lo
añade es la prueba de que el protocolo es anterior a los resultados. Cualquier cambio
posterior se hace con un commit propio y se explica en la memoria.

Modelos: ingenuo estacional, SARIMAX, LightGBM y N-HiTS (más la capa ACI, que se fija
aparte con la capa probabilística).

---

## 1. Orígenes y horizonte

1. **Origen:** las 12:00, hora peninsular, de cada día D. La previsión es para el día
   local D+1. Los orígenes se convierten a UTC con la zona horaria (10:00 UTC en verano,
   11:00 UTC en invierno), nunca con un desfase fijo.
2. **Información disponible:** una fila del dataset puede usarse en la previsión si
   `hora_utc + 1 h ≤ origen_utc` (su hora ya ha terminado). La última hora conocida es la
   de 11:00 a 12:00 locales del día D. Simplificación declarada: el cierre de ofertas del
   mercado diario es a las 12:00, de modo que en la práctica la previsión se emite algo
   antes.
3. **Horas evaluadas:** las del día local D+1 (24; 23 el día del cambio de hora de marzo;
   25 el de octubre). Los modelos de trayectoria (SARIMAX, N-HiTS) predicen 37 pasos UTC
   desde el origen y se recortan a D+1; las 12 horas de la tarde del día D no se evalúan.
4. **Orígenes:**

   | Conjunto | Primer origen | Último origen | Orígenes | Horas evaluadas |
   |---|---|---|---|---|
   | Validación (2023) | 31/12/2022 12:00 | 30/12/2023 12:00 | 365 | 8.760 |
   | Prueba (2024) | 31/12/2023 12:00 | 30/12/2024 12:00 | 366 | 8.784 |

5. **Retardos seguros: ≥ 37 h, contados en horas UTC.** Corrige la regla anterior
   (≥ 36 h): el día de 25 horas la última hora objetivo empieza 36 h después del origen
   (ejemplo: origen 26/10/2024 10:00 UTC, última hora 27/10/2024 22:00 UTC), y un retardo
   de 36 h apuntaría a la hora de 12:00 a 13:00 del día D, aún no observada. Afecta al
   29/10/2023 y al 27/10/2024. Además, la infraestructura comprueba la regla general del
   punto 2 en todas las previsiones.

| Día D+1 | Horas | Anticipación (inicio de hora) | Retardo mínimo seguro |
|---|---|---|---|
| Normal | 24 | 12-35 h | 36 h |
| Cambio de marzo | 23 | 12-34 h | 35 h |
| Cambio de octubre | 25 | 12-36 h | 37 h |

## 2. Reajuste y uso de 2023 y 2024

1. **Ventana de entrenamiento creciente y común:** desde el 1/1/2017 hasta el origen del
   ajuste. Si SARIMAX no cabe con la ventana completa (prueba de tiempos), se le acorta
   **solo a él** y se declara como limitación computacional de la familia clásica.
2. **Distinción:** *actualizar los datos* (cada origen, parámetros fijos) frente a
   *reajustar* (reestimar parámetros).
3. **Fase de selección (2023):** cada configuración candidata se ajusta una sola vez con
   los datos hasta el 31/12/2022 12:00 y se evalúa en los 365 orígenes de 2023
   actualizando los datos, sin reajustes.
4. **Fase final (2024):** solo la configuración ganadora de cada familia. Reajuste
   **mensual** (12 reajustes, en los orígenes cuyo D+1 es el día 1 de cada mes); el
   primero con 2017-2023 hasta el 31/12/2023 12:00.
   **Regla fijada ahora:** si la prueba de tiempos indica que la evaluación final de
   SARIMAX superaría 8 horas, se pasa a reajuste **trimestral para los cuatro modelos**.
5. **2023 se usa para:** elegir hiperparámetros; cuantiles empíricos del error del
   ingenuo (intervalos); posiblemente el arranque de ACI (se decide con la capa
   probabilística).
   **2023 no se usa para:** parada temprana (el conjunto de control son las últimas
   8 semanas de la ventana de entrenamiento, siempre anteriores al origen del ajuste), ni
   para los umbrales de grados-día ni la variable `navidad` (fijados con 2017-2022).
6. **2024 se evalúa una sola vez**, con protocolo, código y configuraciones ganadoras en
   un commit (se anota el hash). Después no se cambia nada para mejorar resultados. Un
   error de programación descubierto después se corrige, se repite la evaluación y se
   informa en la memoria (con los dos resultados si difieren).
7. **Salvaguarda técnica:** en la fase de validación, la carga de datos corta el dataset
   al final del 31/12/2023 (hora local). La carga completa solo se activa con un
   parámetro explícito para la evaluación final.

## 3. Presupuesto de búsqueda

1. **N = 20 configuraciones evaluadas en 2023** para SARIMAX, LightGBM y N-HiTS; una de
   ellas es la configuración por defecto o de referencia. El ingenuo no tiene
   hiperparámetros. Si SARIMAX no puede evaluar 20, evalúa menos y se declara; no se
   recorta a los demás. Alternativa barata (solo por decisión del autor): N = 12 para
   todos.
2. **Solo cuenta como presupuesto lo que se evalúa contra 2023.** Las elecciones internas
   al entrenamiento (armónicos de Fourier por AIC con 2017-2022, parada temprana) son
   parte del procedimiento de la familia.
3. **Método:** búsqueda aleatoria con semilla fija para LightGBM y N-HiTS
   (`ParameterSampler` de scikit-learn); rejilla explícita de como máximo 20 combinaciones
   para SARIMAX. **El espacio de búsqueda de cada familia se escribe y se registra en un
   commit antes de lanzar su búsqueda.**
4. **Criterio de selección:** menor MAE sobre las 8.760 horas de 2023. RMSE, MAPE y MASE
   se calculan y guardan, pero no deciden.
5. **Semillas:** semilla 42 para el sorteo y para todos los entrenamientos; LightGBM en
   modo determinista. La configuración ganadora de N-HiTS se reentrena con 3 semillas
   sobre 2023 para medir el ruido entre semillas.
6. Se mide y se informa el tiempo de cada configuración.

## 4. Variables exógenas comunes

Misma información para SARIMAX, LightGBM y N-HiTS; cada familia la codifica a su manera
(por ejemplo, hora del día como Fourier, como entero o como seno y coseno).

| Información | Resolución |
|---|---|
| Grados-día de calefacción (*Tc* = 15,5 °C), por provincia y ponderados | Diaria (día local), repetida en cada hora |
| Grados-día de refrigeración (*Tr* = 19,5 °C), por provincia y ponderados | Diaria, repetida |
| `t_ponderada` | Horaria |
| Hora del día (local) | Horaria |
| Día de la semana (local) | Diaria |
| Posición en el año (día del año local) | Diaria |
| `fraccion_festivo` | Diaria |
| `puente` | Diaria |
| `navidad`: 1 del 23/12 al 6/1, ambos incluidos | Diaria |

- Umbrales *Tc* y *Tr* fijos (2017-2022); no se reestiman al incluir 2023.
- `navidad`: evidencia de 2017-2022 (16 de los 20 laborables con mayor déficit, entre el
  23/12 y el 2/1); límite final fijado por un criterio externo (vacaciones escolares), no
  ajustado a los datos.
- **No se dan a los modelos:** las 47 temperaturas provinciales sueltas, `festivo` y
  `tipo_dia` (solo para OE8), `excluir_covid` (es un filtro, no una variable), precios o
  actividad económica (fuera del alcance).
- Escenario de previsión meteorológica perfecta: la ventaja de los modelos que más
  dependen de la temperatura horaria es real dentro del escenario, pero podría reducirse
  con previsiones reales; se declara en la discusión.
- Interacciones temperatura × hora y número de armónicos anuales: codificación propia de
  SARIMAX, se deciden con 2017-2022 el día de SARIMAX.

## 5. Información de la propia demanda

1. **Toda la demanda conocida en el origen está disponible para todos los modelos**;
   cada familia la usa a su manera: SARIMAX con el filtro de Kalman; N-HiTS con una
   ventana de entrada que termina en el origen (longitud: hiperparámetro); LightGBM con
   las columnas del punto 3.
2. **Ingenuo estacional:** misma hora local, mismo día de la semana, 7 días antes. Reglas:
   el día de 25 h, las dos 02:00 copian la única 02:00 de la semana anterior; si la semana
   anterior es el día de 23 h (sin 02:00), media de 01:00 y 03:00; si es el día de 25 h
   (dos 02:00), media de las dos.
3. **LightGBM (modelo directo, una fila por par origen-hora objetivo):** retardos
   respecto a la hora objetivo solo ≥ 37 h (lista concreta el día de LightGBM) y
   variables tomadas en el origen: demanda de la última hora conocida, media de las
   últimas 24 horas conocidas y anticipación de la hora objetivo (12-36 h).
4. **Sin variable de año ni de tendencia**; el nivel llega a todos los modelos a través de
   los datos recientes. La diferenciación en SARIMAX es diseño de la familia.
5. **Escalados sin fuga:** cualquier normalización se ajusta solo con datos anteriores al
   origen. En N-HiTS, solo escalado por ventana de entrada; comprobar en el código que no
   se usa un escalado ajustado sobre la serie completa.
6. **COVID:** en entrenamiento se excluye toda muestra cuyo objetivo **o cuyas entradas**
   caigan entre el 01/03/2020 y el 30/06/2020 (fecha local). SARIMAX: horas COVID como
   vacías (el filtro de Kalman las salta). N-HiTS: se descartan las ventanas que solapen
   el periodo (compatibilidad de NeuralForecast por verificar).

## 6. Pendiente de fijar más adelante (sin tocar lo anterior)

- Espacios de búsqueda de cada familia (antes de su búsqueda, con commit).
- Lista concreta de retardos de LightGBM.
- Capa probabilística: intervalos propios, ACI (parámetros y arranque), métricas
  probabilísticas.

## 7. Métricas puntuales (fijadas el 6 de octubre de 2026, antes de ver resultados)

1. **Error:** *e* = real − previsto (positivo: el modelo se queda corto).
2. **Promedio sobre todas las horas evaluadas juntas** (8.760 en 2023, 8.784 en 2024),
   no por días.
3. **MAE** = media de |*e*| (MW); **RMSE** = raíz de la media de *e*² (MW);
   **MAPE** = media de |*e*| / real × 100 (%). El MAPE no tiene problemas aquí: el
   mínimo de la serie es 16.117 MW.
4. **MASE** = MAE / escala. **Escala** = media de |*y*ₜ − *y*ₜ₋₁₆₈| (ingenuo semanal, en
   horas UTC) calculada **una sola vez con 2017-2022**, excluyendo los pares con *t* o
   *t* − 168 en el periodo COVID. Constante fija, igual para 2023 y 2024 y para los
   cuatro modelos. El MASE del ingenuo no tiene por qué valer 1.
5. **MAE relativo al ingenuo estacional** (MAE del modelo / MAE del ingenuo en el mismo
   conjunto): columna adicional; el ingenuo vale 1.
6. **Previsiones guardadas** hora a hora en `data/predicciones/<modelo>_<conjunto>.parquet`
   (columnas `hora_utc`, `fecha_objetivo`, `real`, `prevision`, `modelo`), fuera de git,
   para Diebold-Mariano, OE8 y ACI.
