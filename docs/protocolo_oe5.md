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

## 8. Prueba de tiempos de SARIMAX (criterio fijado el 7 de octubre de 2026, antes de medir)

Concreta la regla del apartado 2.1 ("si SARIMAX no cabe con la ventana completa") y
aplica la regla de las 8 horas del apartado 2.4. Se fijó y se registró en git **antes de
ejecutar ninguna medición**.

1. **Qué se mide.**
   - *t_ajuste*: tiempo de reloj de un ajuste completo de SARIMAX con la ventana completa
     (1/1/2017 hasta el 31/12/2022 12:00, unas 52.600 horas; horas COVID como vacías).
   - *t_origen*: tiempo de una previsión con parámetros fijos (incorporar los datos hasta
     el origen y prever 37 pasos), medido sobre 10 orígenes de 2023 y promediado.
   - La prueba solo mide tiempos: no calcula errores ni se miran previsiones de 2023.
2. **Configuración medida: la más pesada prevista.** Errores ARMA(2,1); términos de
   Fourier sobre la hora local con 10 armónicos diarios (24 h), 10 semanales (168 h; del
   1 al 11 sin el 7, que tiene periodo de 24 h y repetiría el primero diario) y
   4 anuales (8766 h); exógenas comunes del apartado 4 (grados-día de calefacción y de
   refrigeración, `t_ponderada`, `fraccion_festivo`, `puente`, `navidad`) y 6 columnas de
   interacción temperatura × hora. Unas 60 columnas exógenas (método de estimación:
   punto 6, que sustituye a la máxima verosimilitud conjunta). Si la rejilla que se
   registre después resultara más pesada, se vuelve a medir.
3. **Criterio de "cabe" con la ventana completa:**
   20 × (*t_ajuste* + 365 × *t_origen*) × 1,2 ≤ 10 horas
   (20 configuraciones del apartado 3; 10 horas = una noche de cálculo; 1,2 = margen de
   seguridad del 20 % por medir una sola vez).
4. **Si no cabe, en este orden:**
   1. Acortar la ventana **solo de SARIMAX** por años naturales completos: inicio en 2018
      y, si sigue sin caber, en 2019. **Mínimo: inicio en 2019** (cuatro años, unos 3,7
      sin COVID). Supuesto: el tiempo de ajuste crece en proporción al número de horas.
   2. Si con 2019-2022 sigue sin caber: 12 configuraciones para SARIMAX (se declara).
   3. Si tampoco cabe: se detiene el trabajo y se replantea con el autor (por ejemplo,
      estimación en dos etapas). Un ajuste que falla por falta de memoria cuenta como
      "no cabe".
5. **Regla de las 8 horas (apartado 2.4) con la misma medición:**
   *T_final* = 12 × *t_ajuste* × 1,3 + 366 × *t_origen* (1,3: la ventana de 2024 llega a
   unas 70.000 horas, un 30 % más). Si *T_final* > 8 horas, reajuste trimestral para los
   cuatro modelos. Supuesto declarado: el tiempo crece en proporción a las horas.
6. **Método de estimación (modificación del 7 de octubre de 2026, tras la primera
   medición).** Con la máxima verosimilitud conjunta de statsmodels (los 60 coeficientes
   de la regresión y los del ARMA a la vez), el ajuste no converge: agota el límite de
   15.000 evaluaciones de la verosimilitud en unas 195 iteraciones (18 min en el entorno
   del asistente) y, relanzado desde ese punto, la log-verosimilitud aún mejora en 1046
   (de −386.918 a −385.872, 36 min en total, sin converger). Según el punto 4.3 se
   replanteó con el autor, que decidió estimar el **mismo modelo** (regresión con errores
   ARMA) por **mínimos cuadrados generalizados factibles iterados (FGLS)**, el
   procedimiento clásico para este modelo (en statsmodels, `ARIMA(..., gls=True)`, que no
   admite las horas vacías del COVID):
   1. Regresión por mínimos cuadrados ordinarios, sin las horas COVID.
   2. ARMA de los residuos por máxima verosimilitud en espacio de estados (el filtro de
      Kalman salta las horas COVID).
   3. Regresión por mínimos cuadrados generalizados: la demanda y cada columna exógena
      se blanquean con el filtro de Kalman del ARMA estimado (mismas horas saltadas) y se
      reestiman los coeficientes.
   4. Se repiten 2 y 3 hasta que la log-verosimilitud del modelo completo cambie menos de
      0,1 entre dos vueltas (máximo 20 vueltas; si no se alcanza, se informa).
   Previsiones e intervalos analíticos: el SARIMAX de statsmodels con los parámetros
   estimados fijos. En la misma configuración pesada, el FGLS alcanzó una
   log-verosimilitud de −377.659 en unos 2 minutos (estable desde la quinta vuelta),
   más alta que la del ajuste conjunto tras 36 minutos. **La prueba de tiempos del punto 3
   se repite con este método**, con el mismo criterio; *t_ajuste* es el tiempo del FGLS
   completo hasta cumplir la regla de parada.

## 9. Armónicos de Fourier e interacciones de SARIMAX (fijado el 7 de octubre de 2026, antes de medir)

Decisiones 3 y 4 pendientes del archivo de alcance. Es una elección interna al
entrenamiento (apartado 3.2): no gasta las 20 configuraciones evaluadas en 2023.

1. **Datos:** la ventana del ajuste de selección (1/1/2017 a 31/12/2022 12:00, horas
   COVID vacías). 2023 no interviene.
2. **Método:** FGLS iterado (apartado 8.6) con errores **ARMA(2,1) fijos** durante esta
   búsqueda; el orden del ARMA se elige después, entre las configuraciones evaluadas en
   2023.
3. **Exógenas siempre presentes:** grados-día de calefacción y de refrigeración,
   `t_ponderada`, `fraccion_festivo`, `puente` y `navidad`.
4. **Candidatos (rejilla completa, 96 ajustes):** armónicos diarios 4, 6, 8 o 10;
   semanales 4, 6, 8 o 10 (sin los múltiplos de 7); anuales 2, 3 o 4 (mínimo 2 por el
   periodograma); interacciones temperatura × hora: ninguna o 6 columnas (los 3 primeros
   armónicos diarios multiplicados por `t_ponderada`).
5. **Elección:** AIC mínimo, con AIC = −2 × log-verosimilitud + 2 × número de parámetros
   (coeficientes de la regresión con la constante, parámetros del ARMA y varianza). Se
   guardan también la log-verosimilitud, el BIC y el tiempo de cada candidato, pero no
   deciden.
6. **Regla de borde:** si el ganador cae en el máximo de una dimensión (10 diarios,
   10 semanales o 4 anuales), esa dimensión se amplía una sola vez hasta 12, 12 o 6 y se
   repite la búsqueda solo en ella, con las demás fijas en el ganador. Si vuelve a caer en
   el borde, se acepta y se declara. (El armónico diario 12 aporta una sola columna: su
   seno es siempre cero.)
7. **Limitaciones declaradas:** la elección depende del ARMA(2,1) fijado; con unas 52.000
   horas el AIC tiende a favorecer los modelos grandes.

## 10. Rejilla de SARIMAX evaluada en 2023 (fijada el 7 de octubre de 2026, antes de lanzarla)

1. **Fijo** (apartado 9, commit `ba24f06`): 12 armónicos diarios, 12 semanales y 3 anuales,
   interacciones temperatura × hora y exógenas comunes (65 columnas más la constante);
   estimación por FGLS iterado (apartado 8.6).
2. **Varía:** el orden del ARMA de los errores, ARMA(*p*, *q*) con *p* = 1, 2, 3, 4, 5 y
   *q* = 0, 1, 2: **15 configuraciones**. Configuración de referencia: ARMA(2,1), la usada
   para elegir los armónicos.
3. **Excluido:** términos ARMA estacionales (a 24 o 168 horas). En una prueba en el entorno
   del asistente, un FGLS con un AR estacional de orden 1 a 24 horas no terminó en
   28 minutos (el filtro de Kalman pasa de unos 3 a unos 27 estados). Se declara como
   limitación computacional; la persistencia la recogen la parte autorregresiva y las
   exógenas.
4. **Evaluación** (apartado 2.3): cada configuración se ajusta una vez con los datos hasta
   el 31/12/2022 12:00 y prevé los 365 orígenes de 2023 actualizando los datos con
   parámetros fijos. Decide el **menor MAE** sobre las 8.760 horas; se guardan RMSE, MAPE,
   MASE y tiempos.
5. **Prueba de fuga** (`validacion.comprobar_fuga`) con la configuración de referencia en
   los 365 orígenes, antes de evaluar las 15. Si detecta fuga, se detiene todo.
6. **Fallos:** una configuración que no converge (20 vueltas del FGLS) o da error se
   registra como fallida; no se sustituye por otra.
7. **Diagnóstico de la ganadora (OE3), informativo, no decide:** Ljung-Box sobre los
   residuos estandarizados del ajuste 2017-2022 en los retardos 24 y 168, y ADF y KPSS
   sobre esos residuos.
8. **Enmienda del 7 de octubre de 2026, tras evaluar la configuración de referencia:
   variable de nivel.**
   - *Hecho:* ARMA(2,1) sin variable de nivel, evaluada en 2023 tras superar la prueba de
     fuga (0 días con fuga): **MAE 1.888,2 MW**, frente a 1.400 MW del ingenuo estacional.
     Sesgo medio de −1.660 MW (sobreestima la demanda todos los meses); descontado el
     sesgo, el MAE sería de 1.381 MW. Causa: el nivel del modelo queda fijado en el de
     2017-2022 (no hay variable de tendencia, apartado 5.4), la parte ARMA lo olvida en
     pocas horas y la demanda media de 2023 (26.174 MW) es inferior. Esta configuración
     cuenta como **1 de las 20 del presupuesto** y se informa como resultado.
   - *Descartado:* diferenciar (*d* = 1). Con 2017-2022, ARIMA(2,1,1) estima un MA de
     −0,99 (señal de sobrediferenciación) y el FGLS falla numéricamente.
   - *Decisión del autor:* se añade la exógena **`nivel`**: media de la demanda de las
     168 horas UTC que terminan 37 horas antes de cada hora (de *t* − 204 a *t* − 37, ambas
     incluidas). Solo usa demanda con retardo ≥ 37 h (apartado 1.5), la misma información
     de la que disponen los demás modelos. Las horas cuya ventana incluye alguna hora del
     periodo COVID o anterior al inicio de la serie se excluyen del ajuste (apartado 5.6).
     Es la forma de recoger la tendencia compatible con el apartado 5.4.
   - *Rejilla:* las mismas 15 configuraciones del punto 2, con `nivel` (66 columnas más la
     constante). Presupuesto total: 1 + 15 = 16 de 20. Los armónicos del apartado 9 se
     mantienen aunque se eligieron sin `nivel` (se declara). La prueba de fuga se repite
     con ARMA(2,1) con `nivel` antes de evaluar las 15.
9. **Enmienda del 7 de octubre de 2026, antes de lanzar las 15 configuraciones:
   retardos de 48 y 168 horas.**
   - *Hecho:* ARMA(2,1) con `nivel`, evaluada en 2023 tras superar la prueba de fuga:
     **MAE 1.314,8 MW** (0,939 del ingenuo), sesgo −629 MW. Cuenta en el presupuesto.
   - *Diagnóstico con 2017-2022 (no con 2023):* los residuos estandarizados del ajuste
     tienen autocorrelación 0,69 a 24 h, 0,50 a 48 h y 0,76 a 168 h; el modelo repite
     errores a la misma hora del día y de la semana anteriores, algo que ningún ARMA de
     la rejilla (*p* ≤ 5) puede recoger.
   - *Comprobación con 2017-2022:* añadir como exógenas la demanda en *t* − 48 y en
     *t* − 168 sube la log-verosimilitud de −373.485 a −347.623 y deja la autocorrelación
     de los residuos en 0,38 (24 h), 0,14 (48 h) y −0,04 (168 h).
   - *Decisión del autor:* las 15 configuraciones llevan `nivel` y las exógenas
     `y_48` (demanda en *t* − 48) e `y_168` (demanda en *t* − 168): 68 columnas más la
     constante. Retardos ≥ 37 h (apartado 1.5); las horas en las que falta alguno ya
     quedan excluidas por `nivel`, cuya ventana los contiene. La decisión se justifica
     con el diagnóstico de 2017-2022; se declara que antes se habían visto los MAE de
     2023 de las dos configuraciones de referencia.
   - *Presupuesto:* 2 configuraciones de referencia ya evaluadas (ARMA(2,1) sin y con
     `nivel`) + 15 = 17 de 20. La prueba de fuga se repite con ARMA(2,1) con `nivel` y
     retardos antes de evaluar las 15.
10. **Resultado y decisiones del autor tras la evaluación (7 de octubre de 2026, commit
    `51bcf0c`).**
    - *Ganadora:* **ARMA(5,1) con `nivel` y retardos**, MAE 943,4 MW en 2023 (0,674 del
      ingenuo), RMSE 1.414,4 MW, MAPE 3,67 %, MASE 0,710, sesgo −346 MW. Las seis mejores
      configuraciones quedan entre 943 y 954 MW (menos de un 1,2 % de diferencia): el
      orden del ARMA es un ajuste fino; lo decisivo fueron `nivel` y los retardos (1.888 →
      1.315 → 943 MW). Cinco configuraciones forman un segundo grupo (1.059-1.112 MW, sesgo
      positivo); hipótesis sin comprobar: el FGLS llega a una solución distinta.
    - *Borde:* la ganadora tiene *p* = 5, el máximo de la rejilla, que no tiene regla de
      borde. **No se amplía** (quedan 3 configuraciones del presupuesto y la diferencia
      entre las seis mejores es menor del 1,2 %); se declara como limitación.
    - *Avisos de convergencia interna:* en varias configuraciones de orden alto
      (probablemente también la ganadora), el ajuste del ARMA dentro de alguna vuelta del
      FGLS emitió `ConvergenceWarning`; el FGLS cumplió su regla de parada y la
      configuración es válida según el punto 6. **Se mantiene la ganadora** (cambiarla por
      una más sencilla sería elegir a la vista de los resultados) y se declaran los
      avisos.
    - *Diagnóstico (punto 7), residuos 2017-2022:* Ljung-Box rechaza el ruido blanco a 24
      y 168 h (autocorrelación 0,36 a 24 h, −0,04 a 168 h); ADF rechaza la raíz unitaria
      (−36,6); KPSS rechaza la estacionariedad en nivel (2,9; p < 0,01). Limitación
      declarada: dependencia diaria residual no recogida (el término estacional a 24 h no
      cabe en tiempo de cálculo, punto 3).

## 11. LightGBM (fijado el 13 de octubre de 2026, antes de escribir su código y de evaluar nada en 2023)

### 11.1 Variables

Concreta los apartados 4 y 5.3 (la "lista concreta de retardos" pendiente del apartado 6).

1. **Esquema:** modelo directo, un solo modelo para todas las anticipaciones. Una fila por
   hora objetivo *t*, con su origen (las 12:00 locales del día anterior al día local de
   *t*); cada hora del histórico aparece una sola vez. Objetivo: `demanda_mw` sin
   transformar.
2. **Retardos respecto a *t*, en horas UTC (todos ≥ 37 h, apartado 1.5):** `y_48`, `y_72`,
   `y_96`, `y_120`, `y_144`, `y_168` e `y_336` (demanda en *t* − 48 … *t* − 336), y
   **`nivel`**: media de la demanda de *t* − 204 a *t* − 37, ambas incluidas (misma
   definición que en SARIMAX, apartado 10.8). Justificación con 2017-2022 (OE2): ACF de
   0,83 a 24 h (que solo puede recogerse a partir de 48 h), 0,90 a 168 h y 0,84 a 336 h.
   *Limitación declarada:* al contarse en horas UTC, como en SARIMAX, durante los días
   posteriores a cada cambio de hora los retardos apuntan a una hora local desplazada en
   una (el ingenuo trabaja en hora local). Alternativa descartada: retardos en hora local
   con las reglas del ingenuo (más código y más riesgo de error).
3. **Variables tomadas en el origen** (iguales en todas las filas de un mismo origen):
   - `y_ultima`: demanda de la hora UTC que empieza en origen − 1 h (de 11:00 a 12:00
     locales del día D);
   - `y_media24`: media de las 24 horas UTC de origen − 24 h a origen − 1 h;
   - `anticipacion`: horas entre el origen y el inicio de *t* (de 12 a 36).
4. **Exógenas comunes (apartado 4), codificación de LightGBM:** `gd_calefaccion` y
   `gd_refrigeracion` (`exogenas.grados_dia`), `t_ponderada`, `hora_del_dia` (hora local,
   entero 0-23), `dia_semana` (entero 0-6), `dia_anio` (día del año local, entero 1-366),
   `fraccion_festivo`, `puente` y `navidad`. Todas numéricas, sin tratamiento categórico;
   sin términos de Fourier ni interacciones explícitas (los árboles cortan por la hora y
   combinan variables por sí mismos). `anticipacion` y `hora_del_dia` casi coinciden
   (anticipación = hora + 12, salvo los días de cambio de hora); se mantienen las dos.
5. **Total: 20 variables.** No se dan a LightGBM: las 47 temperaturas provinciales,
   `festivo`, `tipo_dia`, `excluir_covid` (apartado 4); retardos de la temperatura
   (información que SARIMAX no recibe); el retardo de 24 h y la variable "misma hora del
   día más reciente conocida" (24 h para las horas 0-11 y 48 h para las 12-23): no tiene
   fuga, pero rompería la regla de retardos ≥ 37 h del apartado 5.3; valorada y
   descartada por el autor.
6. **COVID estricto (apartado 5.6):** todas las entradas de una fila están entre
   *t* − 336 h y *t* − 12 h, así que se excluye toda fila con alguna hora del periodo
   COVID entre *t* − 336 h y *t*, ambas incluidas (en la práctica, objetivos hasta
   aproximadamente el 14/07/2020). También se excluyen las filas con alguna entrada
   ausente (las primeras 336 horas de 2017); no se rellenan valores.
7. **Filas del ajuste de selección:** las de horas objetivo ya terminadas en el origen del
   ajuste (31/12/2022 12:00 locales): `t + 1 h ≤ origen`.
8. **Cálculo sin fuga:** en `prever(datos, origen)` las variables de demanda se calculan a
   partir de los datos recibidos, no de columnas precalculadas con los datos sin alterar
   (si no, la prueba de fuga saldría limpia aunque la hubiera). Ventanas móviles siempre
   hacia atrás.
9. **Código común:** la definición de `navidad` se extrae a una función de
   `src/exogenas.py` usada por SARIMAX y LightGBM; antes de aceptarlo se comprueba que la
   matriz exógena de SARIMAX es idéntica a la anterior (si no lo es, no se toca).

Pendiente en este apartado, antes de evaluar en 2023: intervalos de LightGBM (capa
probabilística) y espacio de búsqueda (apartado 3), cada uno registrado con commit.

### 11.2 Intervalos propios (fijado el 13 de octubre de 2026, antes de escribir su código)

Decisión del autor entre las dos opciones del archivo de alcance: **regresión cuantílica**,
como fijan el archivo de alcance y el capítulo 2 (OE7). Se descarta la alternativa barata
(cuantiles empíricos del error de 2023): daría intervalos de anchura fija y su calibración
en 2023 se mediría dentro de muestra.

1. **Cuantiles 0,05 y 0,95** (intervalo central del 90 %, nivel nominal de referencia).
   Un modelo por cuantil, con el objetivo `quantile` de LightGBM (pérdida pinball),
   independiente del modelo puntual.
2. **Hiperparámetros:** los de la configuración ganadora de la búsqueda del modelo puntual
   (elegida por MAE, apartado 3.4); solo cambian el objetivo y `alpha` (0,05 o 0,95). No
   se buscan hiperparámetros por cuantil y estos modelos no consumen presupuesto (no se
   elige nada con 2023, apartado 3.2).
3. **Número de árboles:** parada temprana con la pérdida pinball de su cuantil sobre el
   mismo conjunto de control que el modelo puntual (últimas 8 semanas de la ventana de
   entrenamiento, apartado 2.5) y con las mismas reglas de parada.
4. **Previsión puntual:** la del modelo ganador, no la de un modelo de la mediana. No se
   fuerza a quedar dentro del intervalo; se informa en cuántas horas queda fuera.
5. **Cruce de cuantiles:** si en una hora el límite inferior supera al superior, se
   intercambian; se cuenta el número de horas con cruce y se informa.
6. **Igual que el modelo puntual:** variables, filas y exclusión COVID (11.1), semilla 42,
   modo determinista y reajuste mensual en 2024. La prueba de fuga se aplica también a sus
   previsiones.
7. **Coste:** 2 ajustes más en la selección y 2 × 12 en la evaluación final. Sin prueba de
   tiempos propia; se mide y se informa el tiempo.
8. **Capa probabilística (fase 2):** si el CRPS exige una rejilla de cuantiles, se decide
   al diseñarla para los cuatro modelos y antes de medir ninguna calibración; añadir
   cuantiles no cambia la elección del modelo puntual.
9. **Riesgo declarado:** los hiperparámetros se eligen para el centro de la distribución y
   pueden no ser los mejores para las colas; si los intervalos propios salen mal
   calibrados, se informa tal cual (es lo que ACI debe corregir).

Pendiente en el apartado 11, antes de evaluar en 2023: espacio de búsqueda (11.3).

### 11.3 Búsqueda de hiperparámetros (fijado el 13 de octubre de 2026, antes de lanzarla)

Decisiones del autor sobre la propuesta del asistente: reajuste final con toda la ventana,
objetivo L1/L2 dentro de la búsqueda y sin prueba de tiempos formal.

1. **Procedimiento de entrenamiento de cada configuración (fijo, no se busca):**
   1. *Número de árboles por parada temprana* (apartado 2.5): conjunto de control = las
      últimas 8 semanas de la ventana (1.344 horas objetivo antes del 31/12/2022 12:00);
      se entrena con las filas anteriores (47.628) y se para tras **100 árboles sin
      mejora** del MAE en el control (pinball de su cuantil en los modelos del 11.2);
      **máximo 10.000 árboles**. Si una configuración llega al tope (más de 9.900
      árboles), se informa y no se amplía.
   2. *Reajuste* con todas las filas de la ventana (48.972, incluidas esas 8 semanas) y el
      número de árboles elegido. Motivo: son los datos más recientes, que SARIMAX y
      N-HiTS también ven. El número de árboles se eligió con un 3 % menos de filas
      (efecto despreciable, se declara). Alternativa descartada: sin reajuste.
   3. *Fijos:* semilla 42 (`seed`, del que derivan las de muestreo), `deterministic`,
      `force_row_wise`, **4 hilos** (`num_threads`; el determinismo de LightGBM exige el
      mismo número de hilos), profundidad libre, `max_bin` por defecto. En 2024, cada
      reajuste mensual repite este procedimiento con su propia ventana.
2. **Espacio:** 19 configuraciones sorteadas con `ParameterSampler` de scikit-learn
   (`random_state=42`) más la de referencia (apartado 3.1): **20 en total**.

   | Hiperparámetro | Valores |
   |---|---|
   | `objective` | `regression` (L2) o `regression_l1` (L1) |
   | `learning_rate` | log-uniforme entre 0,02 y 0,2 |
   | `num_leaves` | 15, 31, 63, 127 o 255 |
   | `min_data_in_leaf` | 20, 50, 100, 200 o 500 |
   | `feature_fraction` | uniforme entre 0,5 y 1 |
   | `bagging_fraction` | uniforme entre 0,5 y 1 (con `bagging_freq` = 1) |
   | `lambda_l2` | log-uniforme entre 0,001 y 10 |

   Referencia (c00): valores por defecto de LightGBM (L2, tasa 0,1, 31 hojas, 20 filas por
   hoja, sin muestreo ni penalización) con el mismo procedimiento del punto 1. El objetivo
   entra en la búsqueda porque se elige por MAE y el objetivo L1 prevé la mediana,
   coherente con la pinball de la capa probabilística, aunque L2 suele converger mejor.
   El mínimo de la tasa (0,02) y el tope de árboles se fijaron tras medir **solo el
   tiempo** del caso más caro (tasa 0,01 y 255 hojas: unos 4.200-5.000 árboles, casi en el
   tope de 5.000 usado en esa medición); no se calculó ningún error.
3. **Las 20 configuraciones** (generadas antes del commit; el script comprueba que el
   sorteo las reproduce; valores redondeados a 4 decimales):

   | Config. | Objetivo | `learning_rate` | `num_leaves` | `min_data_in_leaf` | `feature_fraction` | `bagging_fraction` | `lambda_l2` |
   |---|---|---|---|---|---|---|---|
   | c00 (referencia) | L2 | 0,1000 | 31 | 20 | 1 | 1 (sin muestreo) | 0 |
   | c01 | L2 | 0,0794 | 63 | 50 | 0,9754 | 0,6873 | 0,8472 |
   | c02 | L1 | 0,1021 | 31 | 500 | 0,9331 | 0,5290 | 0,2538 |
   | c03 | L1 | 0,0305 | 31 | 200 | 0,6062 | 0,9162 | 0,0053 |
   | c04 | L1 | 0,0502 | 127 | 200 | 0,5115 | 0,5035 | 0,1256 |
   | c05 | L2 | 0,0482 | 15 | 200 | 0,5453 | 0,6164 | 0,2975 |
   | c06 | L1 | 0,0206 | 127 | 20 | 0,8402 | 0,9300 | 0,0634 |
   | c07 | L2 | 0,0967 | 127 | 200 | 0,6523 | 0,9042 | 0,0025 |
   | c08 | L1 | 0,0304 | 31 | 200 | 0,5867 | 0,9166 | 0,0367 |
   | c09 | L1 | 0,1865 | 127 | 50 | 0,7734 | 0,7600 | 0,0055 |
   | c10 | L1 | 0,0424 | 15 | 200 | 0,9633 | 0,6976 | 0,8111 |
   | c11 | L1 | 0,1348 | 15 | 50 | 0,6943 | 0,6627 | 0,0122 |
   | c12 | L1 | 0,0293 | 15 | 20 | 0,6380 | 0,8035 | 0,0153 |
   | c13 | L2 | 0,0316 | 63 | 100 | 0,6467 | 0,6974 | 0,0011 |
   | c14 | L1 | 0,1644 | 127 | 20 | 0,9632 | 0,8030 | 0,4021 |
   | c15 | L2 | 0,0927 | 63 | 200 | 0,6854 | 0,5477 | 0,4735 |
   | c16 | L2 | 0,1153 | 63 | 50 | 0,5598 | 0,7361 | 0,7128 |
   | c17 | L2 | 0,0318 | 127 | 20 | 0,5554 | 0,8553 | 0,0572 |
   | c18 | L1 | 0,0515 | 255 | 100 | 0,9538 | 0,7543 | 0,0099 |
   | c19 | L1 | 0,1519 | 255 | 50 | 0,7994 | 0,9714 | 0,6014 |

4. **Coste (sin prueba de tiempos formal):** caso más caro medido en el entorno del
   asistente (2 núcleos) en unos 4 min por ajuste; en el PC del autor, unas 4-5 veces más
   rápido según la prueba de fuga, el peor caso rondaría 2-3 min por configuración
   (parada temprana, reajuste y 365 previsiones) y menos de 1 h las 20, muy por debajo de
   las 10 h del criterio de SARIMAX (apartado 8.3). Se mide y se informa el tiempo de cada
   configuración (apartado 3.6).
5. **Selección:** menor MAE sobre las 8.760 horas de 2023 (apartado 3.4). RMSE, MAPE,
   MASE, sesgo, número de árboles y tiempo se guardan, pero no deciden. Una configuración
   que dé error se registra como fallida y no se sustituye.
6. **Tras elegir la ganadora:** sus modelos de los cuantiles 0,05 y 0,95 (11.2) con el
   procedimiento del punto 1; sus previsiones de 2023 se guardan, con la regla de cruce
   aplicada, para la capa probabilística (fase 2). Aquí solo se informa del número de
   cruces y de horas con la previsión puntual fuera del intervalo; la cobertura y la
   calibración se evalúan en la fase 2, con su diseño fijado antes. Se informa también,
   como descripción, la importancia de cada variable por ganancia (no decide nada).
7. **Prueba de fuga:** la del paso anterior (commit `ffa8803`) cubre todas las
   configuraciones: las variables y `prever` son el mismo código; los hiperparámetros no
   intervienen en qué datos se leen.

### 11.4 Resultado y decisiones del autor (13 de octubre de 2026, tras la búsqueda del commit `fe2c59a`)

1. **Resultado:** las 20 configuraciones válidas (ninguna fallida, ninguna en el tope de
   árboles), entre 0,2 y 0,7 min cada una en el PC del autor. MAE de 2023 entre 585,2 y
   656,3 MW; las siete mejores entre 585 y 596 MW (menos de un 2 %): la configuración es un
   ajuste fino. Lo que más pesa es la tasa de aprendizaje (las cinco mejores, de 0,029 a
   0,032; las tres peores, de 0,15 a 0,19); objetivo L1 o L2 sin diferencia clara.
2. **Ganadora: `c17`** (objetivo L2, `learning_rate` 0,0318, `num_leaves` 127,
   `min_data_in_leaf` 20, `feature_fraction` 0,5554, `bagging_fraction` 0,8553,
   `lambda_l2` 0,0572; 1.203 árboles por parada temprana): **MAE 585,2 MW**, RMSE 790,7 MW,
   MAPE 2,25 %, MASE 0,441, sesgo −142 MW, 0,418 del ingenuo (SARIMAX: 943,4 MW, 0,674).
3. **Anticipación del asistente, fijada antes de lanzar:** MAE de 700 a 1.050 MW (fallida:
   el resultado es mejor); sesgo entre −400 y 0 MW, dispersión menor que en SARIMAX y
   ninguna configuración en el tope (cumplidas).
4. **Reconstrucción antes de aceptar la cifra** (entorno del asistente, mismas versiones):
   `c17` reentrenada desde cero da el mismo MAE y previsiones **idénticas bit a bit** a las
   del PC del autor; prueba de fuga con `c17`: 0 días; **prueba de fuga reforzada**
   (demanda posterior al origen borrada, no desplazada, para detectar también variables
   que combinaran valores futuros de forma que el desplazamiento se anulara): 0 días. El
   error crece con la anticipación (MAE 360 MW de 00 a 05 h del día D+1; 824 MW de 12 a
   17 h), sin rastro de fuga.
5. **Borde:** `min_data_in_leaf` = 20 es el mínimo del espacio. El apartado 11.3 no tiene
   regla de borde y el presupuesto está agotado (20 de 20): **no se amplía** y se declara
   como limitación (como el borde *p* = 5 de SARIMAX, apartado 10.10).
6. **Modelos de los cuantiles 0,05 y 0,95** con los hiperparámetros de `c17` (573 y 601
   árboles): 0 horas con cuantiles cruzados; previsión puntual fuera del intervalo en
   543 de 8.760 horas (6,2 %), consecuencia de modelos independientes (riesgo declarado en
   11.2.9). Intervalos guardados en
   `data/predicciones/lightgbm_intervalos_validacion.parquet`; la cobertura se evalúa en la
   fase 2.
7. **Observaciones para la discusión (no deciden nada; sin comprobar):** sesgo de
   −411 MW de 12 a 17 h (hipótesis: autoconsumo solar creciente en 2023, ausente del
   entrenamiento); agosto, el peor mes; los peores días son festivos (1 de mayo, Viernes
   Santo, Navidad). Importancia por ganancia: `y_168` 58,7 %, `y_336` 11,5 %; la
   temperatura suma un 3 %, lo que no mide su efecto (la demanda retardada ya contiene el
   nivel de temperatura de entonces).
8. **Decisión del autor:** los desgloses por anticipación, mes y día que se citen en la
   memoria saldrán de **un solo script común a los cuatro modelos**, que se escribirá con
   la tabla de validación, cuando N-HiTS esté validado.

## 12. N-HiTS (fijado el 14 de octubre de 2026, antes de escribir su código y de evaluar nada en 2023)

### 12.1 Diseño: horizonte, ventanas, COVID, exógenas y escalado

Concreta los apartados 1.3, 4, 5.1, 5.5 y 5.6 para N-HiTS. Librería: NeuralForecast 3.2.2
(`requirements.txt`). El comportamiento de la librería en que se apoya este apartado se
verificó leyendo su código fuente en esa versión y se comprueba además en el script.

1. **Horizonte:** *h* = 37 pasos UTC desde el origen, todos los días; las 23, 24 o 25
   horas de D+1 se recortan con `validacion.recortar` (apartado 1.3), como en SARIMAX.
   La pérdida de entrenamiento promedia los 37 pasos, incluidas las 12 horas de la tarde
   del día D, que no se evalúan: NeuralForecast no permite pesar de forma distinta las
   posiciones del horizonte. Limitación declarada.
2. **Ventana de entrada:** su longitud *L* es un hiperparámetro, con candidatos **168,
   336, 504 y 672 horas** (una a cuatro semanas); el valor se elige en la búsqueda (12.5).
   Mínimo de 168 h para que la misma hora de la semana anterior (ACF de 0,90 en OE2) esté
   dentro de la ventana para todas las horas de D+1; con *L* ≥ 336 entra también la de dos
   semanas antes. La ventana termina en el origen: su última hora es la de 11:00 a 12:00
   locales del día D (apartado 1.2). N-HiTS usa directamente las últimas horas de demanda
   conocidas, como SARIMAX con el filtro de Kalman y LightGBM con `y_ultima` (apartado
   5.1). La regla de retardos ≥ 37 h (apartado 1.5) se refiere a retardos sueltos; aquí la
   ausencia de fuga la garantiza que la ventana no pase del origen, y se comprueba con la
   prueba de fuga (12.3).
3. **Ventanas de entrenamiento:** todas las que caben en los datos del ajuste, empezando
   en cualquier hora (`step_size` = 1); en cada paso de entrenamiento se sortea un lote de
   ellas (`windows_batch_size`). Los datos del ajuste son las horas con
   `hora_utc + 1 h ≤ origen del ajuste` (31/12/2022 12:00 locales en la selección); 2023
   no entra. Diferencia de procedimiento, no de información, con LightGBM (solo filas con
   origen a las 12:00): se declara. Alternativa descartada: solo ventanas que terminan a las
   12:00 locales (unas 2.000, pocas para una red; además, un salto fijo en horas UTC no
   sigue los cambios de hora).
4. **COVID estricto (apartado 5.6):** columna `available_mask` = 1 − `excluir_covid` y
   `training_data_availability_threshold = [1.0, 1.0]`: la librería descarta toda ventana
   de entrenamiento con alguna hora no disponible en la entrada o en el horizonte (método
   `_create_windows`). Los valores de esas horas no se modifican; nunca se usan.
   Comprobaciones en el script: (a) el número de ventanas válidas contado a mano coincide
   con el de la librería; (b) las ventanas de validación de la parada temprana, que no
   pasan por ese filtro, no tocan el periodo COVID. Alternativa si la librería no se
   comportara así: construir las ventanas a mano.
5. **Exógenas comunes (apartado 4), todas futuras** (`futr_exog_list`: N-HiTS las lee en
   la ventana de entrada y en el horizonte), calculadas con las funciones de
   `src/exogenas.py`:

   | Información | Columnas de N-HiTS | Codificación |
   |---|---|---|
   | Grados-día | `gd_calefaccion`, `gd_refrigeracion` | `exogenas.grados_dia`, sin transformar |
   | Temperatura | `t_ponderada` | Sin transformar (°C) |
   | Hora local | `hora_sen`, `hora_cos` | Seno y coseno de 2π · `hora_del_dia` / 24 |
   | Día de la semana | `dia_semana` | Categórica (0-6), con *embedding* de la librería (`cat_exog_list`) |
   | Posición en el año | `anio_sen`, `anio_cos` | Seno y coseno de 2π · (día del año local − 1) / 365,25 |
   | Calendario | `fraccion_festivo`, `puente`, `navidad` | Numéricas 0-1 (`navidad` con `exogenas.navidad`) |

   Total: 11 columnas (10 numéricas y una categórica). Sin exógenas históricas (la demanda
   pasada es la propia ventana de entrada y la temperatura pasada ya está en las futuras)
   ni estáticas (una sola serie). No se dan a N-HiTS las mismas variables excluidas en el
   apartado 4. El vocabulario de la variable categórica (los valores 0-6) lo construye la
   librería al ajustar; es una tabla de categorías, no un escalado.
6. **Escalado (apartado 5.5):**
   - **Demanda: normalización por ventana** con `scaler_type = "robust"` (mediana y
     desviación absoluta mediana de la parte de entrada de cada ventana; el horizonte no
     interviene: la librería anula su máscara antes de calcular). `robust` y no `standard`
     porque un festivo dentro de la ventana distorsiona menos la mediana que la media.
   - **`local_scaler_type = None`**: en NeuralForecast ese parámetro es el escalado
     ajustado por serie sobre todos los datos del ajuste, el que excluye el apartado 5.5
     (con los datos cortados en el origen no sería fuga estricta, pero el protocolo lo
     excluye para N-HiTS).
   - **Exógenas sin escalar**, mediante una subclase mínima de `NHITS` que sustituye el
     método interno `_get_temporal_exogenous_cols` (su único uso en la librería es elegir
     las columnas que pasan por la normalización por ventana) para que solo la demanda se
     normalice. Motivo: la librería normaliza por ventana también las exógenas numéricas,
     lo que convierte las lentas (posición en el año) en una rampa casi igual en todas las
     ventanas y amplifica los grados-día cuando la ventana de entrada es templada y el
     horizonte no. Sin escalar no se ajusta nada, así que no puede haber fuga. Riesgo:
     método interno de la librería (versiones fijadas); el script comprueba que la demanda
     llega normalizada y las exógenas no. Alternativa no elegida: hora, día de la semana,
     `puente` y `navidad` como categóricas, mes categórico en lugar del día del año y
     temperatura y grados-día escalados por ventana.
7. **Comprobaciones contra el escalado global (apartado 5.5), en el script:**
   (a) `local_scaler_type` es `None` y, tras el ajuste, no hay escaladores globales
   (`scalers_` vacío); (b) el normalizador del modelo es `robust`, por ventana; (c) los
   datos del ajuste terminan en el origen del ajuste; (d) prueba funcional: con el modelo
   ya ajustado, multiplicar por 2 la demanda de 2017-2021 no cambia ninguna previsión de
   los 365 orígenes de 2023 (idénticas bit a bit).
8. **Interfaz común `prever(datos, origen)`:** toma las últimas *L* horas con
   `hora_utc + 1 h ≤ origen` (no lee demanda posterior al origen), construye las exógenas
   de las 37 horas siguientes, prevé y devuelve una serie de 37 valores indexada por hora
   UTC.
9. **Reproducibilidad:** `random_seed` = 42, 4 hilos de torch (`torch.set_num_threads`,
   como los 4 de LightGBM), entrenamiento determinista en CPU, sin registros ni puntos de
   control en disco. La coincidencia bit a bit solo se espera en el mismo sistema; las
   cifras de referencia son las del PC del autor.
10. **Pendiente en este apartado, cada uno con commit antes de medir o lanzar:**
    intervalos (12.2), prueba de fuga (12.3), prueba de tiempos (12.4) y procedimiento de
    entrenamiento y espacio de búsqueda (12.5). Aviso para 12.5: la parada temprana de la
    librería conserva los pesos del último paso, no los del mejor; para imitar el
    procedimiento de LightGBM (parada temprana con las últimas 8 semanas y reajuste con
    toda la ventana) habrá que leer el mejor paso y reentrenar.
11. **Corrección del punto 1 (14 de octubre de 2026, después del commit `258a19a` y antes
    de escribir ningún código).**
    - *Qué estaba mal:* el punto 1 afirma que NeuralForecast no permite pesar de forma
      distinta las posiciones del horizonte. Es falso: todas las funciones de pérdida de la
      versión 3.2.2 (las puntuales, como `MAE`, y las de cuantiles, como `MQLoss`) admiten
      `horizon_weight`, un peso por cada uno de los *h* pasos que multiplica la pérdida de
      ese paso (comprobado en el código de `losses/pytorch.py`: `_compute_weights`). El
      error es del asistente, que no había leído ese archivo al redactar el punto 1.
    - *Decisión del autor:* **pesos 0 en los pasos 1 a 12 y 1 en los pasos 13 a 37.** El
      paso *k* empieza *k* − 1 horas después del inicio del horizonte, así que la red se
      entrena solo en anticipaciones de 12 a 36 horas, las que se evalúan (las de LightGBM,
      apartado 11.1.3). Como las ventanas de entrenamiento empiezan en cualquier hora
      (punto 3), cada ventana enseña a prever entre 12 y 36 horas por delante. Los mismos
      pesos se aplican a la pérdida de control de la parada temprana, si se usa (12.5), y a
      la pérdida de los cuantiles (12.2).
    - *Consecuencias:* desaparece la limitación declarada en el punto 1 (la pérdida ya no
      incluye la tarde del día D). El paso 37 (anticipación de 36 h) solo se evalúa el día de
      25 horas, pero se entrena con peso 1 en todas las ventanas. La red sigue produciendo
      los 12 primeros pasos (por la interpolación de N-HiTS), sin supervisión; no se evalúan.
    - *Riesgo declarado:* se pierde la supervisión de los pasos más fáciles, que podría
      ayudar a fijar el nivel; no se comparan ambas opciones (gastaría presupuesto de la
      búsqueda). Alternativa no elegida: los 37 pasos con el mismo peso.
    - *Comprobación en el script:* el vector de pesos tiene 37 elementos, los 12 primeros
      valen 0 y el resto 1.
12. **Enmienda (14 de octubre de 2026, tras probar el código de 12.3 en el entorno del
    asistente y antes de ejecutarlo en el PC del autor): exógenas de horas que no están en
    los datos.**
    - *Hecho:* N-HiTS prevé siempre 37 pasos y necesita las exógenas de los 37 (la red lee
      las del horizonte completo). Para el último origen de cada conjunto, el paso 37 cae
      fuera de los datos: el 30/12/2023 apunta al 1/1/2024 a las 00:00 locales (la carga de
      validación corta 2024, apartado 2.7) y el 30/12/2024, al 1/1/2025 (fuera del
      dataset). SARIMAX no tiene el problema porque solo prevé hasta el final de D+1.
    - *Decisión del autor:* las horas del horizonte que no están en los datos toman las
      exógenas de la última hora disponible (`nhits_modelo.exogenas_futuras`). El código
      comprueba que son siempre horas posteriores al día objetivo (nunca evaluadas). Afecta
      a un origen por conjunto y a un solo paso.
    - *Riesgo declarado:* como la red lee las exógenas de todo el horizonte, las horas
      evaluadas de ese origen pueden variar mínimamente respecto a usar el valor real.
    - *Alternativa descartada:* leer el calendario y la temperatura de esa hora del año
      siguiente: rompe la salvaguarda del apartado 2.7 en validación y es imposible en la
      evaluación final (no hay datos de 2025).

### 12.2 Intervalos propios (fijado el 14 de octubre de 2026, antes de escribir su código)

Concreta para N-HiTS la decisión de OE7 (archivo de alcance, capítulo 2): regresión
cuantílica con pérdida pinball. Decisión del autor sobre la propuesta del asistente: una sola
red para la previsión puntual y los cuantiles. Comportamiento de la librería comprobado en
el código de NeuralForecast 3.2.2 (`losses/pytorch.py`, `common/_base_model.py`).

1. **Una sola red con pérdida multicuantil** (`MQLoss(quantiles=[0.05, 0.5, 0.95])`): la
   red produce tres trayectorias de 37 pasos y se entrena con la media de las tres
   pérdidas pinball, con los pesos del horizonte del punto 12.1.11 (0 en los pasos 1-12,
   1 en los 13-37).
2. **Previsión puntual = salida del cuantil 0,5 (mediana).** Es la que se evalúa con el
   MAE y la que decide la búsqueda (apartado 3.4); prever la mediana es coherente con
   ese criterio (la pinball en 0,5 es la mitad del error absoluto).
3. **Intervalo propio = salidas de los cuantiles 0,05 y 0,95** (intervalo central del 90 %,
   nivel nominal de referencia; mismos cuantiles que LightGBM, apartado 11.2.1).
4. **Cuantiles fijados antes de la búsqueda.** A diferencia de LightGBM, la previsión
   puntual y los cuantiles comparten red: cambiar el conjunto de cuantiles cambiaría la
   previsión puntual. Por eso este conjunto queda fijo para la búsqueda (12.5), el
   reentrenamiento con 3 semillas (apartado 3.5) y la evaluación final.
5. **Cruce de cuantiles:** `MQLoss` no impone el orden de los cuantiles. Misma regla que
   LightGBM (11.2.5): si en una hora el límite inferior supera al superior, se
   intercambian y se cuenta el número de horas con cruce. **La previsión puntual no se
   modifica nunca.** Se informa también del número de horas con la mediana fuera del
   intervalo.
6. **Escala de la pérdida (se declara):** la pérdida de entrenamiento se calcula en la
   escala normalizada de cada ventana (12.1.6), así que las ventanas con más variabilidad
   pesan algo menos por cada MW de error. No es fuga. La pérdida de control de la parada
   temprana, en cambio, la calcula la librería en MW (deshace la normalización).
7. **Parada temprana (se fija en 12.5):** la pérdida de control de la librería sería la
   multicuantil, no el MAE de la mediana. No vale `valid_loss=MAE()`: con tres salidas,
   la librería promediaría el error absoluto de las tres trayectorias, no el de la
   mediana.
8. **Igual que la previsión puntual:** misma red, mismas ventanas, exclusión COVID y
   exógenas (12.1); la prueba de fuga (12.3) se aplica a las tres salidas.
9. **Coste:** solo se triplica la capa de salida; ningún entrenamiento adicional (LightGBM
   necesitó dos modelos más). Se mide en la prueba de tiempos (12.4).
10. **Capa probabilística (fase 2):** si el CRPS exige una rejilla de cuantiles
    (pendiente 11.2.8), a N-HiTS no se le añaden cuantiles a esta red (cambiaría su
    previsión puntual): se entrenaría una segunda red con la rejilla y los
    hiperparámetros de la ganadora, usada solo para el CRPS, como los modelos de cuantiles
    de LightGBM (un entrenamiento más en 2023 y 12 en la evaluación final). Se decide al
    diseñar la fase 2, antes de medir ninguna calibración.
11. **Riesgos declarados:** (a) compartir la red con las colas podría empeorar algo la
    mediana (pesa un tercio de la pérdida); no se compara con una red puntual aparte
    (gastaría presupuesto); (b) si los intervalos propios salen mal calibrados, se informa
    tal cual (es lo que ACI debe corregir). Alternativas no elegidas: red puntual con `MAE`
    más red de cuantiles con los hiperparámetros de la ganadora (doble coste por
    entrenamiento); rejilla de 19 cuantiles fijada ya (adelantaría la decisión de la
    fase 2 y la mediana pesaría 1/19 de la pérdida).

### 12.3 Prueba de fuga y comprobaciones del escalado (fijado el 14 de octubre de 2026, antes de escribir su código)

Concreta para N-HiTS los apartados 5.5 y 10.5 y los puntos 12.1.4, 12.1.7 y 12.2.8.
Decisiones del autor sobre la propuesta del asistente: criterios de la tabla del punto 3 y
versión reforzada de la prueba de fuga incorporada a `src/validacion.py`.

1. **Archivos:** `src/nhits_modelo.py` (módulo sin número: subclase de N-HiTS con las
   exógenas sin escalar, tabla de entrada de NeuralForecast, ajuste con sus
   comprobaciones y función `prever`); `src/21_nhits_fuga.py` (esta prueba); y en
   `src/validacion.py`, un parámetro `modo` en `comprobar_fuga`: `"desplazar"` (valor por
   defecto, el comportamiento actual: +10.000 MW a la demanda posterior al origen) o
   `"borrar"` (versión reforzada: la demanda posterior al origen se deja vacía). El valor
   por defecto no cambia nada para los scripts existentes: se comprueba que la salida de
   `python src/validacion.py` es idéntica a la anterior.
2. **Modelo de referencia (no consume presupuesto: no se calcula ningún error de 2023):**
   arquitectura por defecto de N-HiTS en NeuralForecast 3.2.2, *L* = 336, 100 pasos de
   entrenamiento, semilla 42, con todo lo fijado en 12.1 y 12.2 (pérdida multicuantil con
   pesos del horizonte, escalado `robust` por ventana, exógenas sin escalar, COVID
   estricto). Ajuste con los datos hasta el 31/12/2022 12:00. Como los 100 árboles de
   LightGBM en su prueba de fuga: no tiene que prever bien, sino reaccionar a sus
   entradas.
3. **Comprobaciones y criterios (anticipación del asistente):**

   | # | Comprobación | Criterio |
   |---|---|---|
   | a | Pesos del horizonte | 37 valores: 12 ceros y 25 unos |
   | b | `local_scaler_type` es `None`, sin escaladores globales tras el ajuste y escalado del modelo `robust` | Se cumple |
   | c | Datos del ajuste | Última hora: de 11:00 a 12:00 del 31/12/2022 (local) |
   | d | Ventanas de entrenamiento válidas contadas a mano frente a las de la librería | Iguales |
   | e | Ventanas de validación de la parada temprana | Ninguna toca el periodo COVID |
   | f | Prueba de fuga (`modo="desplazar"`) en las tres salidas, 365 orígenes de 2023 | 0 días con fuga |
   | g | Prueba de fuga reforzada (`modo="borrar"`) en las tres salidas | 0 días con fuga |
   | h | Control positivo: ventana de entrada que termina 24 h después del origen | Fuga los 365 días |
   | i | Demanda de 2017-2021 multiplicada por 2 (modelo ya ajustado) | Previsiones de los 365 orígenes de 2023 idénticas bit a bit |
   | j | +1.000 MW a toda la ventana de entrada, 10 orígenes | Las tres salidas suben 1.000 MW (tolerancia ±1 MW, precisión de 32 bits de la red) |
   | k | +5 °C a `t_ponderada` en entrada y horizonte, 10 orígenes | Las previsiones cambian |

   Qué demuestra cada grupo: (f)-(h), que la previsión no usa demanda posterior al origen
   y que la prueba detecta una fuga real en este montaje; (i), que al prever no interviene
   ninguna estadística de la serie completa; (j), que la demanda se normaliza por ventana
   (desplazar la entrada desplaza la salida en lo mismo); (k), que las exógenas no se
   normalizan (si se normalizaran por ventana, sumar una constante no cambiaría nada).
4. **Regla:** si falla cualquiera de las comprobaciones, se detiene todo y se revisa con
   el autor antes de seguir; no se cambia el criterio a la vista del resultado.
   Excepciones previstas: si (j) queda fuera de ±1 MW, se informa del desvío antes de
   decidir (no sería una fuga, sino redondeo); si (d) no puede hacerse con la librería, se
   sustituye por una comprobación más débil (entrenar con las horas COVID vacías y
   comprobar que no da error) y se declara.
5. **Pruebas en el entorno del asistente:** con el `dataset.parquet` del autor (huella
   comprobada al cargarlo); solo se comparan previsiones entre sí, no se calcula ningún
   error de 2023. Se mide el tiempo para avisar al autor antes de lanzarlo. Las cifras de
   referencia son las del PC del autor (torch 2.14.1+cpu; el asistente usa la compilación
   de PyPI de la misma versión).
6. **Registro:** el código y su resultado se versionan juntos tras ejecutarlo en el PC del
   autor; el resultado se anota en este apartado.
7. **Resultado (14 de octubre de 2026, PC del autor, 12 núcleos lógicos, 4 hilos; tras la
   enmienda 12.1.12, commit `31c371e`):** todas las comprobaciones cumplen el criterio, sin
   ninguna excepción del punto 4.
   - (a) 37 pesos: 12 ceros y 25 unos. (b) Sin escalado de la serie completa; demanda
     normalizada por ventana (`robust`); exógenas sin escalar. (c) 52.571 horas de ajuste;
     la última, de 11:00 a 12:00 del 31/12/2022.
   - (d) Ventanas de 373 h (336 + 37): 52.199 posibles, 48.900 válidas contadas a mano y
     48.900 según la librería; 3.299 descartadas por el COVID (las 2.927 horas del periodo
     más las 372 posiciones adicionales en que una ventana de 373 h lo solapa).
   - (e) Las horas que pueden usar las ventanas de control (desde el 08/10/2022 13:00) no
     tocan el periodo COVID.
   - (f) 0 días con fuga; (g) reforzada: 0 días; (h) control positivo: fuga los 365 días.
   - (i) Demanda de 2017-2021 multiplicada por 2: 0 de 365 orígenes con alguna diferencia.
   - (j) +1.000 MW en la ventana: las tres salidas suben 1.000 MW con un desvío máximo de
     0,0039 MW (redondeo de 32 bits). (k) +5 °C en `t_ponderada`: cambio máximo de 921 MW
     (1.066 MW en el entorno del asistente: compilaciones distintas de torch y una red
     apenas entrenada; el criterio solo exige que cambie).
   - `python src/validacion.py` da la misma salida que antes del parámetro `modo`.
   - Tiempos (solo informativos; la prueba de tiempos es la del apartado 12.4): ajuste del
     modelo de referencia con 100 pasos, 28 s; cada prueba de fuga (730 previsiones),
     unos 19 s.
   - **Anticipación del asistente cumplida** en las once comprobaciones.

### 12.4 Prueba de tiempos en CPU (fijado el 14 de octubre de 2026, antes de medir en el PC del autor)

Concreta para N-HiTS el riesgo 1 del plan de trabajo (tiempo de cálculo en CPU), con el
mismo esquema que SARIMAX (apartado 8). Decisiones del autor sobre la propuesta del
asistente: configuración medida, fórmulas, límites y orden de recortes de este apartado.

1. **Qué cuesta una configuración** (procedimiento que se fijará en 12.5, como LightGBM en
   11.3): entrenamiento con parada temprana de hasta *P* pasos (control: últimas 8
   semanas), reentrenamiento con toda la ventana con el número de pasos elegido (peor caso:
   *P*) y 365 previsiones de 2023. Lo que manda es el tiempo de un paso de entrenamiento,
   *t_paso*; el de una previsión, *t_prever*, es pequeño (unos 0,026 s en N3).
2. **Qué se mide** (`src/22_nhits_tiempos.py`), en el PC del autor con 4 hilos y semilla 42:
   - Configuración **más pesada prevista** para el espacio de 12.5: *L* = 672,
     `mlp_units` 3 × [1.024, 1.024], `n_blocks` [2, 2, 2], 1.024 ventanas por lote, pérdida
     multicuantil y 11 exógenas (unos 70 millones de parámetros). Si el espacio que se
     registre en 12.5 resultara más pesado, se vuelve a medir.
   - *t_paso* = tiempo de reloj de un entrenamiento de 200 pasos con la ventana completa
     hasta el 31/12/2022 12:00, con el control de 8 semanas evaluado cada 100 pasos, dividido
     entre 200 (incluye arranque y control: estimación conservadora).
   - *t_prever* = media de 10 previsiones de 2023, **solo tiempo**: no se calcula ni se mira
     ningún error.
3. **Criterio de "cabe"**, con *P* = tope de pasos, provisionalmente **1.000** (valor por
   defecto de la librería; se fija en 12.5):
   - *T_conf* = 2 × *P* × *t_paso* + 365 × *t_prever* (peor caso de una configuración).
   - Búsqueda y semillas: 22 × *T_conf* × 1,2 ≤ **10 horas** (20 configuraciones más 2
     reentrenamientos de la ganadora con otras semillas, apartado 3.5; 1,2 = margen de
     seguridad, como en 8.3).
   - Evaluación final de 2024: 12 × 2 × *P* × *t_paso* × 1,3 + 366 × *t_prever* ≤
     **8 horas** (regla de 2.4; 1,3 por el crecimiento de la serie, que la librería recorre
     entera en cada paso).
4. **Si no cabe, en este orden** (se aplica el primer paso que haga cumplir las dos
   condiciones):
   1. Bajar *P* en múltiplos de 100, como mínimo hasta 500.
   2. Bajar los máximos del espacio: *L* = 504, `mlp_units` 3 × [512, 512], `n_blocks`
      [1, 1, 1] (1.024 ventanas por lote), y volver a medir con el mismo criterio y el
      mismo paso 1. Para no repetir la ejecución, el script mide esta configuración
      reducida a continuación si la más pesada no cabe ni con *P* = 500.
   3. Pasar a 12 configuraciones (apartado 3.1), solo por decisión del autor.
5. **Anticipación del asistente.** Fijada antes de escribir el script (por razonamiento a
   partir de N3): *t_paso* entre 0,5 y 1,5 s; la búsqueda cabe con *P* = 1.000 solo si
   *t_paso* ≤ 0,68 s. Actualizada tras probar el script en el entorno del asistente (solo
   tiempos, 2 núcleos; esos tiempos no cuentan para el criterio): 3,6 s por paso con la
   configuración más pesada y 0,9 s con la reducida; con la proporción observada en N3
   (PC del autor unas 2,2 veces más rápido), se espera en el PC del autor unos **1,6 s**
   (fuera del intervalo anticipado: **no cabría ni con *P* = 500**) y unos **0,4 s** con la
   reducida (**cabría con *P* = 1.000**, búsqueda de unas 6 h).
6. **Riesgos:** el peor caso supone que la parada temprana llega siempre al tope (puede
   sobreestimar mucho); otros programas abiertos en el PC durante la medición la
   falsearían. Alternativa más barata no elegida: fijar los máximos bajos sin medir.
7. **Resultado (14 de octubre de 2026, PC del autor, 12 núcleos lógicos, 4 hilos; commit
   del criterio `0de601b`):**

   | Configuración | Parámetros | 200 pasos | *t_paso* | *t_prever* | Búsqueda + semillas (*P* = 1.000) | Evaluación final | Veredicto |
   |---|---|---|---|---|---|---|---|
   | Más pesada prevista (*L* 672, 3 × [1.024, 1.024], [2, 2, 2]) | 69.769.579 | 423,2 s | 2,116 s | 0,034 s | 31,13 h | 18,34 h | No cabe ni con *P* = 500 |
   | Reducida, paso 2 (*L* 504, 3 × [512, 512], [1, 1, 1]) | 12.073.171 | 98,0 s | 0,490 s | 0,030 s | **7,27 h** | **4,25 h** | **Cabe con *P* = 1.000** |

   - **Consecuencia (por el criterio del punto 4, sin decisión a la vista de resultados):**
     los máximos del espacio de búsqueda (12.5) son los de la configuración reducida
     (*L* ≤ 504, `mlp_units` hasta 3 × [512, 512], `n_blocks` [1, 1, 1], 1.024 ventanas por
     lote) y el tope de pasos es *P* = 1.000. El candidato *L* = 672 del punto 12.1.2 queda
     fuera del espacio; se declara como limitación computacional (como la ventana y el ARMA
     estacional de SARIMAX).
   - **Anticipación del asistente:** la primera (0,5-1,5 s, antes de escribir el script)
     falló para la configuración más pesada (2,1 s). La actualizada tras la prueba del
     script acertó el veredicto de las dos configuraciones (la pesada no cabe; la reducida
     cabe con *P* = 1.000) pero se quedó corta en las cifras (1,6 s frente a 2,1 s y 0,4 s
     frente a 0,49 s): el PC del autor fue unas 1,7 veces más rápido que el entorno del
     asistente con redes grandes, no 2,2 veces como en N3, donde el arranque pesaba más.

### 12.5 Búsqueda de hiperparámetros (fijado el 14 de octubre de 2026, antes de lanzarla)

Decisiones del autor sobre la propuesta del asistente: procedimiento, espacio, referencia y
semillas de este apartado. Máximos del espacio y tope de pasos fijados por la prueba de
tiempos (12.4.7). Script: `src/23_nhits_validacion.py`; procedimiento en
`nhits_modelo.entrenar`.

1. **Procedimiento de entrenamiento de cada configuración (fijo, no se busca):**
   1. *Parada temprana* (apartado 2.5): control = últimas 8 semanas de la ventana (1.344
      horas antes del 31/12/2022 12:00); la red se entrena con lo anterior, evalúa el
      control cada **100 pasos** y para tras **3 evaluaciones seguidas sin mejora**;
      máximo ***P* = 1.000 pasos**. La librería conserva los pesos del último paso, así que
      el **mejor paso** se lee de su historial de evaluaciones. Si es el tope, se informa y
      no se amplía.
   2. *Reentrenamiento* con todas las horas de la ventana (incluidas esas 8 semanas)
      durante el mejor paso, con la misma semilla. Motivo: los datos más recientes, como
      LightGBM (11.3.1). Alternativa descartada: sin reentrenamiento (mitad de coste).
   3. *Métrica de control:* la pérdida multicuantil en MW con los pesos del horizonte
      (12.2.7), no el MAE de la mediana; se declara.
   4. *Se declara además:* la librería reparte 3 bajadas de la tasa de aprendizaje a lo
      largo de los pasos de cada entrenamiento; en el reentrenamiento, más corto, las
      bajadas llegan antes.
   5. *Fijos:* semilla 42, 4 hilos, escalado `robust` por ventana, tres pilas con un bloque
      cada una (12.4.7), un lote de series de una sola serie, todo lo de 12.1 y 12.2.
2. **Espacio:** 19 configuraciones sorteadas con `ParameterSampler` de scikit-learn
   (`random_state=42`) más la de referencia: **20 en total**.

   | Hiperparámetro | Valores |
   |---|---|
   | Ventana de entrada *L* (`input_size`) | 168, 336 o 504 |
   | `learning_rate` | log-uniforme entre 0,0001 y 0,003 |
   | Anchura de las capas (`mlp_units` = 3 × [*w*, *w*]) | *w* = 128, 256 o 512 |
   | Ventanas por lote (`windows_batch_size`) | 256, 512 o 1.024 |
   | Escalas por pila (`n_pool_kernel_size` / `n_freq_downsample`) | A: [2, 2, 1] / [4, 2, 1] (por defecto); B: [4, 4, 1] / [12, 4, 1]; C: [8, 4, 1] / [24, 12, 1] |
   | `dropout_prob_theta` | 0 o 0,1 |

   Referencia (c00): arquitectura por defecto de la librería, la de la prueba de fuga
   (12.3.2): *L* = 336, tasa 0,001, anchura 512, 1.024 ventanas por lote, escalas A, sin
   *dropout*.
3. **Las 20 configuraciones** (generadas antes del commit; el script comprueba que el
   sorteo las reproduce; tasas redondeadas a 5 decimales):

   | Config. | *L* | `learning_rate` | Anchura | Ventanas por lote | Escalas | `dropout_prob_theta` |
   |---|---|---|---|---|---|---|
   | c00 (referencia) | 336 | 0,00100 | 512 | 1.024 | A | 0 |
   | c01 | 504 | 0,00121 | 512 | 256 | A | 0,1 |
   | c02 | 504 | 0,00014 | 128 | 1.024 | B | 0 |
   | c03 | 336 | 0,00012 | 128 | 512 | C | 0,1 |
   | c04 | 168 | 0,00028 | 256 | 512 | A | 0,1 |
   | c05 | 168 | 0,00060 | 128 | 1.024 | A | 0,1 |
   | c06 | 336 | 0,00014 | 256 | 512 | C | 0,1 |
   | c07 | 168 | 0,00012 | 512 | 1.024 | C | 0,1 |
   | c08 | 504 | 0,00010 | 512 | 256 | A | 0 |
   | c09 | 168 | 0,00011 | 256 | 512 | B | 0,1 |
   | c10 | 168 | 0,00170 | 512 | 1.024 | C | 0,1 |
   | c11 | 336 | 0,00042 | 256 | 512 | B | 0 |
   | c12 | 168 | 0,00271 | 256 | 512 | B | 0,1 |
   | c13 | 336 | 0,00119 | 256 | 1.024 | B | 0,1 |
   | c14 | 336 | 0,00030 | 256 | 256 | A | 0 |
   | c15 | 336 | 0,00267 | 256 | 256 | A | 0 |
   | c16 | 168 | 0,00013 | 128 | 1.024 | C | 0 |
   | c17 | 504 | 0,00160 | 128 | 256 | A | 0,1 |
   | c18 | 504 | 0,00013 | 512 | 1.024 | A | 0 |
   | c19 | 168 | 0,00083 | 256 | 512 | C | 0 |

   *Observación hecha al generar el sorteo, antes de medir nada:* 8 de las 19 tasas
   sorteadas son menores que 0,0002; con 1.000 pasos como máximo, se espera que lleguen al
   tope y queden poco entrenadas. Es una propiedad del espacio aprobado y no se cambia.
4. **Selección:** menor MAE de la mediana sobre las 8.760 horas de 2023 (apartado 3.4).
   RMSE, MAPE, MASE, sesgo, MAE relativo al ingenuo, mejor paso, pérdida de control,
   número de horas con cuantiles cruzados y tiempo se guardan, pero no deciden. Una
   configuración que dé error (por ejemplo, divergencia con tasas altas) se registra como
   fallida y no se sustituye.
5. **Ganadora:** previsión en el formato común (`data/predicciones/nhits_validacion.parquet`)
   e intervalos 0,05-0,95 con la regla de cruce (`nhits_intervalos_validacion.parquet`);
   se informa de las horas con cruce y con la mediana fuera del intervalo. **Prueba de fuga
   reforzada** con la ganadora; si detecta fuga, se detiene todo.
6. **Ruido entre semillas (apartado 3.5):** la ganadora se reentrena con las semillas 43 y
   44 con el mismo procedimiento. Se informa del MAE de cada semilla, el rango y la
   desviación típica. **La previsión oficial es la de la semilla 42** (elegir la mejor
   semilla sería elegir con 2023). Si el rango supera la distancia entre la ganadora y la
   segunda, se declara que el orden entre ellas está dentro del ruido.
7. **Robustez:** cada configuración se guarda al terminar
   (`data/predicciones/nhits_cXX_validacion.parquet` y `data/interim/nhits_busqueda.parquet`);
   al relanzar, no se repiten. Probado en el entorno del asistente con un arnés que
   sustituye la demanda de 2023 por la de 364 días antes (ninguna cifra mostrada dice nada
   de 2023), 3 configuraciones y 30 pasos.
8. **Anticipación del asistente (antes de lanzar):** MAE de la ganadora entre **620 y
   800 MW**, probablemente peor que LightGBM (585,2 MW) y muy por debajo del SARIMAX
   (943,4 MW); si saliera mejor que LightGBM, se reconstruye antes de aceptarlo. Rango
   entre semillas de 10 a 30 MW. Varias configuraciones en el tope de pasos, sobre todo las
   de tasa baja. Tiempo real de 3 a 5 horas (techo de 7,3 h según 12.4.7).
9. **Riesgos:** divergencia con tasas altas (configuración fallida); muchas
   configuraciones en el tope (no se amplía *P*: no cabe en el presupuesto de tiempo; se
   declara); la métrica de control multicuantil puede elegir un paso algo peor para la
   mediana (efecto pequeño, declarado).

### 12.6 Resultado y decisiones del autor (14 de octubre de 2026, tras la búsqueda del commit `d66e589`)

1. **Resultado:** las 20 configuraciones válidas (ninguna fallida), entre 1,8 y 13,6 min cada
   una en el PC del autor (82 min en total; con las semillas, unos 95 min). MAE de 2023 de la
   mediana entre 716,4 y 936,6 MW; las ocho mejores entre 716 y 767 MW. **12 de 20 llegan al
   tope de 1.000 pasos.** Ninguna configuración tiene cuantiles cruzados. **Todas tienen
   sesgo positivo** (entre +91 y +562 MW: N-HiTS se queda corto), al contrario que SARIMAX y
   LightGBM; sin explicación comprobada (se revisará en el script común de desgloses).
2. **Ganadora: `c13`** (*L* = 336, `learning_rate` 0,00119, anchura 256, 1.024 ventanas por
   lote, escalas B, `dropout_prob_theta` 0,1; 1.000 pasos, en el tope): **MAE 716,4 MW**,
   RMSE 926,5 MW, MAPE 2,73 %, MASE 0,539, sesgo +188 MW, 0,512 del ingenuo (SARIMAX
   943,4 MW, 0,674; LightGBM 585,2 MW, 0,418). Razón RMSE/MAE 1,29, la menor de los cuatro
   modelos. **Resultado:** N-HiTS mejora al SARIMAX un 24 % y queda un 22 % por detrás de
   LightGBM.
3. **Intervalos de la ganadora:** 0 horas con cuantiles cruzados; 0 horas con la mediana fuera
   del intervalo 0,05-0,95 (LightGBM: 543). Guardados en
   `data/predicciones/nhits_intervalos_validacion.parquet`; la cobertura se evalúa en la
   fase 2. **Prueba de fuga reforzada con la ganadora: 0 días.**
4. **Ruido entre semillas (12.5.6):** MAE 716,4 MW (semilla 42, 1.000 pasos), 771,1 MW (43,
   600 pasos) y 704,6 MW (44, 900 pasos): rango 66,4 MW, desviación típica 35,5 MW, media
   730,7 MW. La distancia entre la ganadora y la segunda (`c07`, 723,4 MW) es de 7,0 MW,
   menor que el rango: **el orden entre las mejores configuraciones está dentro del ruido**
   y se declara. La media de las tres semillas se informa como referencia del rendimiento
   esperado de la configuración; no sustituye a la previsión oficial.
5. **Anticipación del asistente (12.5.8):** cumplida en el MAE (620-800 MW) y en la posición
   respecto a LightGBM y SARIMAX; **fallida** en el ruido entre semillas (esperaba 10-30 MW;
   fue 66,4 MW), en la observación de las tasas bajas (`c07`, con 0,00012, queda segunda y
   `c18`, con 0,00013, quinta; `c08`, `c16`, `c03`, `c02` y `c09` sí quedan entre las peores)
   y en el tiempo (esperaba 3-5 h; fueron 1,6 h). Acertó el sentido de "varias en el tope",
   pero fueron 12 de 20.
6. **Decisiones del autor:**
   - **Borde:** la ganadora llega al tope de *P* = 1.000 pasos. **No se amplía**
     (presupuesto de 20 configuraciones agotado; tope fijado por el criterio de tiempos del
     12.4.7) y se declara como limitación, como el borde *p* = 5 de SARIMAX (10.10) y
     `min_data_in_leaf` = 20 de LightGBM (11.4.5). Se declara también que la búsqueda real
     tardó 1,6 h frente al techo de 7,3 h del criterio (peor caso), así que en la práctica
     habría cabido un tope mayor; ampliarlo ahora sería decidir a la vista de 2023.
   - **Se mantiene la ganadora `c13` con la semilla 42 como previsión oficial**
     (`data/predicciones/nhits_validacion.parquet`), aunque el orden entre las mejores esté
     dentro del ruido: cambiarla por otra configuración o por la media de semillas sería
     elegir con 2023.
7. **Para la evaluación final (2024):** 12 reentrenamientos mensuales de `c13` con el
   procedimiento de 12.5.1 (semilla 42); según 12.4.7, por debajo de 4,3 h.
