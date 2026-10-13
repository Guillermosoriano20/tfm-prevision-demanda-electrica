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
