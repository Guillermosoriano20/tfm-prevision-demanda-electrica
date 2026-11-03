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

## 13. Tabla de validación y desgloses de los cuatro modelos (fijado el 15 de octubre de 2026, antes de escribir su código y de calcular nada nuevo)

Cumple la decisión del autor del 13/10 (11.4.8): **un solo script común**,
`src/24_validacion_desgloses.py`, con la tabla de validación de 2023 y los desgloses de los
cuatro modelos. Solo lee previsiones ya guardadas: no entrena, no reajusta y no prevé nada.

### 13.1 Alcance y regla de uso

1. **Entradas:** `data/predicciones/ingenuo_validacion.parquet`, `sarimax_validacion.parquet`,
   `lightgbm_validacion.parquet` y `nhits_validacion.parquet` (las cuatro previsiones
   oficiales); además, solo para 13.4.5, `nhits_c13_s43_validacion.parquet` y
   `nhits_c13_s44_validacion.parquet`.
2. **No decide nada.** SARIMAX (10.10), LightGBM (11.4) y N-HiTS (12.6) están cerrados hasta
   la evaluación final: ningún resultado de este apartado cambia una ganadora, una
   variable, un hiperparámetro ni el procedimiento de 2024. Lo que se observe va a la
   memoria (5.2 y 5.7) como descripción o como hipótesis sin comprobar. Si alguna
   observación sugiriera un cambio de modelo, sería una extensión fuera del núcleo, a
   decidir por el autor y declarada como tomada a la vista de 2023.
3. **Solo 2023.** La carga de datos mantiene el corte de 2024 (`cargar_datos()` por
   defecto). Las funciones reciben el conjunto como parámetro para reutilizarlas en el
   desglose de OE8 sobre 2024 (fase 4), pero el script solo se ejecuta sobre `validacion`.
4. **Descriptivo, sin contrastes.** Con 4 modelos y decenas de celdas, alguna diferencia
   saldrá llamativa por azar; las reglas de lectura se fijan en 13.5.

### 13.2 Controles (el script se detiene si falla alguno)

1. Cada archivo tiene 8.760 filas, sin nulos, con las mismas horas UTC y los mismos días
   objetivo. La hora UTC se lleva a la misma precisión antes de cruzar (SARIMAX la guarda
   en nanosegundos; los demás, en microsegundos).
2. La columna `real` es idéntica bit a bit en los cuatro archivos y a `demanda_mw` del
   dataset (`cargar_datos` comprueba la huella).
3. N-HiTS guarda `prevision` en `float32`: se convierte a `float64` al leerla (redondeo de
   unos 0,004 MW a 40.000 MW; se declara).
4. La escala del MASE recalculada (`validacion.escala_mase`) vale 1.328,2 MW (± 0,05).
5. Se reproducen los MAE registrados: ingenuo 1.400 MW (± 0,5, registrado sin decimales),
   SARIMAX 943,4, LightGBM 585,2 y N-HiTS 716,4 MW (± 0,05); semillas 43 y 44 de N-HiTS,
   771,1 y 704,6 MW (± 0,05).

### 13.3 Tabla de validación de 2023 (una fila por modelo)

MAE, RMSE, MAPE y MASE (`validacion.metricas`, apartado 7); **sesgo** = media de
(real − previsto), positivo si el modelo se queda corto; **MAE relativo al ingenuo**;
**razón RMSE/MAE**; **percentil 95 del error absoluto** (`numpy.percentile`, interpolación
lineal por defecto).

### 13.4 Desgloses (MAE y sesgo de cada modelo)

1. **Anticipación** *a* = hora UTC prevista − origen UTC, en horas (origen de
   `validacion.generar_origenes`: 12:00 locales del día anterior al objetivo). Valores de
   12 a 36, con el número de horas de cada valor (12-34: 365; 35: 364; 36: 1, el día de
   25 horas). **Bloques de 6 h:** 12-17, 18-23, 24-29 y 30-36, que fuera de los días de
   cambio de hora son las horas locales 00-05, 06-11, 12-17 y 18-23 del día objetivo.
2. **Mes** del día objetivo (12 filas).
3. **Día:** MAE diario (media de las horas del día local objetivo); los **10 peores días
   de cada modelo** con día de la semana, `tipo_dia`, `fraccion_festivo` y si está en el
   periodo `navidad`; resumen por **día de la semana** y por **`tipo_dia`** (laborable,
   fin de semana, festivo, puente). `tipo_dia` se usa solo como etiqueta para agrupar,
   nunca como entrada de un modelo (como en OE8).
4. **Sesgo cruzado bloque × mes** (4 × 12) para los cuatro modelos.
5. **N-HiTS con las semillas 43 y 44:** sesgo total, por bloque y por mes, junto a la
   semilla 42.
6. **Comparación entre modelos:** correlación de Pearson de los errores horarios (matriz
   4 × 4) y porcentaje de días en que cada modelo tiene el menor MAE diario (si hay
   empate, cuenta para todos los empatados y se informa del número de empates).
7. **Dos figuras para 5.2**, en `memoria/figuras/` con el estilo del script 14 (16 cm de
   ancho, 300 ppp, sin título, coma decimal, rótulos en español): MAE por anticipación
   (una línea por modelo, *a* de 12 a 35; la anticipación 36, con una sola hora, se omite
   y se declara) y MAE por mes.
8. **Salidas:** tablas en CSV en `data/resultados/validacion/` (fuera de git; se regeneran
   con el script) y resumen en pantalla.

### 13.5 Reglas de lectura (fijadas antes de ver los desgloses)

1. Una diferencia de sesgo entre dos grupos de bloques o de meses se lee como
   **apreciable** solo si supera **150 MW**. Es orientativo: el error medio de un mes
   promedia unas 730 horas fuertemente autocorrelacionadas, así que su incertidumbre es
   mucho mayor que la que daría la fórmula para horas independientes. En N-HiTS se compara
   además con la diferencia entre semillas.
2. **Sesgo propio de la configuración de N-HiTS** (no de la semilla): las tres semillas
   tienen el mismo signo de sesgo total y el mismo orden de bloques (de mayor a menor
   sesgo), con a lo sumo un intercambio entre bloques contiguos.
3. **Autoconsumo solar (LightGBM, de 12 a 17 h):** la observación es **compatible** con la
   hipótesis si el sesgo del bloque 24-29 es más negativo en abril-septiembre que en
   noviembre-febrero en más de 150 MW, y **la debilita** si no. Ninguna de las dos
   lecturas la comprueba: no hay datos de autoconsumo en el trabajo, y en verano se
   confunde con la refrigeración. Se informa como hipótesis.

### 13.6 Anticipación del asistente (fijada antes de escribir el código)

Las cifras de LightGBM por franja (13/10, 11.4.7) ya se vieron en el entorno del asistente;
para ellas solo se anticipa que se reproducen.

1. **Controles:** pasan todos.
2. **Tabla:** razón percentil 95 / MAE entre 2,3 y 3,2 en los cuatro modelos (2,5 con
   errores normales; 3,0 con errores de Laplace).
3. **Anticipación:** en SARIMAX, LightGBM y N-HiTS el MAE es máximo en el bloque 24-29
   (12-17 h); el ingenuo es el más plano (cociente entre su bloque peor y su bloque mejor
   menor que el de los otros tres). LightGBM, bloque 24-29: sesgo de −411 ± 15 MW
   (reproducción; difiere solo en los días de cambio de hora).
4. **Sesgo de N-HiTS:** positivo en al menos 9 de los 12 meses; **mayor en los bloques
   diurnos (24-29 y 30-36) que en el de madrugada (12-17)**, con una diferencia de más de
   150 MW, y el bloque de madrugada entre −150 y +150 MW. Hipótesis en que se apoya:
   entrenamiento incompleto (la ganadora y 12 de 20 configuraciones en el tope de pasos)
   que encoge la previsión hacia el nivel central de la ventana de entrada; las horas por
   encima de ese nivel (día) se quedan cortas y las de por debajo (noche), no. Si es así,
   la semilla 43, que paró antes (600 pasos), tendrá el sesgo total más alto de las tres.
   La hipótesis rival (una deriva de nivel aprendida) predice un sesgo parecido en los
   cuatro bloques; si la diferencia entre bloques es menor de 150 MW, la anticipación
   habrá fallado. Confianza baja (en torno al 50 %). Sesgo propio de la configuración
   (13.5.2): sí.
5. **Sesgo de LightGBM de 12 a 17 h:** negativo en al menos 10 de 12 meses y compatible con
   el autoconsumo según 13.5.3 (confianza en torno al 60 %). N-HiTS, cuya entrada son las
   últimas horas observadas, no tiene sesgo negativo en ese bloque; SARIMAX sí lo tiene, y
   más negativo que su sesgo total (−346 MW).
6. **Mes:** el peor mes de cada modelo es agosto o diciembre.
7. **Días:** en cada modelo, al menos 5 de sus 10 peores días son festivos (`tipo_dia`
   festivo) o caen en el periodo `navidad`; el sesgo medio de los festivos es negativo en
   los cuatro (se prevé más demanda de la que hay).
8. **Comparación:** todas las correlaciones de errores son positivas; las más bajas, las del
   ingenuo; LightGBM-SARIMAX, 0,60 (ya visto el 13/10); la pareja más correlacionada,
   LightGBM-N-HiTS, entre 0,55 y 0,80. Días con el menor MAE: LightGBM 45-65 %, N-HiTS
   20-35 %, SARIMAX 10-25 %, ingenuo menos del 8 %.

### 13.7 Comprobación de alineación horaria de N-HiTS (fijada el 15 de octubre de 2026, tras ver los desgloses del commit e30ade8 y antes de ejecutarla)

**Motivo.** Al ejecutar el script 24 (apartados 13.3-13.6), el sesgo de N-HiTS por
anticipación oscila con fuerza a lo largo del día (+849 MW a las 07 h, −734 MW a las 12 h,
+1.275 MW a las 21 h), igual en los 12 meses y en las tres semillas. Esa forma es
compatible con una previsión desplazada en el tiempo. Las pruebas de fuga de 12.3
comprueban que no se usa información futura, no que cada previsión esté en su hora. La
lectura del código (`nhits_modelo.py`) no muestra ningún error: la entrada acaba en la hora
de 11:00 a 12:00 del día D, las exógenas futuras empiezan en el origen y un `assert`
comprueba que las horas devueltas por la librería son las pedidas. Lo que el código no
controla es cómo empareja la librería, por dentro, exógenas y demanda al entrenar.

1. **Desplazamiento.** Para cada modelo, MAE de la previsión de la hora *t* frente a la
   demanda real de *t* + *s*, con *s* = −3 … +3 h, sobre las 8.754 horas comunes a todos
   los desplazamientos. Un modelo bien alineado tiene el mínimo en *s* = 0; uno retrasado,
   en *s* < 0. SARIMAX y LightGBM sirven de referencia. Comprobado en el entorno del
   asistente con previsiones inventadas: una previsión retrasada 1 h a propósito da el
   mínimo en *s* = −1; las alineadas, en *s* = 0.
2. **Perfil diario medio** por hora local de la demanda real y de cada previsión: amplitud
   (máximo − mínimo del perfil), amplitud relativa a la real y horas del máximo y del
   mínimo. Distingue un desfase (perfil desplazado) de un suavizado (perfil aplanado).
3. **Regla de decisión.** *Desalineación probable* si el mínimo de N-HiTS está en *s* ≠ 0,
   mejora al de *s* = 0 en al menos un 3 % y SARIMAX y LightGBM tienen el mínimo en
   *s* = 0. En ese caso se detiene la lectura de los desgloses de N-HiTS, se busca el error
   con una serie inventada de patrón conocido y, si se confirma, corregirlo y rehacer la
   búsqueda de N-HiTS sería la corrección de un error (no una elección a la vista de 2023),
   a decidir por el autor y declarada. *Sin desalineación* (mínimo en *s* = 0): el patrón
   es comportamiento del modelo y se interpreta en los desgloses.
4. **Anticipación del asistente:** sin desalineación (confianza en torno al 70 %); mínimo
   de N-HiTS en *s* = 0 y perfil diario de N-HiTS más plano que el real (amplitud relativa
   menor de 0,95), por suavizado (escalas B: la primera pila dibuja un punto cada 12 h y lo
   interpola; entrenamiento en el tope de pasos).
5. **Salidas:** `alineacion_desplazamiento.csv`, `perfil_diario_medio.csv` y
   `perfil_diario_resumen.csv` en `data/resultados/validacion/`, y resumen en pantalla.
   El resto del script no cambia (tablas y figuras idénticas).

### 13.8 Resultado (15 de octubre de 2026, script 24 con el commit 2bc8d2b)

1. **Controles (13.2):** superados (seis archivos de 8.760 horas, `real` idéntica al dataset,
   escala 1.328,2 MW, seis MAE reproducidos).
2. **Tabla de validación de 2023 (13.3)** (MW salvo MAPE y cocientes):

   | Modelo | MAE | RMSE | MAPE | MASE | Rel. ingenuo | Sesgo | RMSE/MAE | P95 \|e\| |
   |---|---|---|---|---|---|---|---|---|
   | Ingenuo | 1.400,0 | 2.014,9 | 5,32 % | 1,054 | 1,000 | +45,7 | 1,44 | 4.307 |
   | SARIMAX | 943,4 | 1.414,4 | 3,67 % | 0,710 | 0,674 | −345,9 | 1,50 | 2.767 |
   | LightGBM | 585,2 | 790,7 | 2,25 % | 0,441 | 0,418 | −142,2 | 1,35 | 1.617 |
   | N-HiTS | 716,4 | 926,5 | 2,73 % | 0,539 | 0,512 | +188,2 | 1,29 | 1.855 |

3. **Alineación (13.7):** los cuatro modelos tienen el MAE mínimo en *s* = 0 (N-HiTS: 716,5
   en *s* = 0, 1.088,4 en *s* = −1 y 1.164,3 en *s* = +1); **sin desalineación**, el patrón
   de N-HiTS es comportamiento del modelo. Perfil diario medio: amplitud relativa a la real
   0,995 (ingenuo), 1,017 (SARIMAX), 1,011 (LightGBM) y **0,880 (N-HiTS)**, que además
   sitúa el máximo a las 20 h (real: 21 h). N-HiTS dibuja un día aplanado: −850 MW a las
   07 h, +734 MW a las 12 h y −1.280 MW a las 21 h frente a la demanda media real.
   Anticipación de 13.7.4 cumplida; la causa (escalas B y entrenamiento en el tope de
   pasos) sigue siendo una hipótesis no comprobada (N-HiTS está cerrado).
4. **Desgloses (13.4), lo principal:**
   - Con un único origen diario, **anticipación y hora del día coinciden** y no se pueden
     separar: el error sigue la hora (madrugada, la más fácil para todos), no crece de
     forma regular con la anticipación.
   - MAE por bloque (00-05 / 06-11 / 12-17 / 18-23 h): ingenuo 947 / 1.420 / 1.723 / 1.510;
     SARIMAX 518 / 960 / 1.237 / 1.058; LightGBM 360 / 542 / 824 / 615; N-HiTS 472 / 720 /
     750 / 923. N-HiTS es el mejor de 13 a 17 h y el peor de los tres modelos ajustados
     a las 07 h y a las 21 h.
   - Por mes: LightGBM es el mejor en 10 de 12 (N-HiTS en abril y agosto); N-HiTS, el más
     regular (587-852 MW); el ingenuo sufre en agosto (2.399), enero y diciembre.
   - Festivos (10 días): MAE 4.260 (ingenuo), 4.194 (SARIMAX), 1.131 (LightGBM) y 910
     (N-HiTS), sesgo negativo en los cuatro; 8 de los 10 peores días de SARIMAX son
     festivos. Peores días: el ingenuo, los festivos y los días una semana después
     (13/01, 14/04, 8/05, 22/08, 19/10, 8/11); LightGBM, festivos y finales de agosto;
     N-HiTS, el principio y el final del año.
   - N-HiTS: sesgo de +490 MW los lunes (SARIMAX −524).
   - Correlación de errores: ingenuo-SARIMAX 0,70 (la más alta), LightGBM-SARIMAX 0,60,
     ingenuo-LightGBM 0,59, LightGBM-N-HiTS 0,50, SARIMAX-N-HiTS 0,45, ingenuo-N-HiTS 0,27.
     Días con el menor MAE: LightGBM 48,8 %, N-HiTS 22,2 %, SARIMAX 16,7 %, ingenuo 12,3 %.
5. **D2, sesgos (13.5):**
   - **LightGBM de 12 a 17 h:** −411 MW; negativo en 10 de 12 meses; media abril-septiembre
     −507 MW frente a −210 MW en noviembre-febrero (diferencia 297 > 150): **compatible con
     el autoconsumo** (13.5.3), sin comprobar. SARIMAX (−508) y N-HiTS (−313) también se
     pasan en esa franja y el ingenuo no (+53): los tres modelos entrenados con 2017-2022
     sobreestiman el mediodía de 2023.
   - **N-HiTS:** sesgo positivo en los 12 meses; **propio de la configuración** (13.5.2: las
     tres semillas con sesgo positivo, 188 / 182 / 114 MW, y el mismo orden de bloques);
     se explica por el perfil aplanado (punto 3), no por una deriva de nivel (los bloques
     difieren en más de 1.000 MW).
6. **Anticipación del asistente (13.6):** cumplida en los controles, P95/MAE (3,08 / 2,93 /
   2,76 / 2,59), el ingenuo como el más plano, la reproducción de −411 MW, el sesgo de
   N-HiTS en ≥ 9 meses y en la madrugada, el sesgo propio de la configuración, LightGBM y
   SARIMAX de 12 a 17 h, el peor mes, el sesgo de los festivos y los días ganados por los
   tres modelos ajustados. **Fallida** en el bloque de máximo error de N-HiTS (18-23 h, no
   12-17 h), en su sesgo de mediodía (negativo, no mayor que el de madrugada) y en la
   semilla 43 (no tiene el mayor sesgo): el mecanismo no es un encogimiento hacia el nivel
   de la ventana, sino un perfil diario aplanado. **Fallida** también en los peores días
   (solo SARIMAX tiene 5 o más festivos o de Navidad), en las correlaciones (la más alta es
   ingenuo-SARIMAX; LightGBM-N-HiTS, 0,50) y en los días ganados por el ingenuo (12,3 %).
7. **Decisiones:** ninguna (13.1.2). Las ganadoras no cambian; lo observado va a 5.2 y 5.7.
   Tablas en `data/resultados/validacion/`; figuras `validacion_mae_anticipacion.png` y
   `validacion_mae_mes.png` en `memoria/figuras/`.

## 14. Capa probabilística en 2023 (fijado el 31 de octubre de 2026, antes de escribir su código y de calcular nada con 2023)

Cumple lo dejado pendiente en 6, 11.2.8 y 12.2.10: intervalos propios al 90 %, métricas de
calibración (con CRPS), y parámetros y arranque de ACI. **Ninguna cifra de calibración de 2023 se ha
visto.** Lo único que el asistente ha leído de 2023 en esta sesión es el *esquema* (nombres y tipos
de las columnas) de los archivos de previsiones e intervalos ya guardados, no sus valores; las
pruebas de MAPIE se hicieron con datos sintéticos. Del apartado 11.4 se conocía ya que la
previsión puntual de LightGBM queda fuera de su intervalo en 543 de 8.760 horas (6,2 %).

### 14.1 Alcance y reglas de uso

1. **Solo 2023.** La carga de datos mantiene el corte de 2024 (`cargar_datos()` por defecto).
   Las funciones reciben el conjunto como parámetro para reutilizarlas en la evaluación final.
2. **No decide nada sobre los modelos.** Las cuatro previsiones puntuales oficiales (10.10, 11.4,
   12.6 y el ingenuo) no cambian. Una calibración mala se informa tal cual (11.2.9, 12.2.11); no se
   corrige reajustando un modelo. Lo único que se elige con 2023 es el paso γ de ACI (14.6.7), con
   una regla fijada aquí; se declara que su rendimiento en 2023 es optimista.
3. **Lo que ya se sabe que contamina la calibración de 2023** (se dice al interpretar): los
   hiperparámetros de LightGBM y de N-HiTS se eligieron con el MAE de 2023 (la ganadora de N-HiTS
   incluye sus cuantiles 0,05 y 0,95); γ de ACI se elige con 2023. Las cifras que se informan como
   rendimiento son las de 2024.
4. **Descriptivo y sin contrastes de hipótesis;** las diferencias se leen con las reglas de 14.8.

### 14.2 Decisiones y motivos

Decisiones del autor (31/10/2026, tras plantearlas el asistente):

| # | Decisión | Motivo |
|---|---|---|
| D1 | **CRPS con rejilla de 19 cuantiles** (0,05 a 0,95, paso 0,05) para los cuatro modelos | Mantiene OE7 como está escrito (pinball, CRPS, cobertura, anchura) y es el camino previsto en 11.2.8 y 12.2.10. Coste: LightGBM +17 modelos de cuantil; N-HiTS una segunda red (14.4). Riesgos: la segunda red tiene otra mediana que la oficial; la rejilla ignora las colas fuera de 0,025-0,975 (14.5.5) |
| D2 | **ACI con MAPIE, un conjunto de residuos por bloque de anticipación de 6 h** (4 `TimeSeriesRegressor` por modelo) | El error depende mucho de la hora (13.8: MAE de LightGBM 360 MW de 00 a 05 h y 824 MW de 12 a 17 h); un solo conjunto global daría la misma anchura a las 3 y a las 15 h. Consecuencia que se declara: MAPIE construye el intervalo con los residuos pasados alrededor de la previsión puntual y **no reutiliza los intervalos propios**; el intervalo «después de ACI» es una construcción común a los cuatro, no una recalibración del propio |
| D3 | **Cuantiles del ingenuo con los errores de 2017-2022 sin COVID**, no con los de 2023 | El ingenuo no tiene parámetros, así que sus errores de 2017-2022 son legítimos y la cobertura de 2023 sale fuera de muestra. Con los de 2023 sería circular (≈ 90 % por construcción), el mismo argumento con que 11.2 descartó esa vía para LightGBM. **Corrige** el archivo 02 y el protocolo 2.5, que decían 2023 |

Decisiones tomadas por el asistente por defecto (el autor puede cambiarlas antes de que se escriba el
código):

| # | Decisión | Motivo |
|---|---|---|
| D4 | La recursión de ACI se escribe a mano (tres líneas) con el **error del intervalo realmente emitido**; MAPIE calcula los intervalos | Con origen a las 12:00 la salida de cada hora se conoce 12-36 h después de emitirla. `adapt_conformal_inference` de MAPIE recalcula el intervalo con el estado *actual* (puntuaciones y α_t de hoy), no con el emitido, y eso rompe la identidad en que se apoya la garantía de ACI (la suma de (α − error) debe usar los errores de los intervalos emitidos). Control: sin retardo, la recursión propia coincide con la de MAPIE (14.7.2) |
| D5 | Las puntuaciones de cada bloque son una **ventana móvil de W = 336 residuos** (8 semanas × 6 h), reconstruida en cada origen con `fit(..., cv="prefit")` | Evita depender de cómo `update` de MAPIE reemplaza puntuaciones; 8 semanas es el mismo periodo de control usado en 2.5, 11.3 y 12.5 |
| D6 | **Arranque de 2023:** 57 días (1/1 a 26/2) sin emitir intervalos de ACI; primer día emitido, el 27/02/2023; α_t inicial = 0,10 | No hay residuos honestos anteriores a 2023 (los de 2017-2022 son de ajuste); rehacer 2022 con modelos entrenados hasta 2021 costaría entrenamientos nuevos con hiperparámetros elegidos en 2023. Las puntuaciones iniciales son los 336 residuos más recientes de cada bloque observados a las 12:00 del 26/02/2023 |
| D7 | **Arranque de 2024 (fijado ahora):** puntuaciones iniciales = los 336 residuos más recientes de cada bloque de 2023 observados en el primer origen de 2024; α_t inicial = 0,10 (no se arrastra el α_t de 2023); mismo γ, mismo W | Los residuos de 2023 son fuera de muestra para modelos entrenados hasta 2022 (limitación: en 2024 los modelos se reajustan cada mes). No se arrastra α_t porque compensa la deriva de un modelo distinto del que se evaluará |
| D8 | **Intervalos infinitos:** MAPIE los devuelve solo si α_t = 0; se sustituyen por la extensión máxima de la ventana (límite inferior = previsión + mínimo de los residuos; superior = previsión + máximo) y se cuenta el número de horas | El intervalo emitido (y con el que se evalúa y se actualiza ACI) debe ser finito para definir anchura y puntuación de intervalo |
| D9 | Cuantiles del ingenuo **por bloque de anticipación** (los mismos 4 de ACI) | Sin esto el ingenuo tendría anchura constante todo el día y sería un listón artificialmente malo; con bloques tiene las mismas oportunidades que los demás |
| D10 | Incertidumbre de las cifras: **bootstrap de bloques de días** (1.000 réplicas, semilla 42) | Las horas de un día están fuertemente autocorrelacionadas; remuestrear días respeta eso (13.5.1) |
| D11 | N-HiTS de la rejilla en un módulo nuevo (`nhits_rejilla.py`), duplicando unas 35 líneas de `nhits_modelo.py` | No tocar un módulo cerrado y validado (12.3); el costo es duplicar el procedimiento 12.5.1 con otra lista de cuantiles |

### 14.3 Intervalos propios al 90 % (nivel nominal 0,90; cuantiles 0,05 y 0,95)

1. **Ingenuo estacional.** Previsión del ingenuo (15) para todos los días objetivo de 2017-2022
   desde el 8/1/2017 (los siete primeros no tienen semana anterior), **sin los días cuyo objetivo o
   cuya fuente (7 días antes) toquen el periodo COVID** (`excluir_covid`). Error e = real − previsto.
   Para cada bloque de anticipación (14.5.1) y cada nivel τ de la rejilla: cuantil empírico de *e*
   (`numpy.quantile`, interpolación lineal). Intervalo = previsión + cuantil. Para 2023 se usan los
   errores de 2017-2022 (D3); en la evaluación final, los de 2017-2023 (sin COVID).
2. **SARIMAX.** Analítico, gaussiano, con los parámetros FGLS fijos de la ganadora ARMA(5,1): en
   cada origen, `get_forecast(...)` sobre la serie actualizada (la misma que 18) y cuantil τ =
   previsión + *z*_τ · error típico del paso. Reproduce las previsiones oficiales (control 14.7.1).
   Supuestos que se declaran: normalidad, varianza de innovación estimada con 2017-2022, parámetros
   sin incertidumbre.
3. **LightGBM.** Cuantiles 0,05 y 0,95 de `lightgbm_intervalos_validacion.parquet` (11.2, con la
   regla de cruce). Para la rejilla (14.4), 19 modelos de cuantil con los hiperparámetros de `c17`;
   los de 0,05 y 0,95 deben reproducir los guardados (control 14.7.1).
4. **N-HiTS.** Cuantiles 0,05 y 0,95 de `nhits_intervalos_validacion.parquet` (12.2, red oficial
   `c13`, semilla 42). La rejilla sale de una segunda red (14.4); el intervalo del 90 % **no** se
   toma de ella, sino de la red oficial.

### 14.4 Rejilla de cuantiles para el CRPS

1. **Niveles:** τ_k = k/20, k = 1 … 19 (0,05; 0,10; …; 0,95). Columnas `q05`, `q10`, …, `q95`.
2. **Ingenuo y SARIMAX:** sin coste (cuantiles empíricos por bloque y cuantiles gaussianos).
3. **LightGBM:** 19 modelos con `entrenar(X, y, fin, params_c17, alpha=τ)` (11.3: parada temprana
   con la pinball de su cuantil en las últimas 8 semanas, reajuste con toda la ventana), semilla 42,
   4 hilos. La previsión puntual y el intervalo del 90 % siguen siendo los oficiales.
4. **N-HiTS:** una **segunda red** con `MQLoss` de los 19 niveles, los hiperparámetros de `c13`
   (*L* 336, tasa 0,00119, anchura 256, 1.024 ventanas por lote, escalas B, *dropout* 0,1), semilla 42, los
   pesos del horizonte de 12.1.11 y el procedimiento de 12.5.1 (parada temprana con el control de
   8 semanas y reentrenamiento). No cambia nada de la red oficial. Se informa del MAE de su
   mediana y de la cobertura de sus cuantiles 0,05-0,95 frente a los de la red oficial, porque son
   dos sistemas distintos.
5. **Reordenación monótona** (se ordenan los 19 valores de cada hora) y número de horas en que
   hizo falta, por modelo. Es independiente de la regla de cruce del 90 % (11.2.5, 12.2.5).
6. **Prueba de fuga** (modo `borrar`, 13 de `validacion.comprobar_fuga`) sobre todas las salidas
   nuevas: los 17 modelos de LightGBM, la segunda red de N-HiTS (19 salidas) y las previsiones e
   intervalos analíticos del SARIMAX. Debe dar 0 días; se detiene todo si falla.

### 14.5 Métricas (todas en MW, sobre las mismas horas)

1. **Bloques de anticipación** *a* (13.4.1): B1 = 12-17 h, B2 = 18-23, B3 = 24-29, B4 = 30-36.
2. **Conjunto de evaluación E:** los **308 días objetivo del 27/02/2023 al 31/12/2023** (7.392
   horas), que es lo que ACI emite. Las métricas de los intervalos propios se informan en E (para
   compararlos con ACI) y también en las 8.760 horas.
3. **Cobertura** (proporción de horas con inferior ≤ real ≤ superior), global, por bloque y por mes,
   con intervalo de confianza del 95 % por bootstrap de días; además, la proporción de fallos por
   debajo (real < inferior) y por encima (real > superior).
4. **Anchura** media y mediana (superior − inferior) y anchura media relativa a la demanda real.
5. **Puntuación de intervalo** (Gneiting y Raftery, 2007) con α = 0,10: IS = (u − l) +
   (2/α)(l − y)⁺ + (2/α)(y − u)⁺, media sobre las horas. Es igual a 20 veces la suma de las
   pinball de 0,05 y 0,95; se informan también estas dos pinball.
6. **CRPS aproximado por la rejilla:** CRPS ≈ (2/20) · Σ_k ρ_τk(y, q_τk), con ρ_τ(y, q) =
   (y − q)(τ − 1{y < q}). Es la suma de Riemann de CRPS = 2 ∫₀¹ ρ_τ dτ con paso 0,05; **omite las
   colas fuera de 0,025-0,975**, de modo que es una aproximación por defecto, igual para los cuatro
   modelos y comparable entre ellos, no el CRPS exacto. Se llama «CRPS₁₉» en las tablas.
7. **Fiabilidad de la rejilla:** para cada τ_k, frecuencia observada de real ≤ q_τk; tabla y figura
   (diagrama de fiabilidad) por modelo, y desviación máxima |frecuencia − τ|.
8. **Después de ACI:** las mismas métricas con el intervalo de ACI (14.6) y, para el CRPS y la
   fiabilidad, los cuantiles de ACI a los 9 niveles centrales (14.6.8).

### 14.6 ACI con MAPIE 1.5.0 (igual para los cuatro modelos)

1. **Entrada:** previsión puntual oficial *ŷ* y demanda real *y* de cada hora de 2023; residuo
   s = y − ŷ. Para N-HiTS, *ŷ* es la mediana de la red oficial; la segunda red no interviene en ACI.
   Ningún intervalo propio interviene.
2. **Mecanismo de MAPIE:** `TimeSeriesRegressor(estimator=<búsqueda de previsiones guardadas>,
   method="aci", cv="prefit")`, puntuación de conformidad por defecto (residuo con signo, límites
   asimétricos: cuantiles de s con la corrección de muestra finita de MAPIE). Una instancia por
   bloque y por modelo. El «estimador» devuelve la previsión guardada de cada hora; no se entrena
   nada.
3. **Información disponible en cada origen** (regla de 1.2 aplicada a ACI): solo los pares
   (previsión, real) cuya hora ha terminado (`hora_utc + 1 h ≤ origen`). En el origen del día *D*
   están observadas todas las horas del día objetivo *D*−1 (previstas hace dos orígenes) y las
   horas 00:00-11:00 del día *D* (previstas en el origen anterior); **no** el día que se va a prever
   ni las 12 horas siguientes del día *D*. La retroalimentación de cada hora llega 24 h (horas
   00-11 del día objetivo) o 48 h (horas 12-23) después de emitir su intervalo, lo que hace
   aproximada la garantía teórica de ACI (pensada para retroalimentación inmediata); se declara.
4. **En cada origen, para cada bloque:** (a) ventana = los W = 336 pares observados más recientes
   del bloque (D5); (b) `fit` de MAPIE con esa ventana; (c) se fija α_t en `current_alpha`; (d)
   `predict` con nivel 0,90 de las horas del día objetivo de ese bloque; (e) sustitución de
   límites infinitos (D8); (f) el intervalo emitido se guarda.
5. **Actualización (D4):** cuando el par de una hora emitida entra en la información (primer origen
   posterior a su fin), en orden cronológico dentro de su bloque: err = 1 si real ≤ inferior o real ≥
   superior del intervalo emitido (igual criterio estricto que MAPIE), y α_t ← clip(α_t + γ(α₀ −
   err), 0, 1), con α₀ = 0,10. Las horas de los 57 días de arranque no actualizan α_t (no tienen
   intervalo emitido).
6. **Arranque 2023 y 2024:** D6 y D7.
7. **Elección de γ con 2023** (regla fijada ahora). Candidatos G = {0,001; 0,005; 0,01; 0,02};
   γ = 0 (ventana móvil sin adaptación) se calcula siempre como referencia, no como candidato. Para
   cada γ y modelo, IS del 90 % en E; puntuación del candidato = media de los cuatro IS_m(γ)/IS_m(0).
   Gana el γ con menor puntuación; si dos difieren menos de un 0,5 %, el menor. Un solo γ para los
   cuatro modelos. Si ningún candidato queda por debajo de 1, ACI no mejora a la ventana móvil
   sin adaptación en esa métrica: se informa como resultado negativo y se mantiene el γ de menor
   puntuación para la evaluación final. Motivo del criterio: el IS premia a la vez cobertura y
   nitidez (es una puntuación propia), y normalizar por modelo evita que el ingenuo, con IS mucho
   mayores, decida solo.
8. **Cuantiles después de ACI (para CRPS y fiabilidad):** con el γ elegido, ACI se repite en los 9
   niveles centrales 0,10; 0,20; …; 0,90 (α₀ = 0,90; …; 0,10), cada uno con su propio α_t; los
   cuantiles son τ = (1 − c)/2 y (1 + c)/2 con *c* el nivel central; el cuantil 0,5 es *ŷ* más la
   mediana de la ventana de residuos (sin adaptación). Reordenación monótona como en 14.4.5.
9. **Salidas:** `data/predicciones/{modelo}_aci_validacion.parquet` (intervalos de ACI del 90 % para
   cada γ de G ∪ {0} y los 9 niveles con el γ elegido) y tablas CSV en
   `data/resultados/probabilistica/` (fuera de git; se regeneran).

### 14.7 Controles y pruebas (el script se detiene si falla alguno)

1. **Reproducción:** ingenuo (MAE 1.400 ± 0,5 MW); SARIMAX reajustado: previsiones idénticas a
   `sarimax_validacion.parquet` (diferencia máxima < 10⁻⁶ MW; si no, se detiene y se informa:
   no se sustituye en silencio); LightGBM: modelos de 0,05 y 0,95 idénticos a los de
   `lightgbm_intervalos_validacion.parquet` (diferencia 0); `real` idéntica al dataset en todos los
   archivos; 8.760 filas sin nulos.
2. **ACI sin retardo = MAPIE:** con retroalimentación inmediata (datos sintéticos), la trayectoria
   de α_t de la recursión propia coincide con `adapt_conformal_inference` al bit.
3. **Corte de causalidad de ACI:** para varias fechas de corte *c*, se sustituye la demanda real
   posterior a *c* por +10.000 MW y se comprueba que todos los intervalos emitidos con origen ≤ *c*
   son idénticos. **Control positivo:** una variante que viera el día objetivo debe fallar la misma
   prueba.
4. **Sintético:** con errores independientes de escala conocida, ACI con cada γ de G cubre
   0,90 ± 0,02; con un salto de escala a mitad de serie, recupera la cobertura.
5. **Pruebas sin 2023:** el código se prueba en el entorno del asistente con un arnés que sustituye
   la demanda de 2023 por datos inventados o por la de 364 días antes; ninguna cifra impresa en ese
   entorno dice nada de 2023.

### 14.8 Reglas de lectura (fijadas antes de ver los resultados)

1. **Cobertura global del 90 % en E:** *bien calibrada* si está entre 88 y 92 %; *infracobertura*
   por debajo de 88 %; *sobrecobertura* por encima de 92 % (±2 puntos es orientativo: el error de
   muestreo de 308 días autocorrelacionados ronda ese orden). Se da el IC del 95 % por bootstrap y
   si contiene el 90 %.
2. **Cobertura por bloque o mes** se lee como distinta de la global solo si difiere en más de
   5 puntos.
3. **Asimetría:** el cociente fallos por debajo / fallos por encima > 2 o < 0,5 se lee como
   intervalo desplazado (sesgo), no solo estrecho.
4. **IS y CRPS₁₉:** una diferencia entre dos modelos, o entre el intervalo propio y el de ACI de un
   modelo, es *apreciable* solo si el IC del 95 % por bootstrap de días de la diferencia pareada
   excluye 0; en otro caso, *no distinguible*.
5. **ACI mejora la calibración** de un modelo si su desviación |cobertura − 90 %| baja al menos un
   punto o pasa de fuera a dentro de 88-92 %; **mejora el IS** según la regla 4.
6. **Fiabilidad de la rejilla:** desviación máxima ≤ 0,03 *buena*; hasta 0,06 *aceptable*; más,
   *mala*.
7. Más de un 2 % de horas con límite infinito sustituido (D8) en un modelo se informa como defecto
   del método para ese modelo.
8. Nada de esto cambia un modelo (14.1.2). Un resultado negativo (por ejemplo, ACI sin efecto o
   peor que la ventana móvil) se informa tal cual.

### 14.9 Anticipación del asistente (fijada antes de escribir el código)

Cifras entre paréntesis: intervalo de confianza subjetivo.

1. **Controles:** pasan todos (confianza ~85 %).
2. **Cobertura del 90 % propio, 8.760 horas:** ingenuo 89 % (86-92); SARIMAX 87 % (82-91);
   **LightGBM 78 % (68-86)**, la peor (confianza ~70 % de que sea la más baja y de que esté por
   debajo de 85 %: hiperparámetros de un modelo de la media, 6,2 % de previsiones fuera del
   intervalo, ajuste de cuantiles en muestra); N-HiTS 86 % (78-92).
3. **Asimetría:** LightGBM y SARIMAX fallan más por debajo del límite inferior que por encima del
   superior (sesgo −142 y −346 MW; cociente > 2, confianza ~65 %); N-HiTS al revés (sesgo +188 MW,
   confianza ~60 %); ingenuo entre 0,6 y 1,6.
4. **Anchura media (MW):** ingenuo 6.000 (5.200-7.000); SARIMAX 4.600 (3.600-5.800); LightGBM
   2.100 (1.500-2.900); N-HiTS 3.300 (2.200-4.500).
5. **Puntuación de intervalo y CRPS₁₉ (propios):** orden LightGBM < N-HiTS < SARIMAX < ingenuo
   (~70 %). IS en MW, orientativo: LightGBM 4.000-5.000, N-HiTS 4.700-5.500, SARIMAX 6.300-7.500,
   ingenuo 8.500-10.000. CRPS₁₉ ≈ 0,70-0,85 del MAE de cada modelo (LightGBM ~450 MW,
   N-HiTS ~540, SARIMAX ~760, ingenuo ~1.100).
6. **Segunda red de N-HiTS:** MAE de su mediana entre 680 y 820 MW, peor que 716,4 en el ~65 %; su
   cobertura del 90 % a ±3 puntos de la de la red oficial (~60 %).
7. **ACI (γ elegido):** γ = 0,005 o 0,01 (~60 %). Cobertura en E entre 88 y 92 % en los cuatro
   modelos (~75 %). El IS mejora con claridad solo en LightGBM (−5 a −20 %); en SARIMAX, ingenuo y
   N-HiTS, no distinguible (~60 %). La anchura de ACI de LightGBM sube a 2.700-3.500 MW. Horas con
   límite infinito sustituido: menos del 1 % (~80 %).
8. **Fiabilidad de la rejilla propia:** LightGBM y SARIMAX por encima de la diagonal (sobreprevisión:
   el real cae por debajo de los cuantiles más a menudo que τ); N-HiTS por debajo en la parte alta;
   desviación máxima: ingenuo ≤ 0,04, SARIMAX 0,03-0,08, LightGBM 0,06-0,14, N-HiTS 0,03-0,10.
9. **Por bloque:** la cobertura propia de LightGBM es la más baja en B3 (12-17 h) (~60 %).
10. Si una de estas cifras falla, se dice claramente al explicar el resultado.

### 14.10 Riesgos y límites declarados

1. Ingenuo: sus errores de 2017-2022 incluyen años distintos al de 2023 (por ejemplo la bajada de
   2022); si 2023 se parece menos a esos años, la cobertura se resiente.
2. SARIMAX: normalidad, varianza de innovación estimada en muestra y parámetros fijos; las
   colas de la demanda son más pesadas (RMSE/MAE 1,50).
3. LightGBM y N-HiTS: hiperparámetros elegidos con 2023 para la media o la mediana; la calibración de
   2023 es algo optimista (14.1.3). La segunda red de N-HiTS es otro sistema distinto de la oficial.
4. ACI con retroalimentación retrasada (14.6.3): la garantía de cobertura es aproximada.
5. ACI ignora la forma de los intervalos propios y la variación dentro de un bloque (por ejemplo, 12
   frente a 17 h) y entre tipos de día (festivos); sus ventanas de 8 semanas mezclan días
   laborables y festivos.
6. El γ elegido con 2023 hereda el optimismo de elegir con los mismos datos que se informan.
7. La rejilla de 19 niveles no ve las colas fuera de 0,025-0,975 (14.5.6).
8. Coste de la evaluación final de 2024: SARIMAX igual que el puntual; LightGBM +17 modelos × 12
   reajustes (~1 h); N-HiTS +12 reentrenamientos de una segunda red (< 4,3 h según 12.4.7); ACI,
   minutos.

### 14.11 Archivos, orden de ejecución y registro de coherencia

1. `src/probabilistica.py` (módulo: rejilla, bloques, intervalos del ingenuo, métricas, motor de ACI,
   bootstrap), `src/nhits_rejilla.py` (segunda red, D11), `src/26_cuantiles_validacion.py`
   (genera y guarda las cuatro rejillas de 2023, con pruebas de fuga; reanudable) y
   `src/27_capa_probabilistica.py` (métricas, ACI, elección de γ, tablas y figuras).
2. **Orden:** 26 y luego 27. Estimación sin medir (se medirá y se informará): 26, de 30 a 60 minutos
   en el PC del autor (SARIMAX unos 3 min, LightGBM unos 5 min, segunda red de N-HiTS y pruebas
   de fuga el resto); 27, minutos.
3. **Para el registro de coherencia del 05 (la memoria no se toca hasta R2):** CRPS por rejilla y segunda red
   (4.4 y 5.3); diseño de ACI, retraso y D4 (3.7, 4.4); cuantiles del ingenuo con 2017-2022 (el
   capítulo 2 y el 02 dicen «error de validación»); ACI como construcción común, no recalibración del propio
   intervalo (la introducción y OE7 hablan de «recalibración»). Verificar en la fuente la referencia de la
   puntuación de intervalo (Gneiting y Raftery, 2007) antes de citarla en R2.

### 14.12 Implementación (31 de octubre de 2026, antes de ejecutar nada con 2023)

Código: `src/probabilistica.py`, `src/nhits_rejilla.py`, `src/26_cuantiles_validacion.py` y
`src/27_capa_probabilistica.py` (14.11). **Pruebas hechas en el entorno del asistente sin mirar 2023:** la
demanda de 2023 se sustituyó por la de 364 días antes (2022) con ruido inventado del 1,2 % y se eliminó
2024; los hiperparámetros y presupuestos de entrenamiento se redujeron (120 árboles, 60 pasos). Ninguna
cifra de esas pruebas dice nada de 2023. Pasaron: la recursión de ACI idéntica a MAPIE al bit (γ = 0, 0,005
y 0,02); el corte de causalidad en los cuatro modelos con su control positivo (la variante con fuga falla,
como debe); la prueba sintética (cobertura 0,90 ± 0,02, con recuperación tras un salto de escala); las
pruebas de fuga reforzadas de SARIMAX, de los 19 modelos de LightGBM y de la red de 19 salidas (0 días); la
reproducción del SARIMAX (diferencia 0) y de los cuantiles 0,05 y 0,95 de LightGBM (diferencia 0).

Decisiones de implementación y motivos:

| # | Decisión | Motivo |
|---|---|---|
| I1 | La prueba de fuga se pasa **una vez** sobre un predictor de varias salidas (los 19 modelos de LightGBM; la red de 19 salidas; media y error típico del SARIMAX), y sus resultados de los datos reales se reutilizan para construir la rejilla | Mismas garantías que 19 pruebas separadas (todas las salidas se comparan con `equals`); coste de una sola. Sin esto, 19 pruebas de LightGBM costarían del orden de una hora |
| I2 | En ACI se ajusta **una instancia de MAPIE por bloque y origen** y se llama a `predict` una vez por configuración (cada γ o nivel, con su propio α_t) | La ventana de puntuaciones no depende de γ ni del nivel; el resultado es el mismo que con ejecuciones separadas y el coste, mucho menor |
| I3 | Se descarta **un** error del ingenuo de 2017 (hora 00 del 8/1/2017) | El dataset empieza a las 01:00 locales del 1/1/2017: esa hora no existe y el ingenuo de ese día a las 00:00 queda vacío. El código comprueba que no falte más de uno. Lo descubrió la prueba con los datos reales de 2017-2022 (49.320 errores válidos sin COVID: 12.329, 12.330, 12.330 y 12.331 por bloque) |
| I4 | 27 compara **automáticamente** las cifras anticipadas en 14.9 con las observadas (✓ o ✗) y guarda la tabla | Evita que la comparación dependa de la memoria o del ánimo al ver los resultados; se leen los ✗ sin reinterpretar |

Cómo se leen los puntos de 14.9 (fijado ahora, antes de ver cifras reales):

1. Cobertura, asimetría, anchura, IS, razón CRPS₁₉/MAE, desviación máxima de fiabilidad y cobertura por
   bloque de los intervalos **propios**: las 8.760 horas (14.5.2), salvo lo que se diga en contra.
2. «Por encima de la diagonal» = la media de (frecuencia observada − τ) sobre los 19 niveles es positiva.
3. «Al revés» para N-HiTS = cociente fallos por debajo / por encima menor que 0,5.
4. Cobertura de ACI y anchura de ACI: el conjunto E. «El IS de ACI mejora con claridad» = baja entre un 5 y
   un 20 % respecto al IS propio en E y la diferencia pareada es *apreciable* (14.8.4); «no distinguible» =
   la diferencia pareada no lo es.
5. El orden de IS y CRPS₁₉ se compara con el anticipado en las 8.760 horas.
6. Horas infinitas: porcentaje sobre las 7.392 horas de E para el intervalo del 90 % con el γ elegido.

Estimación de tiempo en el PC del autor (sin medir): 26, entre 30 y 90 minutos (los 19 modelos de LightGBM
son lo más incierto: cada uno hace su propia parada temprana); 27, unos 2-5 minutos. Se medirá y se
informará.

### 14.13 Resultados de 2023 (31 de octubre de 2026)

Ejecución en el PC del autor: `26` en unos 19 minutos (ingenuo, menos de 1; SARIMAX, 7,2; LightGBM, 2,6; N-HiTS,
8,9); `27` con sus controles en 17 s. Todos los controles pasaron: entrada (4 archivos de 8.760 filas, sin
nulos, `real` idéntica al dataset); ingenuo con MAE 1.400,0 MW; SARIMAX reajustado idéntico al oficial
(diferencia 0,00); modelos de LightGBM de 0,05 y 0,95 idénticos a los guardados (573 y 601 árboles);
pruebas de fuga reforzadas de SARIMAX, de los 19 modelos de LightGBM y de la red de 19 salidas: 0 días;
recursión de ACI igual a MAPIE al bit; corte de causalidad de ACI en los cuatro modelos con su control
positivo; prueba sintética (0,90 ± 0,02). Cifras de este apartado desde los CSV de
`data/resultados/probabilistica/` (la impresión del script redondeaba a un decimal las filas de cobertura;
corregido sin cambiar ningún cálculo).

**14.13.1 Intervalos propios del 90 %, 8.760 horas.**

| | Ingenuo | SARIMAX | LightGBM | N-HiTS |
|---|---|---|---|---|
| Cobertura (IC 95 %) | 88,3 % (85,6-90,8) | 92,5 % (90,4-94,2) | 70,0 % (67,3-72,5) | 68,1 % (66,3-70,0) |
| Fallos por debajo / por encima | 5,7 % / 5,9 % | 4,9 % / 2,7 % | 18,3 % / 11,7 % | 13,2 % / 18,7 % |
| Cociente fallos | 0,97 | 1,84 | 1,57 | 0,70 |
| Anchura media (MW) | 5.925 | 4.385 | 1.926 | 1.781 |
| IS (MW) | 9.357 | 6.679 | 4.788 | 4.899 |
| CRPS₁₉ (MW) | 1.038 | 707 | 446 | 531 |
| CRPS₁₉ / MAE | 0,741 | 0,750 | 0,762 | 0,742 |
| Desviación máx. de fiabilidad | 0,050 | 0,165 | 0,163 | 0,184 |

En E (14.8.1): ingenuo 88,0 % (bien calibrado, en el borde); SARIMAX 93,0 % (IC 90,9-94,9: sobrecobertura);
LightGBM 70,9 % y N-HiTS 67,3 % (infracobertura, con el IC lejos del 90 %). Orden por IS y por CRPS₁₉:
LightGBM < N-HiTS < SARIMAX < ingenuo. LightGBM frente a N-HiTS: IS no distinguible (−104 MW, IC −607 a +441);
CRPS₁₉ apreciable (−84 MW). Todos los demás pares, apreciables. Con los intervalos propios, CRPS₁₉ es casi
proporcional al MAE (0,74-0,76) y ordena igual que la precisión puntual.

**14.13.2 Lo comprobado sobre el porqué (cobertura por hora local del día objetivo).** (a) La anchura del SARIMAX
es casi plana a lo largo del día (3.898 MW a las 00 h, 4.746 MW a las 23 h) mientras su MAE va de 481 a
1.277 MW: cubre el 99 % de 00 a 05 h y el 87 % de 13 a 17 h. El bloque de anticipación equivale a una franja
horaria (B1 = 00-05 h, B2 = 06-11 h, B3 = 12-17 h, B4 = 18-23 h), así que la cobertura por bloque es, en la
práctica, cobertura por hora del día. (b) N-HiTS tiene sesgos fuertes por hora (+850 MW a las 07 h, −734 a las
12 h, +1.279 a las 21 h) y bandas de 1,5 a 3 veces su MAE: cobertura del 32 % a las 21 h y del 86 % a las 05 h.
(c) LightGBM: la banda sigue la forma del día (2,6 a 5 veces el MAE) pero es demasiado estrecha en todas las
horas; con la rejilla, la demanda cae por debajo del cuantil 0,05 el 16,5 % de las veces y por encima del 0,95
el 10,6 %. **Hipótesis sin comprobar:** que los cuantiles de LightGBM y de N-HiTS se ajustan mejor en entrenamiento
que fuera de muestra; que agosto (cobertura del 68 % en el ingenuo, 56 % en LightGBM, 84 % en SARIMAX) fue un mes
difícil para todos por una causa común; que el ingenuo falla alrededor de festivos al copiar la semana anterior.

**14.13.3 Rejilla.** Reordenación monótona: 0 horas en ingenuo y SARIMAX; 8.760 en LightGBM y 7.737 en N-HiTS
(en casi todas las horas hay algún par de cuantiles cruzados). En LightGBM el efecto es pequeño: los cuantiles
0,05 y 0,95 cambian de media unos 30 MW (en el 20 % y el 17 % de las horas; máximos de 1.086 y 1.322 MW) frente
a una anchura de unos 1.900 MW. Segunda red de N-HiTS: 1.000 pasos (el tope), MAE de su mediana 710,3 MW (la
oficial, 716,4), cobertura 0,05-0,95 del 66,3 % (la oficial, 68,1 %).

**14.13.4 ACI (W = 336, un conjunto por bloque, retroalimentación retrasada).** Elección de γ (14.6.7): puntuación
media de IS(γ)/IS(0) de 1,0049 (γ = 0,001), 1,0251 (0,005), 1,0128 (0,01) y 1,0093 (0,02). **Resultado negativo:**
ningún candidato queda por debajo de 1; ACI no mejora el IS de la ventana móvil sin adaptación. Por la regla se
mantiene γ = 0,001 (para la evaluación final de 2024 también). Horas infinitas sustituidas (D8) con γ = 0,005:
36, 6, 6 y 0 (ingenuo, SARIMAX, LightGBM, N-HiTS); con 0,01: 162, 114, 96 y 18; con 0,02: 300, 288, 216 y 114;
con γ = 0,001: 0 en los cuatro (sobre 7.392 horas). Interpretación coherente con los recuentos, no aislada: cada
fallo resta 0,9γ a α_t y la retroalimentación llega en bloques de 6 horas por día y bloque, de modo que un día
malo puede llevar α_t a 0 con γ grande.

Con γ = 0,001, en E:

| | Ingenuo | SARIMAX | LightGBM | N-HiTS |
|---|---|---|---|---|
| Cobertura del 90 % propio | 88,0 % | 93,0 % | 70,9 % | 67,3 % |
| Cobertura con ventana (γ = 0) | 86,3 % | 87,9 % | 88,0 % | 89,2 % |
| Cobertura con ACI (IC 95 %) | 88,4 % (85,5-91,3) | 89,0 % (86,3-91,6) | 89,2 % (87,0-91,3) | 89,1 % (87,5-90,7) |
| IS propio / ventana / ACI (MW) | 9.446 / 10.018 / 10.001 | 6.646 / 6.655 / 6.785 | 4.764 / 3.324 / 3.327 | 4.868 / 3.321 / 3.325 |
| IS de ACI − propio | +555 (apreciable, peor) | +139 (no distinguible) | −1.437 (−30 %, apreciable) | −1.544 (−32 %, apreciable) |
| CRPS₁₉ de ACI − propio | +23 (no distinguible) | −40 (apreciable) | −35 (apreciable) | −87 (apreciable) |
| Desviación máx. de fiabilidad | 0,039 | 0,009 | 0,031 | 0,010 |

Mejora de calibración (14.8.5): SARIMAX, LightGBM y N-HiTS sí; ingenuo no (desviación de 0,020 a 0,016). **La
mejora del IS en LightGBM y N-HiTS viene de usar los residuos de las últimas 8 semanas (la ventana), no de la
adaptación de α_t**; en el ingenuo la ventana de 336 residuos pierde frente a los 49.320 errores de 2017-2022.
La cobertura por mes con ACI sigue oscilando (el ingenuo, de 0,81 a 1,00).

**14.13.5 Anticipación de 14.9 frente a lo observado: 25 aciertos y 16 fallos de 41 comprobaciones
automáticas.** *Fallos:* cobertura propia del SARIMAX (92,5 % frente a 82-91) y de N-HiTS (68,1 % frente a
78-92); LightGBM no es el de menor cobertura (lo es N-HiTS); cocientes de fallos 1,57, 1,84 y 0,70 (anticipados
> 2, > 2 y < 0,5: la dirección acertó en los tres, no la magnitud; en E el del SARIMAX es 2,41, pero 14.12
fijó las 8.760 horas); anchura de N-HiTS (1.781 MW, bajo mi rango de 2.200-4.500); desviación máxima de
fiabilidad de los cuatro (mala en SARIMAX, LightGBM y N-HiTS por 14.8.6; aceptable en el ingenuo); γ elegido
(0,001, no 0,005 ni 0,01); anchura de ACI en LightGBM (2.439 MW, bajo mi rango de 2.700-3.500); IS de ACI frente
al propio en el ingenuo (peor, apreciable), LightGBM (−30 %, fuera de −20 a −5 %) y N-HiTS (apreciable, no «no
distinguible»). Fuera de esas 41 comprobaciones: el MAE de la mediana de la segunda red de N-HiTS (710,3 MW) está en mi
rango (680-820), pero es mejor que el de la oficial (716,4 MW), y yo daba ~65 % a que fuera peor; su
cobertura 0,05-0,95 (66,3 %) queda a menos de 3 puntos de la oficial (68,1 %), como anticipé. *Aciertos:* cobertura del ingenuo
y de LightGBM (70,0 %, en el borde del rango 68-86); anchuras de ingenuo, SARIMAX y LightGBM; los cuatro IS
propios dentro de rango; orden por IS y por CRPS₁₉; CRPS₁₉/MAE de los cuatro en 0,70-0,85; sesgo de fiabilidad
de LightGBM y SARIMAX hacia arriba de la diagonal; cobertura de ACI en 88-92 % en los cuatro; horas infinitas
< 1 %. La cobertura propia de LightGBM resulta la más baja en B3 en las 8.760 horas (65,4 % frente a 66,5 % en
B1), pero por 14.8.2 no se distingue de la global; en E la más baja es B1 (67,0 % frente a 67,7 %).

**14.13.6 Lo que se hereda para la memoria y para 2024.** (1) Hallazgo: los cuantiles de LightGBM y de N-HiTS
entrenados con pérdida de cuantil no están calibrados (cobertura del 70 % y 68 % para un 90 % nominal), y el
gaussiano homocedástico del SARIMAX reparte mal la anchura a lo largo del día. (2) Una ventana de residuos
recientes (con o sin ACI) los lleva a cerca del 90 % y baja su IS un 30 %; la adaptación de ACI no aporta en IS
y γ = 0,001 es casi inerte. (3) Optimismo declarado (14.1.3): hiperparámetros y γ elegidos con 2023.
(4) En 2024: γ = 0,001, arranque de D7, mismas reglas de lectura. (5) Para el registro de coherencia del 05:
agrupar la cobertura «por bloque de anticipación» con la cobertura «por franja horaria» (son lo mismo) y anotar
el ingenuo de 2017-2022 (descarte de un error, I3).

## 15. Evaluación final en 2024 (fijado el 3 de noviembre de 2026, antes de escribir su código y de calcular nada con 2024)

Cumple lo dejado pendiente en 2.4, 2.6, 2.7, 14.1.3 y 14.6: la evaluación final, que se hace **una sola vez**, de las
cuatro ganadoras con reajuste mensual, intervalos propios, ventana y ACI, métricas puntuales y probabilísticas,
Diebold-Mariano y reglas de lectura. **No se ha visto ninguna cifra de demanda de 2024.** Para que se sepa qué
conoce el asistente de ese año (se declara porque condiciona su anticipación, 15.9):

1. Las entradas de 2024 (temperatura de reanálisis y calendario) están en el dataset; no se han leído valores.
2. Las cifras de control globales del dataset (archivo 06: media 27.613 MW, máximo 41.829 MW el 8/1/2021, mínimo
   16.117 MW el 25/12/2022; temperatura ponderada máxima 36,52 °C frente a 35,83 °C en 2017-2022) se calcularon con
   las 70.127 horas, 2024 incluido. Implican que ninguna hora de 2024 supera ese máximo ni baja de ese mínimo de
   demanda, y que una hora de 2023 o de 2024 es más cálida que cualquiera de 2017-2022.
3. Conocimiento general del asistente, **no extraído de los datos y no verificado aquí**: la demanda peninsular de 2024
   fue algo mayor que la de 2023 (del orden del 1 %); el autoconsumo solar siguió creciendo; 2024 fue bisiesto; la
   Semana Santa cayó en marzo (Jueves y Viernes Santo, 28 y 29 de marzo) y la de 2023 en abril; la DANA de Valencia
   (29/10/2024) pudo afectar a la demanda de octubre y noviembre. No conoce ninguna cifra horaria, diaria ni mensual
   de 2024.

### 15.1 Alcance y reglas de uso

1. **Una sola evaluación (2.6).** Las previsiones de 2024 se generan una vez, con el protocolo y el código fijados en
   commits anteriores (se anotan los hashes en 15.13). Un error de programación descubierto después se corrige, se
   repite la evaluación y se informa con los dos resultados si difieren (2.6). Nada que se vea en 2024 cambia un
   modelo, un hiperparámetro, γ, una regla de lectura ni este apartado.
2. **Qué se evalúa:** las cuatro ganadoras, con la configuración congelada:
   - **Ingenuo estacional** (15).
   - **SARIMAX** ARMA(5,1) con 12 armónicos diarios, 12 semanales, 3 anuales, interacciones temperatura × hora (3),
     exógenas comunes, `nivel` y retardos de 48 y 168 h (10.10).
   - **LightGBM `c17`** (L2, tasa 0,0318, 127 hojas, `min_data_in_leaf` 20, `feature_fraction` 0,5554,
     `bagging_fraction` 0,8553, `lambda_l2` 0,0572; 11.4), con el procedimiento de 11.3 en cada reajuste.
   - **N-HiTS `c13`** (*L* 336, tasa 0,00119, anchura 256, 1.024 ventanas por lote, escalas B, *dropout* 0,1; 12.6),
     semilla 42, con el procedimiento de 12.5.1 en cada reajuste.
3. **Datos:** `cargar_datos(incluir_prueba=True)`, con la huella comprobada (2.7). En la fase de validación (hasta 14) el
   corte de 2024 seguía activo; esta es la primera vez que se carga completo.
4. **Nada de 2024 se imprime mientras se generan las previsiones.** El script que las genera (28) imprime solo tiempos,
   número de árboles o de pasos, convergencia, pruebas de fuga y controles; ninguna métrica de error, cobertura o
   sesgo. La primera cifra de error de 2024 que se ve es la del script de lectura (29), que se ejecuta una vez y
   compara con la anticipación de 15.9 de forma automática.
5. **Entradas con pérdida de generalidad que se mantienen:** escenario de previsión meteorológica perfecta (temperatura
   de reanálisis, 2.2 y 4); regla de información de 1.2; retardos ≥ 37 h (1.5).

### 15.2 Decisiones y motivos

Decisiones del autor (3/11/2026, en el encargo de la sesión):

| # | Decisión | Motivo |
|---|---|---|
| F1 | Se evalúan las **cuatro ganadoras** de 15.1.2, con la configuración congelada | Evita una segunda selección con 2024: los hiperparámetros se eligieron con 2023 (14.1.3) y 2024 es lo único que no ha intervenido en ninguna elección |
| F2 | **Reajuste mensual de los cuatro** con ventana creciente: desde el 1/1/2017 hasta el origen del reajuste, es decir, 2017-2023 más el 2024 ya ocurrido | Es lo fijado en 2.4 y lo que haría un operador real. La regla de 2.4 (si el SARIMAX superara 8 h, trimestral para los cuatro) no se activa: el SARIMAX cuesta unos 20-30 minutos (8.3, 15.10) |
| F3 | **Reajuste mensual también de los 19 modelos de cuantil de LightGBM y de la segunda red de N-HiTS** | La rejilla del CRPS₁₉ debe describir el mismo sistema que la previsión puntual. Con modelos de cuantil de 2023 aplicados a 2024, la rejilla y la previsión puntual tendrían edades distintas y el CRPS₁₉ mediría también esa diferencia |
| F4 | **Intervalos propios del 90 %** (14.3), **ventana** (ACI con γ = 0, referencia de 14.6.7) y **ACI con γ = 0,001**, con el **arranque D7** de 14.2 | γ no se vuelve a elegir con 2024: sería circular (2.6) y la regla de 14.6.7 ya dio su resultado (negativo, 14.13.4). La ventana se informa siempre porque es lo que separa «ventana» de «adaptación» (14.13.4) |
| F5 | Métricas **puntuales** (7) y **probabilísticas** (14.5), **Diebold-Mariano** (15.6) y **reglas de lectura** (15.8), con criterios y anticipación escritos antes de ver nada | Es la condición de 2.6 y de 14.9: sin ella, la lectura se haría a la vista de las cifras |
| F6 | **Coste estimado y orden de recortes fijados ahora** (15.10), antes de lanzar | Regla del plan (archivo 03, riesgo 5): el orden de recortes se escribe antes de lanzar, no después |

Decisiones tomadas por el asistente por defecto (el autor puede cambiarlas antes del commit de este apartado):

| # | Decisión | Motivo |
|---|---|---|
| F7 | **Los cuantiles empíricos del ingenuo se actualizan cada mes:** errores de los días objetivo que terminan antes del origen del reajuste (2017-2023 en enero; 2017-2023 más el 2024 ya ocurrido después), sin COVID. **Precisa 14.3.1**, que decía «los de 2017-2023» para la evaluación final | Mismo trato de la información que a los demás modelos (ventana creciente común, 2.1). Dejarlos fijos en 2017-2023 dejaría al ingenuo como único modelo que no aprovecha el 2024 ya ocurrido. Alternativa no elegida: fijos en 2017-2023 (más sencillo; el cambio esperado es pequeño porque cada mes añade unos 700 errores a unos 49.000) |
| F8 | **SARIMAX: FGLS desde cero en cada reajuste** (mismo orden ARMA(5,1), mismos armónicos, mismas exógenas, misma regla de parada de 8.6); si no converge, **se detiene todo** y se informa; no hay recurso silencioso a otro orden | Es el procedimiento de 2023. En 10.9 hubo un «segundo grupo» de soluciones del FGLS (sesgo positivo, 1.059-1.112 MW): un reajuste podría caer en otra solución. Se guardan los parámetros ARMA de cada mes para poder verlo |
| F9 | **Parada temprana repetida en cada reajuste** en los 20 modelos de LightGBM (puntual y 19 cuantiles) y en las dos redes de N-HiTS, con las últimas 8 semanas anteriores al origen como control, semilla 42 y los hiperparámetros congelados (11.3, 12.5.1, 14.4). Se guarda el número de árboles o de pasos de cada reajuste | Es el procedimiento descrito en 11.3 y 12.5.1: «reajuste» incluye volver a elegir el número de árboles o de pasos con datos anteriores al origen, nunca con 2024 posterior. Alternativa más barata no elegida: fijar el número de la validación (no es lo que dicen 11.3 y 12.5.1) |
| F10 | **Intervalo propio de LightGBM = cuantiles 0,05 y 0,95 de los modelos de la rejilla** (idénticos al procedimiento de 11.2, con la regla de cruce); **el de N-HiTS = la red oficial** (12.2), no la segunda red | Igual que en 14.3: en 2023 los modelos de 0,05 y 0,95 de la rejilla reproducían los guardados con diferencia 0 |
| F11 | **ACI sobre la concatenación 2023 + 2024:** las horas de 2023 solo aportan residuos; el primer día emitido es el 1/1/2024 (día 365 de la concatenación); α_t inicial = 0,10; ventana de 336 residuos por bloque. La **evaluación ocupa las 8.784 horas de 2024** (ya no hay 57 días sin emitir). Se calculan γ ∈ {0; 0,001; 0,005; 0,01; 0,02} (sensibilidad, 15.8.8), y las métricas oficiales usan γ = 0,001 | Es D7. Los residuos de 2023 son los del modelo de 2023 (ajustado hasta 2022); los modelos de 2024 se reajustan cada mes. Limitación declarada, no corregible sin reentrenar 2023 |
| F12 | **Diebold-Mariano (15.6):** pérdida diaria, error absoluto como principal y cuadrado como robustez, varianza de Newey-West con 7 retardos, corrección de Harvey, Leybourne y Newbold, distribución *t* de *n* − 1, **Holm** sobre los seis pares | Motivos en 15.6. Se fija todo antes de ver cifras, incluido el retardo 7 (sin ajustarlo) |
| F13 | Las diferencias de **IS y CRPS₁₉** se leen como en 14.8.4 (bootstrap de días), no con Diebold-Mariano | Las dos son puntuaciones propias de la distribución, no una pérdida del punto; el bootstrap de días ya está probado en 14 |
| F14 | **Comparación con 2023 solo descriptiva:** cambio relativo del MAE y regla de ±10 % (15.8.3). Sin contraste | 2023 y 2024 son años distintos con modelos distintos (reajustados); un contraste daría una apariencia de rigor que no existe. La «maldición del ganador» (14.1.3) no se puede separar del efecto del año |
| F15 | **Candado:** el script de lectura (29) se niega a ejecutarse si ya existe su carpeta de resultados salvo con la opción explícita `--repetir-por-error "motivo"`, que deja el motivo en el manifiesto. Manifiesto con el hash del commit, el estado limpio del árbol, las huellas del dataset y las versiones de las librerías | 2.6: una sola vez, y trazabilidad de qué código dio qué cifra |
| F16 | **Pruebas de fuga reforzadas (modo `borrar`) con el modelo de cada mes sobre todos sus orígenes,** en las seis salidas (ingenuo, SARIMAX, LightGBM puntual, LightGBM 19 cuantiles, N-HiTS oficial, N-HiTS 19 salidas); más una prueba nueva del **reajuste** (15.7.3) | Con reajustes mensuales aparece un camino que en 2023 no existía: el entrenamiento con un `fin` variable. La prueba de 2023 comprobaba la previsión, no que el ajuste ignorara los datos posteriores al origen del reajuste |
| F17 | **No se hace un control «sin reajuste»** (los modelos de 2023 aplicados a 2024 sin reajustar). Los pendientes del archivo 06 («si el reajuste mensual reduce el sesgo del SARIMAX» y «si cambia la calibración de los cuantiles») se contestan solo **descriptivamente**, comparando 2024 con 2023, con el año y el reajuste mezclados | Sale del alcance definido (el protocolo 2.4 fija el reajuste mensual; no pide medir su efecto). Coste del control si el autor lo pidiera: unos 15 minutos de cálculo (SARIMAX 2, LightGBM 3, N-HiTS 8 y la rejilla de N-HiTS 9, con modelos ya entrenados en 2023 reproducibles) más código. **Decisión del autor antes del commit** |
| F18 | **Lanzamiento opcional en dos terminales** (15.10.3): el script 28 admite `--componentes` | No cambia ningún cálculo (hilos fijados: LightGBM 4, torch 4, SARIMAX 1; el PC tiene 12 núcleos lógicos); reduce el tiempo de reloj |

### 15.3 Calendario de reajustes y procedimiento de cada componente

**Calendario (2.4).** El reajuste *k* se hace en el origen del primer día objetivo de cada mes (las 12:00 locales del
último día del mes anterior); el modelo resultante prevé todos los orígenes hasta el siguiente reajuste. Con
`generar_origenes("prueba")` (366 orígenes, 8.784 horas: 365 × 24, más el 29/2, menos la hora del 31/3, más la hora
del 27/10):

| *k* | Origen del reajuste (hora local) | Días objetivo que prevé | Días |
|---|---|---|---|
| 1 | 31/12/2023 12:00 | 1/1 a 31/1/2024 | 31 |
| 2 | 31/1/2024 12:00 | 1/2 a 29/2 | 29 |
| 3 | 29/2/2024 12:00 | 1/3 a 31/3 (día de 23 h el 31/3) | 31 |
| 4 | 31/3/2024 12:00 | 1/4 a 30/4 | 30 |
| 5 | 30/4/2024 12:00 | 1/5 a 31/5 | 31 |
| 6 | 31/5/2024 12:00 | 1/6 a 30/6 | 30 |
| 7 | 30/6/2024 12:00 | 1/7 a 31/7 | 31 |
| 8 | 31/7/2024 12:00 | 1/8 a 31/8 | 31 |
| 9 | 31/8/2024 12:00 | 1/9 a 30/9 | 30 |
| 10 | 30/9/2024 12:00 | 1/10 a 31/10 (día de 25 h el 27/10) | 31 |
| 11 | 31/10/2024 12:00 | 1/11 a 30/11 | 30 |
| 12 | 30/11/2024 12:00 | 1/12 a 31/12 | 31 |

La ventana de cada reajuste son las horas con `hora_utc + 1 h ≤ origen del reajuste`, con COVID estricto como en 11.1.3
y 12.1.3 (en 2024 no interviene). El reajuste 1 usa 2017-2023 hasta el 31/12/2023 12:00; los siguientes añaden el
2024 ya ocurrido. Reajuste «independiente»: cada mes se ajusta desde cero (el SARIMAX desde el mismo punto inicial del
FGLS; los árboles y las redes desde la inicialización de la semilla 42), sin arrastrar parámetros del mes anterior.

**15.3.1 Ingenuo.** Sin entrenamiento. Previsión de 15. Intervalo propio y rejilla: previsión + cuantil empírico del error
por bloque de anticipación (14.3.1), con los errores (F7) de los días objetivo desde el 8/1/2017 que terminan antes del
origen del reajuste *k*, sin los que tocan COVID (objetivo o fuente a 7 días) y descartando los no finitos que se
cuenten (14.12, I3: uno en enero).

**15.3.2 SARIMAX.** En cada reajuste, FGLS con `ajustar_fgls` (orden (5, 0, 1), exógenas de 15.1.2) sobre la ventana;
convergencia obligatoria (F8). En cada origen del mes, `res.apply` con los datos anteriores al origen y parámetros
fijos, `get_forecast`: media (previsión puntual) y error típico (intervalo gaussiano y rejilla, 14.3.2). Se guardan los
parámetros y el número de vueltas del FGLS de cada mes.

**15.3.3 LightGBM.** En cada reajuste: `filas_entrenamiento(datos, fin)` y 20 entrenamientos con `lm.entrenar` (11.3): el
puntual (parámetros de `c17`) y los 19 de cuantil (`alpha` = 0,05 … 0,95, mismos parámetros, 14.4.3). Se guardan los
árboles de cada modelo y mes. Intervalo propio = cuantiles 0,05 y 0,95 con la regla de cruce (11.2.5): se intercambian
y se cuenta. Rejilla = los 19, con reordenación monótona (14.4.5) y recuento.

**15.3.4 N-HiTS.** En cada reajuste, dos entrenamientos con el procedimiento de 12.5.1 y la semilla 42: la **red oficial**
(`nm.entrenar`; mediana = previsión puntual; cuantiles 0,05 y 0,95 = intervalo propio con regla de cruce, 12.2.5) y la
**segunda red** de 19 salidas (`nr.entrenar`; solo rejilla del CRPS₁₉, 14.4.4). Se guardan los pasos de cada red y mes.
La segunda red no sustituye a la oficial y se sigue informando de su MAE y de su cobertura frente a la oficial.

**15.3.5 Salidas.** Cada componente y mes se guarda al terminar en `data/interim/final_2024/` (reanudable, 15.12). Al
acabar todos, se ensamblan en `data/predicciones/` con el formato común: `{modelo}_prueba.parquet` (hora, fecha
objetivo, real, previsión), `{modelo}_cuantiles_prueba.parquet` (rejilla q05 … q95) y, para LightGBM y N-HiTS,
`{modelo}_intervalos_prueba.parquet` (inferior, superior).

### 15.4 Intervalos y capa probabilística en 2024

1. **Intervalos propios del 90 %:** 14.3 con los reajustes de 15.3. Rejilla de 19 cuantiles: 14.4.
2. **Ventana y ACI (14.6, F11):** entradas = previsión puntual oficial y demanda real de las 8.760 horas de 2023
   (previsiones de validación de cada modelo) y de las 8.784 de 2024. Orígenes de 2023 y de 2024 en orden cronológico;
   `primer_dia` = 365. Bloques de anticipación de 14.5.1 (B1 = 00-05 h locales, B2 = 06-11, B3 = 12-17, B4 = 18-23).
   Las puntuaciones iniciales del primer origen de 2024 son los 336 residuos más recientes de cada bloque observados a
   las 12:00 del 31/12/2023 (D7); α_t = 0,10 en todos los bloques. Recursión con el error del intervalo realmente
   emitido (D4).
3. **Cuantiles después de ACI** para el CRPS₁₉ y la fiabilidad: 14.6.8 con γ = 0,001.
4. **Conjunto de evaluación:** las 8.784 horas de 2024 para todos los intervalos (propios, ventana y ACI). Es la
   diferencia con 2023, donde ACI se evaluaba en 308 días (E) y los propios, además, en 8.760.

### 15.5 Métricas

1. **Puntuales** (7, para los cuatro modelos, 8.784 horas): MAE, RMSE, MAPE, MASE (escala fija 1.328,2 MW), MAE relativo
   al ingenuo, sesgo medio (real − previsto), razón RMSE/MAE; MAE y sesgo por mes y por bloque de anticipación;
   comparación con 2023 (cambio relativo del MAE). Intervalo de confianza del 95 % del MAE por bootstrap de días (D10).
2. **De los intervalos y de la rejilla** (14.5.3-14.5.8): para los intervalos propios, ventana (γ = 0) y ACI (γ = 0,001):
   cobertura (global, por bloque y por mes, con IC por bootstrap de días), fallos por debajo y por encima y su
   cociente, anchura media y mediana, IS con sus dos pinball, CRPS₁₉, fiabilidad de la rejilla y su desviación máxima,
   horas con límite infinito sustituido (D8), horas reordenadas, y diferencias pareadas con IC (14.8.4).
3. **Sensibilidad a γ** (solo descriptiva, 15.8.8): cobertura, anchura, IS e IS relativo a γ = 0 para cada γ de G, y la
   puntuación de la regla 14.6.7 recalculada con 2024.
4. **Semana de ejemplo** de la figura, fijada ahora: del lunes 13/5/2024 al domingo 19/5/2024 (la misma posición
   semanal que la del 15 al 21/5/2023).
5. **Qué no se calcula aquí:** el desglose por hora del día y por tipo de día (OE8) es el apartado 16 y se fijará, con
   commit, antes de leerlo; leerá las previsiones guardadas y no decidirá nada.

### 15.6 Diebold-Mariano

**Qué contrasta.** Si dos modelos tienen la misma precisión esperada. Para dos modelos *a* y *b* se toma la pérdida de
cada día (media de las horas del día objetivo) y su diferencia *d_t* = pérdida_*a*(*t*) − pérdida_*b*(*t*), en MW
(error absoluto) o en MW² (error cuadrático). *d_t* negativa significa que *a* se equivocó menos ese día.

1. **Unidad de análisis: el día** (*n* = 366). Motivo: las 24 horas de un día comparten errores muy correlacionados
   (13.8, 14.13.2); tratar las horas como independientes inflaría la significación. Las horas de un día se promedian
   antes de contrastar.
2. **Pérdida principal: error absoluto** (coherente con el MAE, el criterio de selección de 3.4 y con 14.8.4). **Robustez:
   error cuadrático** (coherente con el RMSE).
3. **Estadístico:** DM = media(*d*) / √(*V̂* / *n*), con *V̂* la varianza de largo plazo de *d_t*: *V̂* = γ̂₀ + 2 Σₖ₌₁ᵐ
   (1 − *k*/(*m* + 1)) γ̂ₖ (núcleo de Bartlett, Newey y West), γ̂ₖ la autocovarianza muestral de orden *k* (divisor *n*).
   **m = 7 días**, fijado ahora: la dependencia diaria de los errores tiene memoria semanal (OE2: ACF de la demanda a
   168 h) y la meteorología persiste varios días. Sensibilidad (solo informada): *m* = 3 y *m* = 14.
4. **Corrección de muestra pequeña** de Harvey, Leybourne y Newbold con horizonte *h* = 1 (un origen por día):
   DM\* = DM · √((*n* − 1)/*n*), contrastada con una *t* de Student de *n* − 1 grados de libertad, a dos colas.
5. **Seis pares** (ingenuo, SARIMAX, LightGBM, N-HiTS): se informa la matriz de p-valores sin ajustar y **ajustados por
   Holm** (control de la probabilidad de algún falso positivo entre las seis comparaciones). La lectura usa los
   ajustados (15.8.1).
6. **Cuándo no vale:** si *V̂* ≤ 0 se informa NA (el núcleo de Bartlett lo evita en la práctica); si el orden de dos modelos
   cambia entre mes y mes, el contraste anual promedia ese cambio y no lo detecta (se mira el MAE por mes).
7. **Ejemplo numérico inventado (solo para explicarlo):** si en los 366 días la diferencia de MAE diario entre dos modelos
   fuera −100 MW de media, con desviación de 250 MW y autocorrelación semanal moderada, la desviación típica de la media
   sería ≈ 250/√366 · 1,5 ≈ 20 MW y DM ≈ −5: un valor así rechaza la igualdad con p muy pequeño. Con −15 MW, DM ≈ −0,7 y no.

### 15.7 Controles y pruebas (el script se detiene si falla alguno)

**Generación (28):**

1. **Calendario:** 12 reajustes con los orígenes de la tabla de 15.3; 366 orígenes y 8.784 horas; el día de 23 h es el 31/3/2024
   y el de 25 h el 27/10/2024; las horas recortadas coinciden con las del dataset; `real` idéntica al dataset en todos
   los archivos; sin nulos.
2. **Configuración congelada:** `parametros_c17()` y `config_c13()` (sus aserciones reconstruyen y comprueban el sorteo
   de la validación), orden ARMA (5, 0, 1) y exógenas del SARIMAX de 10.10.
3. **Fuga del reajuste (nueva):** para el reajuste 1 y el reajuste 7 (fin = 31/12/2023 y 30/6/2024), el ajuste del
   SARIMAX y de LightGBM (puntual y cuantiles 0,05 y 0,95) con la demanda posterior al origen del reajuste sustituida
   por +10.000 MW produce modelos y previsiones **idénticos** (diferencia 0). Para N-HiTS y la segunda red se comprueban
   las aserciones del propio ajuste (`hora_utc + 1 h ≤ fin` en todas las filas, `local_scaler_type=None`) porque repetir
   su entrenamiento costaría unos 15 minutos por reajuste; la prueba completa con alteración se hace en el entorno del
   asistente con presupuestos reducidos (15.12.3).
4. **Fuga de la previsión:** prueba reforzada (demanda posterior al origen borrada, `comprobar_fuga(modo="borrar")`) con el
   modelo de cada mes sobre sus orígenes y en las seis salidas. Debe dar 0 días.
5. **Convergencia:** FGLS convergido en los 12 meses; árboles y pasos dentro de los límites (0 < árboles ≤ 10.000; 100 ≤ pasos ≤
   1.000, múltiplos de 100); rejillas finitas.
6. **Cuantiles de cada fila:** número de cruces (regla 11.2.5) y de horas reordenadas por modelo y mes, guardados, sin
   interpretar aún.

**Lectura (29):**

7. **ACI:** recursión propia igual a MAPIE al bit; corte de causalidad con control positivo (con cortes en 2024: días 465,
   565 y 665 de la concatenación); prueba sintética de 0,90 ± 0,02 (14.7.2 a 14.7.4).
8. **Diebold-Mariano:** (a) con diferencias independientes de media 0 y 366 días, la tasa de rechazo al 5 % en 2.000
   réplicas está entre 3 y 8 %; (b) con una diferencia de −0,3 desviaciones típicas por día, rechaza en más del 80 % de
   las réplicas; (c) con autocorrelación AR(1) de 0,5 en *d_t*, el retardo 7 mantiene la tasa de rechazo por debajo
   del 10 %. Las réplicas usan semillas fijas y no tocan 2024.
9. **Reproducción de métricas puntuales:** MAE, RMSE y MASE recalculados con dos implementaciones (la de `validacion.py`
   y una independiente) coinciden a 10⁻⁹.
10. **Candado y manifiesto** (F15).

### 15.8 Reglas de lectura (fijadas antes de ver los resultados)

1. **Diferencia entre dos modelos (puntual):** *apreciable* si el p-valor de Diebold-Mariano ajustado por Holm (pérdida absoluta,
   *m* = 7) es < 0,05; en otro caso, *no distinguible*. Si la pérdida cuadrática da una conclusión distinta, se dice. El
   orden de los cuatro por MAE solo se llama «establecido» si los tres pares consecutivos son apreciables.
2. **Sesgo:** *apreciable* si |sesgo| supera 150 MW (13.5). Se dice si el reajuste mensual parece haber reducido el sesgo
   del SARIMAX (−346 MW en 2023): «sí» si |sesgo| ≤ 196 MW, con el aviso de que el año y el reajuste están mezclados (F17).
3. **Comparación con 2023:** cambio relativo del MAE entre −10 % y +10 %: *similar*; por encima de +10 %: *peor que en
   validación*; por debajo de −10 %: *mejor*. Sin contraste (F14). Se recuerda que los hiperparámetros de LightGBM y de
   N-HiTS se eligieron con 2023, así que una pérdida respecto a 2023 puede ser maldición del ganador, efecto del año o
   ambas, sin poder separarlas.
4. **Cobertura global del 90 %, bloque, mes, asimetría, IS, CRPS₁₉, mejora de calibración, fiabilidad e infinitos:** las
   reglas 14.8.1 a 14.8.7, sin cambios, sobre las 8.784 horas.
5. **Ventana frente a adaptación:** la mejora del IS se atribuye a la adaptación solo si IS(γ = 0,001) es *apreciablemente*
   menor que IS(γ = 0) (14.8.4); en otro caso se atribuye a la ventana.
6. **Calibración heredada de 2023:** la infracobertura de LightGBM y N-HiTS «se repite» si su cobertura propia de 2024 es
   < 88 % con el IC del 95 % por debajo de 90 %. Compatible con, pero no demuestra, la hipótesis de 14.13.2 (cuantiles mejor
   ajustados en entrenamiento que fuera de muestra).
7. **Resultados negativos:** se informan tal cual (14.8.8). En particular, si un modelo no mejora al ingenuo o al
   anterior, es un hallazgo.
8. **Sensibilidad a γ:** la puntuación de 14.6.7 recalculada con 2024 es solo descriptiva; aunque otro γ saliera mejor,
   γ = 0,001 sigue siendo el de la evaluación (F4).
9. **Nada de lo leído modifica un modelo ni este apartado** (15.1.1).

### 15.9 Anticipación del asistente (fijada antes de escribir el código y de ver nada de 2024)

Cifras entre paréntesis: confianza subjetiva. Rangos para las 8.784 horas de 2024, salvo que se diga otra cosa. El script
de lectura (29) compara cada punto con lo observado y guarda la tabla (✓ o ✗), como 27 con 14.9 (I4); se leen los ✗ sin
reinterpretar. Los rangos incorporan el conocimiento general de la introducción y la lección de 14.13.5 (16 fallos de 41 en
la capa probabilística: tiendo a quedarme estrecho en las coberturas y a no anticipar bien las magnitudes).

| # | Comprobación | Anticipación |
|---|---|---|
| A0 | Todos los controles de 15.7 pasan | 85 % |
| A1 | MAE del ingenuo | 1.220-1.500 MW (centro 1.350) |
| A2 | MAE del SARIMAX | 830-1.060 MW (centro 920) |
| A3 | MAE de LightGBM | 540-680 MW (centro 600) |
| A4 | MAE de N-HiTS | 650-850 MW (centro 730) |
| A5 | Orden por MAE LightGBM < N-HiTS < SARIMAX < ingenuo | 70 % |
| A6 | LightGBM es el de menor MAE | 88 % |
| A7 | N-HiTS tiene menor MAE que el SARIMAX | 75 % |
| A8 | MAE relativo al ingenuo | LightGBM 0,38-0,52; N-HiTS 0,46-0,64; SARIMAX 0,60-0,80 |
| A9 | Sesgo (real − previsto) | ingenuo −100 a +180; SARIMAX −330 a +100; LightGBM −250 a +60; N-HiTS +20 a +330 MW |
| A10 | Signo del sesgo | N-HiTS > 0 (85 %); LightGBM < 0 (75 %); SARIMAX < 0 (70 %) |
| A11 | Razón RMSE/MAE | entre 1,18 y 1,62 en los cuatro; la de N-HiTS menor que la del SARIMAX (70 %) |
| A12 | Cambio de MAE frente a 2023 dentro de ±10 % (*similar*) | ingenuo 70 %; SARIMAX 65 %; LightGBM 65 %; N-HiTS 60 % |
| A13 | LightGBM y N-HiTS no mejoran su MAE de 2023 (cambio ≥ 0) | 65 % |
| A14 | Sesgo del SARIMAX con \|sesgo\| ≤ 196 MW (el reajuste lo reduce) | 50 % |
| A15 | Pares apreciables por Diebold-Mariano (Holm, pérdida absoluta) | LightGBM-SARIMAX 95 %; LightGBM-N-HiTS 75 %; N-HiTS-SARIMAX 75 %; SARIMAX-ingenuo 85 %; N-HiTS-ingenuo 97 %; LightGBM-ingenuo 99 %; los seis a la vez 55 % |
| A16 | Con pérdida cuadrática se llega a la misma conclusión en los seis pares | 75 % |
| A17 | Cobertura del intervalo propio | ingenuo 86-91 %; SARIMAX 89-95 %; LightGBM 60-78 %; N-HiTS 58-76 % |
| A18 | LightGBM y N-HiTS infracubren (< 88 %); los dos por debajo del ingenuo y del SARIMAX | 93 %; 85 % |
| A19 | Cociente de fallos | SARIMAX > 1,3 (65 %); LightGBM > 1,2 (60 %); N-HiTS < 0,9 (60 %); ingenuo entre 0,7 y 1,4 (70 %) |
| A20 | Anchura media del propio (MW) | ingenuo 5.500-6.300; SARIMAX 4.100-4.800; LightGBM 1.600-2.400; N-HiTS 1.500-2.300 |
| A21 | IS del propio (MW) | ingenuo 8.800-10.200; SARIMAX 6.100-7.300; LightGBM 4.400-5.500; N-HiTS 4.500-5.600 |
| A22 | Orden por IS LightGBM < N-HiTS < SARIMAX < ingenuo | 55 % |
| A23 | CRPS₁₉ del propio (MW) | ingenuo 950-1.120; SARIMAX 640-780; LightGBM 400-500; N-HiTS 470-600 |
| A24 | Orden por CRPS₁₉ LightGBM < N-HiTS < SARIMAX < ingenuo | 75 % |
| A25 | CRPS₁₉/MAE entre 0,70 y 0,80 en los cuatro | 80 % |
| A26 | Desviación máxima de fiabilidad > 0,06 (*mala*) en SARIMAX, LightGBM y N-HiTS; ≤ 0,06 en el ingenuo | 80 %; 70 % |
| A27 | Cobertura del SARIMAX en B1 (00-05 h) mayor que en B3 (12-17 h) en más de 5 puntos | 80 % |
| A28 | Con ACI (γ = 0,001), cobertura entre 88 y 92 % en los cuatro; entre 86 y 93 % en los cuatro | 60 %; 85 % |
| A29 | IS de ACI frente al propio: LightGBM y N-HiTS bajan entre un 15 y un 40 % con diferencia apreciable | 75 % |
| A30 | IS de ACI frente al propio: SARIMAX no distinguible; ingenuo peor y apreciable | 55 %; 50 % |
| A31 | IS(γ = 0,001)/IS(γ = 0) entre 0,99 y 1,02 en los cuatro (la adaptación sigue sin aportar) | 85 % |
| A32 | Puntuación de 14.6.7 recalculada con 2024 > 1 para los cuatro γ; γ = 0,001 la menor | 60 %; 65 % |
| A33 | Horas infinitas con γ = 0,001: 0 en los cuatro; con γ = 0,02: > 0 en al menos tres modelos | 90 %; 70 % |
| A34 | Desviación máxima de fiabilidad con ACI ≤ 0,06 en los cuatro | 80 % |
| A35 | CRPS₁₉ de ACI frente al propio: LightGBM y N-HiTS mejoran con diferencia apreciable | 75 % |
| A36 | Tiempo total de 28 en serie entre 3,5 y 6,5 h; N-HiTS > 60 % del total; ningún recorte de 15.10 | 75 %; 80 %; 85 % |
| A37 | Pasos de las dos redes de N-HiTS en el tope de 1.000 en al menos 8 de los 12 reajustes | 70 % |

### 15.10 Coste, presupuesto y orden de recortes

**15.10.1 Estimación (sin medir en 2024; a partir de lo medido en 2023 en el PC del autor).**

| Componente | Base medida en 2023 | Estimación para 12 reajustes (sin margen / con ×1,3) |
|---|---|---|
| Ingenuo | menos de 1 min en total | despreciable |
| SARIMAX | ajuste FGLS 1,2 min (8.3); 0,22 s por origen (8.3); el script 26 completo, 7,2 min | 0,3-0,4 h / 0,5 h |
| LightGBM (puntual + 19 cuantiles) | 19 modelos con su prueba de fuga, 2,6 min (14.13); puntual de `c17`, 0,28 min | 0,6 h / 0,8 h |
| N-HiTS oficial | `c13` en la búsqueda, 6,4 min (parada temprana, reentrenamiento y 365 previsiones) | 1,3 h / 1,7 h |
| N-HiTS, segunda red | 8,9 min con su prueba de fuga (14.13), 1.000 pasos | 1,8 h / 2,3 h |
| **Total en serie** | | **≈ 4,1 h / ≈ 5,3 h** |
| Lectura (29) | 27 tardó 17 s con la mitad de datos | unos 1-2 min |

El factor 1,3 es el de 12.4.3 (la serie crece con cada reajuste y la librería la recorre entera). En dos terminales
(15.10.3) el tiempo de reloj baja a ≈ 3,1-4,0 h, más hasta un 20 % por competencia de núcleos que no se ha medido.

**15.10.2 Presupuesto y condición de «cabe».** El presupuesto es el de 2.4 y 12.4.3: **evaluación final ≤ 8 h de reloj**. 28 mide
el tiempo del primer reajuste de cada componente, imprime la proyección (×12 × 1,3) y la suma. Se usa el orden de recortes de
15.10.4 solo si la suma proyectada supera 8 h.

**15.10.3 Primero, sin recorte:** lanzar en dos terminales: terminal A, `--componentes ingenuo,sarimax,lightgbm`; terminal B,
`--componentes nhits`. No cambia ningún cálculo.

**15.10.4 Orden de recortes (fijado ahora; se aplica el primero que haga cumplir las 8 h; cada uno requiere la confirmación del
autor, regla de los puntos de control del archivo 03):**

1. **Segunda red de N-HiTS: reajuste trimestral** (reajustes 1, 4, 7 y 10; ahorro ≈ 1,2-1,5 h). Efecto: el CRPS₁₉ y la fiabilidad
   de N-HiTS usan una red de hasta dos meses de antigüedad respecto a la oficial; se declara.
2. **Los 17 modelos de cuantil de LightGBM distintos de 0,05 y 0,95: reajuste trimestral** (ahorro ≈ 0,3 h). Efecto: la rejilla
   mezcla cuantiles de edades distintas (los de 0,05 y 0,95 siguen siendo mensuales); la reordenación monótona lo absorbe y se
   cuenta.
3. **Parar y decidir con el autor.** No se recortan nunca: los cuatro modelos puntuales, los intervalos propios, ventana, ACI,
   Diebold-Mariano ni las pruebas de fuga.

**15.10.5 Interrupciones.** 28 guarda cada (componente, mes) al terminar y se reanuda con el mismo comando; los reajustes son
independientes y deterministas (misma semilla, mismos hilos), así que reanudar no cambia ninguna cifra. El PC no debe
suspenderse durante el cálculo.

### 15.11 Riesgos y límites declarados

1. **Maldición del ganador (14.1.3):** los hiperparámetros de LightGBM y N-HiTS y γ se eligieron con 2023; la caída de rendimiento
   frente a 2023 mezcla ese efecto con el del año (F14). Los diseños del SARIMAX (`nivel`, retardos) también se decidieron
   viendo 2023 (10.8 y 10.9).
2. **Ruido de N-HiTS entre reentrenamientos:** el rango entre semillas fue de 66,4 MW en 2023 (12.6). Cada reajuste mensual es otro
   entrenamiento con la misma semilla y datos distintos; el MAE anual de N-HiTS tiene una incertidumbre que no es de muestreo
   de días y que el bootstrap de días no recoge.
3. **SARIMAX:** el FGLS puede caer en otra solución en algún mes (F8); normalidad y parámetros fijos dentro del mes (14.10.2).
4. **ACI:** retroalimentación con 24-48 h de retraso (14.6.3); arranque con residuos de 2023 del modelo de 2023 (F11); ventanas de
   8 semanas que mezclan laborables y festivos (14.10.5). Cuatro bloques de 6 h.
5. **Segunda red de N-HiTS:** otro sistema distinto de la oficial (14.10.3).
6. **Escenario de previsión meteorológica perfecta:** los resultados acotan por arriba el rendimiento (2.2).
7. **Un solo año de prueba:** 2024 tiene sus rarezas (bisiesto, Semana Santa en marzo, la DANA de octubre, 8/12 en domingo). Se
   cuentan en el desglose, no se corrigen.
8. **El asistente conoce hechos generales de 2024** (introducción): su anticipación no es ciega al contexto, aunque sí lo es a
   los datos.
9. **Lo que 28 guarda contiene la demanda real de 2024 junto a las previsiones:** el autor no debe abrir esos archivos antes de
   ejecutar 29.

### 15.12 Archivos, orden de ejecución y registro de coherencia

1. `src/finales.py` (módulo sin número: calendario de reajustes, ingenuo mensual, ensamblado, Diebold-Mariano, candado y manifiesto),
   `src/28_prevision_final_2024.py` (genera previsiones por componente y mes; no imprime métricas de error) y
   `src/29_evaluacion_final_2024.py` (lectura única: métricas, ACI, Diebold-Mariano, tabla de anticipación, tablas y figuras en
   `data/resultados/final/`). Se reutilizan sin modificar `validacion.py`, `lightgbm_modelo.py`, `nhits_modelo.py`,
   `nhits_rejilla.py`, `probabilistica.py`, `exogenas.py` y `sarimax_fgls.py`.
2. **Orden:** commit de este apartado → código y pruebas en el entorno del asistente → commit del código (hash a 15.13) → 28 en el
   PC del autor (en una o dos terminales) → 29 una vez → apartado 15.14 con los resultados, y apartado 16 (desglose) antes de leerlos.
3. **Pruebas sin 2024:** en el entorno del asistente, un arnés sustituye la demanda de 2024 por la de 364 días antes más ruido inventado
   del 1,2 % (misma posición semanal) y reduce los presupuestos (120 árboles, 60 pasos, 1 vuelta de FGLS); con él se ejecutan 28 y 29
   completos, incluida una prueba de fuga del reajuste con alteración de la demanda posterior al origen en los cuatro modelos y las
   dos redes. Ninguna cifra impresa en ese entorno dice nada de 2024; el dataset real de 2024 no se abre en él.
4. **Estimación de tiempo del código** en el entorno del asistente (2 núcleos): no sirve para el criterio de 15.10 (solo el PC del
   autor mide).
5. **Para el registro de coherencia del 05** (la memoria no se toca hasta R2; C45-C52):
   C45 reajuste mensual de los cuatro y calendario de 15.3 (4.2, 4.4); C46 cuantiles del ingenuo actualizados cada mes (precisa
   C38/C42; 4.4); C47 Diebold-Mariano: pérdida diaria, Newey-West con 7 retardos, Harvey-Leybourne-Newbold y Holm (3.6 y 4.4;
   comprobar que el texto actual de 3.6 coincide); C48 ACI en 2024: concatenación con 2023, arranque D7 y su limitación (3.7, 4.4,
   5.5); C49 comparación con 2023 solo descriptiva y maldición del ganador (5.4, 5.7); C50 coste y tiempo reales de la evaluación final
   (4.5, entorno); C51 el asistente conocía hechos generales de 2024: la anticipación es del protocolo, no de la memoria, salvo decisión
   del autor (como C44); C52 sin control «sin reajuste» (F17): los pendientes del archivo 06 se contestan descriptivamente (5.4, 5.7).
   Verificar en la fuente, antes de citarlos en R3, Diebold y Mariano (1995), Harvey, Leybourne y Newbold (1997), Newey y West (1987)
   y Holm (1979); no dar sus datos de publicación de memoria.

### 15.13 Hashes (se rellena al cerrar cada paso)

Protocolo de este apartado: `f668ede` (3 de noviembre de 2026, antes de escribir el código y de calcular nada con 2024). Código
(`finales.py`, `28_prevision_final_2024.py`, `29_evaluacion_final_2024.py`): `3f47085`. Ejecución de 28 y de 29: PC del autor,
3 de noviembre de 2026, con el código de `3f47085` sin cambios (28 una vez completo, sin recortes; 29 una vez, sin repetición).
Resultados de 15.14: commit siguiente (hash *pendiente*).

### 15.14 Resultados de 2024 (3 de noviembre de 2026)

**15.14.1 Ejecución y controles.** `28` en un solo terminal, 4,33 h, sin recortes (proyecciones tras el primer reajuste: SARIMAX 0,8 h,
LightGBM 0,8 h, N-HiTS 1,9 h, segunda red 2,2 h; suma 5,6 h frente al presupuesto de 8 h; estimación de 15.10: 4,1 h en serie y 5,3 h
con ×1,3). Tiempos por reajuste: SARIMAX 2,9-3,3 min (FGLS: 5 vueltas y convergencia en los 12 meses; los `ConvergenceWarning` de
statsmodels son de ajustes internos), LightGBM 1,7-4,2 min, N-HiTS 7,1-7,6 min y segunda red 6,7-8,5 min; 9 de las 12 redes oficiales y
10 de las 12 segundas redes llegaron al tope de 1.000 pasos (el resto, 900). Fugas: 0 días en los cuatro modelos y en la segunda red;
pruebas de reajuste (meses 1 y 7) idénticas con la demanda posterior alterada. Rejilla: 0 cruces; reordenación de 0, 0, 8.784 y 7.430
horas (ingenuo, SARIMAX, LightGBM, N-HiTS), el mismo patrón que en 2023 (14.13.3: 8.760 y 7.737). `29` una sola vez, con todos los
controles de 15.7 superados en 23 s (Diebold-Mariano sintético: nivel 5,0 %, potencia 100 %, AR(1) 7,6 %; recursión propia igual a MAPIE
al bit; causalidad de los cuatro modelos en los tres cortes con su control positivo; métricas reproducidas con dos implementaciones;
fugas registradas por 28: 0). Cifras de este apartado: salida del 29 (`data/resultados/salida_29.txt`) y CSV de
`data/resultados/final/`. La consola de Windows mostró caracteres mal codificados; no afecta a las cifras.

**15.14.2 Precisión puntual de 2024** (error = real − previsto; MW; IC 95 % por bootstrap de días).

| | Ingenuo | SARIMAX | LightGBM | N-HiTS |
|---|---|---|---|---|
| MAE | 1.217,9 (1.106,6-1.342,2) | 1.089,8 (983,8-1.195,2) | 573,6 (545,1-604,6) | 665,6 (632,6-699,7) |
| MAE / ingenuo | 1 | 0,895 | 0,471 | 0,547 |
| RMSE | 1.871,7 | 1.616,0 | 770,7 | 875,8 |
| MAPE | 4,56 % | 4,19 % | 2,16 % | 2,54 % |
| MASE | 0,917 | 0,820 | 0,432 | 0,501 |
| Sesgo (no apreciable en ninguno) | −6,7 | +6,0 | +135,0 | −41,1 |
| RMSE / MAE | 1,54 | 1,48 | 1,34 | 1,32 |
| MAE de 2023 y cambio | 1.400,0 (−13,0 %, mejor) | 943,4 (+15,5 %, peor) | 585,2 (−2,0 %, similar) | 716,4 (−7,1 %, similar) |

MAE por mes (ingenuo / SARIMAX / LightGBM / N-HiTS, MW): ene 1.936 / 1.655 / 623 / 797; feb 898 / 689 / 669 / 702; mar 1.273 / 1.282 /
680 / 866; abr 1.042 / 1.441 / 745 / 897; may 906 / 1.048 / 426 / 485; jun 935 / 767 / 562 / 582; jul 1.017 / 780 / 518 / 490; ago
1.698 / 988 / 548 / 499; sep 985 / 804 / 536 / 662; oct 723 / 691 / 571 / 742; nov 1.010 / 1.385 / 445 / 556; dic 2.143 / 1.523 /
566 / 713. Por bloque (B1 00-05 h, B2 06-11, B3 12-17, B4 18-23): ingenuo 745 / 1.265 / 1.649 / 1.212; SARIMAX 825 / 1.118 / 1.304 /
1.112; LightGBM 375 / 578 / 792 / 549; N-HiTS 537 / 678 / 762 / 685. Descriptivo: N-HiTS queda por debajo de LightGBM en julio y agosto
y en B3; SARIMAX queda por encima del ingenuo en marzo, abril y mayo y en B1. Sin contraste y sin explicación (véase 15.14.8).

**15.14.3 Diebold-Mariano** (pérdida absoluta diaria, 366 días, Newey-West con 7 retardos, HLN, t de 365 g. l., Holm sobre 6 pares).
Apreciables (p ajustado < 0,001 salvo indicación): ingenuo frente a LightGBM (644 MW de diferencia media) y frente a N-HiTS (552);
SARIMAX frente a LightGBM (516) y frente a N-HiTS (424); LightGBM frente a N-HiTS (92 MW a favor de LightGBM; DM = −4,55).
**No distinguible: ingenuo frente a SARIMAX** (128 MW a favor de SARIMAX; DM = 1,72; p ajustado 0,087). Robustez: con pérdida cuadrática
la lectura es la misma en los seis pares (ingenuo-SARIMAX, p ajustado 0,063); con retardos 3, 7 y 14 el p ajustado de ingenuo-SARIMAX
es 0,057, 0,087 y 0,120 y los otros cinco pares no cambian de lectura (0,0000).

**15.14.4 Intervalos propios del 90 % y rejilla, 8.784 horas.**

| | Ingenuo | SARIMAX | LightGBM | N-HiTS |
|---|---|---|---|---|
| Cobertura (IC 95 %) | 92,35 % (90,0-94,4) | 93,27 % (91,3-95,2) | 76,09 % (73,7-78,4) | 78,75 % (76,7-80,8) |
| Lectura | sobrecobertura | sobrecobertura | infracobertura | infracobertura |
| Fallos por debajo / por encima | 3,8 % / 3,9 % | 3,5 % / 3,2 % | 10,6 % / 13,3 % | 11,6 % / 9,6 % |
| Cociente fallos | 0,97 | 1,09 | 0,79 | 1,21 |
| Anchura media (MW) | 5.948 | 5.717 | 2.221 | 2.081 |
| IS (MW) | 8.811 | 7.392 | 4.418 | 4.157 |
| CRPS₁₉ (MW) | 912,6 | 826,4 | 420,2 | 485,4 |
| Desviación máx. de fiabilidad | 0,023 | 0,096 | 0,141 | 0,084 |

Cobertura por bloque (B1-B4): ingenuo 93,3 / 91,3 / 91,8 / 93,0; SARIMAX 95,7 / 92,0 / 90,9 / 94,4; LightGBM 78,6 / 76,9 / 68,9 / 80,0;
N-HiTS 82,8 / 78,5 / 74,8 / 78,9 %. Por mes, el mínimo es 74 % (ingenuo, diciembre), 87 % (SARIMAX, diciembre), 62 % (LightGBM,
enero) y 62 % (N-HiTS, abril). Fiabilidad de la rejilla: la frecuencia del real por debajo del cuantil 0,5 es 0,486, 0,506, 0,394 y
0,524 (LightGBM desplazado hacia abajo, N-HiTS centrado pero estrecho: 0,132 en 0,05 y 0,842 en 0,90). Pares (IS y CRPS₁₉): todos
apreciables salvo LightGBM-N-HiTS en IS (+262 MW, IC −146 a +675) y, en CRPS₁₉, el par LightGBM-N-HiTS es apreciable (−65 MW, IC −93 a
−40). Frente a 2023 (14.13.1; solo descriptivo): cobertura propia 88,3 → 92,4 % (ingenuo), 92,5 → 93,3 (SARIMAX), 70,0 → 76,1
(LightGBM) y 68,1 → 78,8 (N-HiTS); la infracobertura de las dos redes se repite (15.8.6).

**15.14.5 Ventana (γ = 0) y ACI (γ = 0,001) con arranque D7** (366 días emitidos).

| | Ingenuo | SARIMAX | LightGBM | N-HiTS |
|---|---|---|---|---|
| Cobertura de la ventana (IC 95 %) | 86,9 % (84,2-89,6), infra | 87,3 % (84,7-89,8), infra | 88,2 % (86,3-89,9), bien | 89,4 % (87,6-91,1), bien |
| Cobertura de ACI (IC 95 %) | 88,0 % (85,4-90,6) | 88,6 % (86,1-91,0) | 89,1 % (87,4-90,7) | 89,4 % (87,7-91,1) |
| IS de la ventana / de ACI | 9.048 / 9.140 | 8.115 / 8.144 | 3.300 / 3.314 | 3.751 / 3.768 |
| CRPS₁₉ de la ventana / de ACI | 926,6 / 926,5 | 824,3 / 825,2 | 398,5 / 398,8 | 469,5 / 469,7 |
| Desviación máx. de fiabilidad con ACI | 0,030 | 0,016 | 0,009 | 0,006 |

ACI frente al intervalo propio (IS, negativo = ACI mejor): ingenuo +329 MW (no distinguible), SARIMAX +752 (apreciable, ACI peor),
LightGBM −1.104 (−25,0 %, apreciable), N-HiTS −388 (−9,3 %, apreciable); en CRPS₁₉, ingenuo y SARIMAX no distinguibles, LightGBM
−21 y N-HiTS −16 (apreciables). **ACI frente a la ventana: sin beneficio** (IS: +92, +29, +15 y +18 MW; solo el de N-HiTS es apreciable
y a favor de la ventana, un 0,5 %; CRPS₁₉: diferencias de 1 MW o menos). La mejora sobre el intervalo propio de las dos redes viene de
recalibrar con ventana, no de la adaptación de α. Sensibilidad: IS(γ)/IS(0) con γ = 0,001 de 1,0102, 1,0035, 1,0045 y 1,0047; con
γ = 0,02 el IS baja en el ingenuo (0,950) y en SARIMAX (0,991) pero hay 396 y 408 horas con intervalo infinito (210 y 156 en LightGBM y
N-HiTS); con γ = 0,001, 0 horas infinitas en los cuatro. Puntuación de 14.6.7 recalculada con 2024 (solo descriptiva, **γ no se
reelige**): 1,0057 (0,001), 1,0275 (0,005), 1,0034 (0,01) y 0,9965 (0,02). Fiabilidad con ACI: todas las desviaciones máximas ≤ 0,030.

**15.14.6 Anticipación de 15.9 frente a lo observado: se cumplen 49 de las 79 comprobaciones y fallan 30** (en 2023, 25 de 41; los
conjuntos no son comparables uno a uno). Se cumplen: el orden por MAE (A5, A6, A7) y por CRPS₁₉ (A24), los MAE de LightGBM y N-HiTS
(A3, A4) y su MAE relativo (A8), el cociente RMSE/MAE (A11), el sesgo de SARIMAX en valor absoluto (A14), cinco de los seis pares de
Diebold-Mariano (A15) y la misma lectura con pérdida cuadrática (A16), la infracobertura de las dos redes (A17 LightGBM y A18),
CRPS₁₉/MAE (A25), la fiabilidad (A26), la cobertura con ACI (A28), A31, A33 y A34, la mejora de CRPS₁₉ con ACI (A35), el IS de ACI
en LightGBM (A29) y todas las de coste (A36, A37). **Fallan:**
 - *Mejor o peor de lo anticipado por modelo:* MAE del ingenuo (1.218 frente a 1.220-1.500, por 2 MW) y su cambio frente a 2023
   (−13 %, A12); MAE de SARIMAX (1.090 frente a 830-1.060), su MAE relativo (0,895 frente a 0,6-0,8) y su cambio (+15,5 %, A12); el
   par SARIMAX-ingenuo en Diebold-Mariano (A15: no distinguible; por tanto, «los seis pares a la vez» también falla).
 - *Sesgo:* los rangos de LightGBM y N-HiTS (A9) y los signos de los tres (A10: LightGBM +135 y N-HiTS −41, al revés; SARIMAX +6);
   A13 (se anticipó que no mejorarían su MAE de 2023; mejoraron un 2,0 % y un 7,1 %).
 - *Intervalos propios:* cobertura del ingenuo (92,4 % frente a 86-91) y de N-HiTS (78,8 % frente a 58-76); los tres cocientes de
   fallos de A19 (SARIMAX 1,09 frente a >1,3; LightGBM 0,79 frente a >1,2; N-HiTS 1,21 frente a <0,9); anchura e IS de SARIMAX (A20,
   A21), IS de N-HiTS (4.157 frente a 4.500-5.600), el orden por IS (A22: N-HiTS delante de LightGBM, sin diferencia apreciable),
   CRPS₁₉ del ingenuo (913 frente a 950-1.120) y de SARIMAX (826 frente a 640-780), y A27 (cobertura de SARIMAX en B1 menos B3:
   4,8 puntos frente a más de 5).
 - *ACI:* A29 en N-HiTS (−9,3 % frente a −15 % a −40 %), A30 en SARIMAX (apreciable, ACI peor, frente a no distinguible) y en el
   ingenuo (peor, pero no distinguible), y A32 (la puntuación de 14.6.7 no es >1 para los cuatro γ y el mínimo no está en γ = 0,001
   sino en 0,02).
Lectura: la anticipación acertó en el orden, la infracobertura de las redes, la utilidad de recalibrar y el coste; se equivocó sobre
SARIMAX (la apuesta más importante: se esperó que el reajuste mensual lo separara con claridad del ingenuo y no lo hizo), sobre el
signo del sesgo de las dos redes y sobre la asimetría de sus fallos.

**15.14.7 Lectura con las reglas de 15.8** (resultados negativos incluidos).
 1. En 2024 el orden de precisión es LightGBM < N-HiTS < SARIMAX < ingenuo, el mismo que en la validación; LightGBM supera a N-HiTS
    de forma apreciable y robusta (pérdida absoluta y cuadrática; retardos 3, 7 y 14), con IC del MAE sin solapamiento. No gana en
    todos los desgloses: N-HiTS tiene menor MAE en julio y agosto y en B3, y menor IS propio sin diferencia apreciable.
 2. **Resultado negativo: SARIMAX no se distingue del ingenuo estacional** en 2024 (p ajustado 0,087 con retardo 7; 0,063 con pérdida
    cuadrática; 0,057-0,120 según el retardo), con +15,5 % de MAE respecto a 2023 mientras el ingenuo mejora un 13 %. El reajuste
    mensual elimina su sesgo (+6 MW) pero no su error.
 3. Intervalos propios: los clásicos sobrecubren (92,4 y 93,3 %) y los de aprendizaje automático infracubren (76,1 y 78,8 %), como
    en 2023 aunque algo menos. Con recalibración la cobertura queda entre 86,9 y 89,4 %, y con ACI entre 88,0 y 89,4 %.
 4. **Resultado negativo: ACI con γ = 0,001 no mejora a la ventana sin adaptación** (diferencias de IS de 0,5 % o menos, no
    distinguibles o ligeramente peores), igual que en 2023 (14.13.4). Se mantiene γ = 0,001 por la regla de 14.6.7, sin reelegir con 2024.
 5. Comparación con 2023: solo descriptiva y con la salvedad de la maldición del ganador (C49); ninguna decisión de 2023 se reabre.

**15.14.8 Lo que no se ha comprobado.** No se explica todavía por qué SARIMAX empeora (sus peores meses son enero, marzo, abril,
noviembre y diciembre; en B1 queda por encima del ingenuo), por qué cambia el signo del sesgo de las dos redes frente a lo
anticipado, ni por qué N-HiTS gana en julio, agosto y B3. Son hipótesis sin comprobar. El desglose queda para el apartado 16 (OE8),
que se fija con commit antes de mirar nada más que lo impreso por el 29. El asistente no ha abierto ningún otro archivo de 2024.

**15.14.9 Para el registro de coherencia del 05** (la memoria no se toca hasta R2), además de C45-C52: C53 SARIMAX no distinguible del
ingenuo en 2024 y su cambio frente a 2023 (5.4, 5.7; afecta a la jerarquía de modelos del texto); C54 ACI sin beneficio sobre la
ventana en 2023 y 2024 (amplía C48; 5.5); C55 tiempos reales del 28 (4,33 h; amplía C50).
