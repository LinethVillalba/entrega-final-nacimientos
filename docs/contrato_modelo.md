# Contrato de los modelos guardados (E4)

> Generado automáticamente por `generar_contrato.py` el 2026-11-25 a partir de los archivos `.joblib`, `modelos/modelos_e4.json`, `resultados/prueba_final.csv` y la base. No editar a mano: si cambia un modelo, se vuelve a ejecutar el script. La versión para programas es `modelos/contrato.json`.

Este contrato dice qué recibe cada modelo, en qué formato, qué devuelve y qué pasa con los datos faltantes o desconocidos. Quien use los modelos (`cargar_y_predecir.py`, `app.py`) debe cumplirlo.

## 1. Archivos

| Objetivo | Archivo | Modelo | MB | SHA-256 (inicio) |
|:--|:--|:--|:--|:--|
| Cesárea | `modelos/modelo_cesarea.joblib` | Random Forest | 17.75 | `6913cea47290f109…` |
| Bajo peso al nacer (< 2.500 g) | `modelos/modelo_bajo_peso.joblib` | Regresión logística | 0.0 | `6b212ff8868ec1f8…` |
| Tiempo de gestación (semanas aproximadas) | `modelos/modelo_tiempo_gestacion.joblib` | Regresión lineal múltiple | 0.0 | `d2a6fce99c174d1c…` |

Cada archivo es un diccionario (ver sección 2) cuyo `pipeline` es un `Pipeline` completo de scikit-learn (preprocesamiento + modelo): recibe los datos crudos ya limpios de códigos y hace solo la imputación y la codificación. El SHA-256 completo está en `modelos/contrato.json` y sirve para comprobar que el archivo no cambió.

## 2. Cómo cargarlos

Cada `.joblib` guarda un **diccionario con el contrato del modelo**. Se carga con solo `joblib` y `pandas`, sin definir ninguna clase propia:

```python
import joblib
import pandas as pd

m = joblib.load("modelos/modelo_bajo_peso.joblib")
pipe, umbral = m["pipeline"], m["umbral"]
proba = pipe.predict_proba(datos[m["columnas"]])[:, 1]      # datos: texto, con 99/999/9 ya en nulo
clase = (proba >= umbral).astype(int)
```

| Clave | Contenido |
|:--|:--|
| `pipeline` | Pipeline ajustado (preprocesamiento + modelo) |
| `version_modelo` | Versión del conjunto de modelos |
| `versions` | Versiones de Python, scikit-learn, pandas, numpy y joblib con que se guardó |
| `columnas` | Las 16 columnas de entrada, en el orden del entrenamiento |
| `umbral` | Umbral de clasificación fijado en la CV (`None` en la regresión) |
| `metrica_prueba` | Métricas de la única evaluación en prueba (AUC, exactitud, precisión, recall y F1; RECM, MAE y R² en la regresión) |
| `prevalencia_entrenamiento` | Proporción de la clase positiva en entrenamiento más validación (`None` en la regresión) |
| `entrenado_el` | Fecha del modelo |
| `objetivo`, `tipo` | Objetivo (`cesarea`, `bajo_peso`, `tiempo_gestacion`) y tipo de problema |
| `hiperparametros`, `n_ajuste_final`, `clase_positiva` | Solo en los clasificadores |
| `semanas_por_intervalo` | Solo en la regresión: semanas usadas por código de `T_GES` |

Versiones con las que se guardaron (deben ser las mismas para cargar sin advertencias): scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, joblib 1.6.0. `utils_modelos.py` ofrece `cargar_modelo`, `leer_base` y `preparar_datos` (la misma limpieza que el artículo).

## 3. Entrada

**Momento de uso.** Última consulta prenatal (antes del parto). Por eso los modelos no usan `T_GES` (edad gestacional al nacer) ni `MES` (mes del nacimiento), ni ninguna variable del parto o del recién nacido.

**Formato.** Una tabla (CSV o `DataFrame`) con una fila por gestante y, como mínimo, las 16 columnas de abajo con estos nombres exactos (en mayúsculas). Las columnas de más se ignoran; si falta una, el modelo da error. Los tres modelos piden las mismas columnas. **Todo se lee como texto** (`dtype=str`, codificación `cp1252` como la base del DANE): varios códigos llevan cero a la izquierda (`NIV_EDUM` = `01`, `CODPTORE` = `05`) y si se leen como número (`1`, `5`) el modelo no los reconoce. Después, solo las numéricas se convierten a número.

### 3.1 Numéricas

| Columna | Descripción | Código «sin información» | Mediana de imputación | Rango en entrenamiento: mín. a máx. (p1 a p99) |
|:--|:--|:--|:--|:--|
| `NUMCONSUL` | Número de consultas prenatales | 99 | 7 | 0 a 25 (0 a 13) |
| `N_HIJOSV` | Número de hijos nacidos vivos de la madre | 99 | 2 | 1 a 17 (1 a 6) |
| `N_EMB` | Número de embarazos de la madre | 99 | 2 | 1 a 20 (1 a 7) |
| `EDAD_PADRE` | Edad del padre en años | 999 | 30 | 12 a 82 (17 a 53) |

Un valor fuera del rango de entrenamiento no da error, pero el modelo extrapola: `cargar_y_predecir.py` debe avisarlo. La logística y la regresión lineal estandarizan estas columnas con la media y la desviación del entrenamiento (guardadas dentro del Pipeline); el bosque no.

### 3.2 Categóricas: códigos que el modelo conoce

| Columna | Descripción | Códigos aceptados | Código «sin información» | Referencia (logística y RLM) |
|:--|:--|:--|:--|:--|
| `SEXO` | Sexo del recién nacido | `1` `2` `3` | — | `1` |
| `MUL_PARTO` | Tipo de embarazo: simple, doble u otro múltiple | `1` `2` `3` `4` | 9 | `1` |
| `IDPERTET` | Pertenencia étnica registrada (variable sensible) | `1` `2` `3` `4` `5` `6` | 9 | `6` |
| `EDAD_MADRE` | Edad de la madre agrupada en rangos (código DANE) | `1` `2` `3` `4` `5` `6` `7` `8` `9` | 99 | `3` |
| `EST_CIVM` | Estado conyugal de la madre | `1` `2` `3` `4` `5` `6` | 9 | `1` |
| `NIV_EDUM` | Nivel educativo de la madre | `01` `02` `03` `04` `05` `06` `07` `08` `09` `10` `11` `12` `13` | 99 | `04` |
| `ULTCURMAD` | Último curso aprobado por la madre | `0` `1` `10` `11` `12` `13` `2` `3` `4` `5` `6` `7` `8` `9` | 99 | `11` |
| `CODPTORE` | Departamento de residencia de la madre | `05` `08` `11` `13` `15` `17` `18` `19` `20` `23` `25` `27` `41` `44` `47` `50` `52` `54` `63` `66` `68` `70` `73` `76` `81` `85` `86` `88` `91` `94` `95` `97` `99` | — | `11` |
| `AREA_RES` | Área de residencia: cabecera, centro poblado o rural disperso | `1` `2` `3` | — | `1` |
| `SEG_SOCIAL` | Régimen de seguridad social en salud: contributivo, subsidiado, excepción, especial o no asegurado | `1` `2` `3` `4` `5` | 9 | `2` |
| `NIV_EDUP` | Nivel educativo del padre | `01` `02` `03` `04` `05` `06` `07` `08` `09` `10` `11` `12` `13` | 99 | `04` |
| `ULTCURPAD` | Último curso aprobado por el padre | `0` `1` `10` `11` `12` `13` `2` `3` `4` `5` `6` `7` `8` `9` | 99 | `11` |

Los códigos son los del diccionario de datos del DANE (Estadísticas Vitales 2023, catálogo 876 de microdatos). La referencia es la categoría más frecuente del entrenamiento: los coeficientes de la logística y de la regresión lineal se leen frente a ella.

## 4. Datos faltantes y códigos desconocidos

1. **Códigos de «sin información».** Antes de predecir se convierten en nulo los códigos de la tabla anterior (por ejemplo `99` en `NUMCONSUL` o `999` en `EDAD_PADRE`). Si no se convierten, el modelo los toma como un valor real: 99 consultas o un padre de 999 años.
2. **Nulos.** Se envían como celda vacía o `NaN`, nunca como texto (`"sin_info"`, `"NA"`). El Pipeline imputa la mediana del entrenamiento en las numéricas y una categoría propia de «sin información» en las categóricas, aprendida en el entrenamiento.
3. **Códigos que el modelo no vio.** No dan error: `handle_unknown="ignore"` los codifica como un vector de ceros. En el bosque eso es «ninguna categoría»; en la logística y en la regresión lineal equivale, sin aviso, a la categoría de referencia. Por eso `cargar_y_predecir.py` debe revisar cada código contra la lista de la sección 3.2 y marcar la fila (columna `advertencias`) si alguno no está.
4. **Filas con muchos faltantes.** Se predicen igual, pero la predicción se apoya en las medianas y en la categoría «sin información»; conviene informar cuántas columnas venían vacías.

## 5. Salidas

| Objetivo | Salida | Umbral | Clase | Clase 1 significa |
|:--|:--|:--|:--|:--|
| Cesárea | probabilidad entre 0 y 1 de la clase 1 | 0.50 | 1 si probabilidad ≥ umbral; 0 en otro caso | cesárea |
| Bajo peso al nacer (< 2.500 g) | probabilidad entre 0 y 1 de la clase 1 | 0.16 | 1 si probabilidad ≥ umbral; 0 en otro caso | bajo peso (< 2.500 g) |
| Tiempo de gestación (semanas aproximadas) | semanas aproximadas (número continuo) | — | — | — |

Los umbrales se fijaron con los costos de los errores en la validación cruzada (cesárea: falso positivo y falso negativo cuestan lo mismo; bajo peso: un caso no detectado cuesta 5 veces una falsa alarma) y **no se cambian** al usar el modelo. Para planear recursos de cesárea se debe usar la suma de las probabilidades (número esperado), no el conteo de casos marcados (ver la auditoría de equidad).

La regresión lineal se entrenó con un valor representativo por intervalo de `T_GES`, así que su salida no son semanas exactas:

| Código de `T_GES` | Semanas usadas en el entrenamiento |
|:--|:--|
| 1 | 21.0 |
| 2 | 24.5 |
| 3 | 32.5 |
| 4 | 39.5 |
| 5 | 42.0 |

Salida mínima que debe escribir `cargar_y_predecir.py` por fila: `id` (si la entrada lo trae), `prob_cesarea`, `clase_cesarea`, `prob_bajo_peso`, `clase_bajo_peso`, `semanas_aprox`, `advertencias`, y la versión del contrato.

## 6. Desempeño de referencia

Medido una sola vez en la prueba (20 % de la base, no usada para elegir nada), con el umbral de la tabla anterior:

| Objetivo | Modelo | Hiperparámetros | Registros de ajuste | AUC | Recall | Precisión | Prevalencia (entren. + valid.) |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Cesárea | Random Forest | n_estimators=100, min_samples_leaf=25, max_features=0.3 | 407 020 | 0.723 | 0.623 | 0.676 | 50.2 % |
| Bajo peso al nacer (< 2.500 g) | Regresión logística | sin hiperparámetros | 407 020 | 0.684 | 0.293 | 0.276 | 11.0 % |

Si en datos nuevos el AUC cae claramente por debajo de estas cifras o la prevalencia cambia mucho, el modelo deja de cumplir lo que promete (ver el plan de monitoreo). Las métricas de la regresión lineal están en el artículo.

## 7. Verificación del contrato

Al generar este documento, el script cargó los tres modelos guardados, preparó los registros de prueba con las reglas de la sección 4 y comparó con `resultados/predicciones_prueba.csv`:

- Cesárea: 101 756 registros; diferencia máxima de probabilidad 4.44e-16; clases iguales en 100.00 % de los casos.
- Bajo peso al nacer (< 2.500 g): 101 756 registros; diferencia máxima de probabilidad 1.11e-16; clases iguales en 100.00 % de los casos.
- Tiempo de gestación (semanas aproximadas): 101 756 registros; predicciones entre 30.6 y 41.7 semanas.

Ejemplo de entrada ya preparada (tres registros de prueba) y su salida:

| SEXO | NUMCONSUL | MUL_PARTO | IDPERTET | EDAD_MADRE | EST_CIVM | NIV_EDUM | ULTCURMAD | CODPTORE | AREA_RES | N_HIJOSV | N_EMB | SEG_SOCIAL | EDAD_PADRE | NIV_EDUP | ULTCURPAD |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | 7 | 1 | 6 | 4 | 1 | 09 | 5 | 11 | 1 | 1 | 1 | 1 | 31 | 09 | 5 |
| 2 | 9 | 1 | 6 | 3 | 2 | 03 | 8 | 11 | 1 | 1 | 1 | 1 | 42 | 03 | 7 |
| 2 | 8 | 1 | 6 | 3 | 1 | 08 | 3 | 15 | 1 | 1 | 1 | 1 | 25 | 08 | 3 |

| prob_cesarea | clase_cesarea | prob_bajo_peso | clase_bajo_peso | semanas_aprox |
|:--|:--|:--|:--|:--|
| 0.470 | 0 | 0.113 | 0 | 37.5 |
| 0.440 | 0 | 0.111 | 0 | 38.1 |
| 0.283 | 0 | 0.092 | 0 | 38.2 |

## 8. Fuera del contrato

- No sirve para datos de otro año sin volver a validarlo, ni para registros tomados después del parto.
- No es una herramienta para decidir sobre una paciente en particular: es una alerta (bajo peso) y un insumo de planeación (cesárea). El uso previsto y no recomendado está en la ficha del modelo.
- `IDPERTET` (pertenencia étnica) es uno de los predictores; la auditoría de equidad documenta lo que eso implica.
