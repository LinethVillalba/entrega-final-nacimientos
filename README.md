# Entrega final: predicción de cesárea, bajo peso al nacer y tiempo de gestación (DANE, EEVV 2023)

**Machine Learning · NRC 1484 · Universidad Tecnológica de Bolívar · 2026-II**
Profesor: Jorge Luis Villalba Acevedo

**Autoras:**

* Luisa Fernanda Acosta Romero - T00082202
* Isabel Sofia Buelvas de la Hoz - T00077937
* María José Espitaleta Julio - T00038486
* Lineth Villalba Dominiquett - T00083377

## Pregunta del trabajo

¿Qué tanto se pueden anticipar, con información previa al parto, la cesárea, el bajo peso al nacer (menos de 2 500 g) y el tiempo de gestación a partir de las Estadísticas Vitales 2023 del DANE?

## Qué hay en este repositorio

```
.
├── Entrega_Final_Nacimientos.qmd    Artículo completo y código (fuente única; Quarto + Python)
├── Entrega_Final_Nacimientos.pdf    Artículo ya renderizado
├── BD-EEVV-Nacimientos-2023.csv     Base del DANE (nacimientos 2023)
├── referencias.bib, apa.csl         Referencias y estilo APA
├── requirements.txt                 Versiones exactas de las librerías
├── LICENSE                          Licencia MIT del código
├── resultados/                      Resultados del render final (ver abajo)
├── modelos/                         Modelos finales guardados (.joblib) y sus umbrales
├── utils_modelos.py                 Carga de los modelos y limpieza de registros (usar siempre)
└── modelos/, resultados/            Modelos finales y tablas del render
```

Todo el código del análisis está dentro del `.qmd`: al renderizarlo se ejecuta y genera las tablas, las figuras, `resultados/` y `modelos/`.

## Datos

- **Fuente:** DANE, Estadísticas Vitales (EEVV), nacimientos 2023, de libre acceso. Archivo descargado el 13 de noviembre de 2026.
- **Tamaño:** 515 549 registros y 39 variables. Tras la limpieza quedan 508 776 registros y 16 predictores previos al parto.
- **Códigos 9, 99 y 999:** se tratan como nulos y se imputan dentro del *pipeline*, para no filtrar información de la prueba.
- **Privacidad:** son datos públicos y anonimizados; el trabajo no intenta identificar personas.

## Método

- **Partición estratificada 60/20/20** (entrenamiento, validación y prueba, semilla 82202).
- **Validación cruzada anidada** de 5 pliegues externos sobre entrenamiento más validación (407 020 registros), con 3 pliegues internos para hiperparámetros y umbral. Los pliegues se crean una vez y son los mismos para todos los modelos.
- **Seis clasificadores:** regresión logística (línea base), árbol de decisión, KNN, Random Forest, SVM (RBF) y red neuronal (MLP con parada temprana y regularización L2).
- **Tiempo de gestación:** regresión lineal múltiple, con Ridge y Lasso como comparación.
- **Umbral por costos:** cesárea (FP = 1, FN = 1) y bajo peso (FP = 1, FN = 5).
- **Prueba:** se usa una sola vez, con un modelo por objetivo.
- **Clustering y PCA:** K-means (k = 7, elegido por silueta), contraste con Ward, y PCA solo para interpretar y dibujar.

## Modelos finales y resultados en prueba

| Objetivo | Modelo | Resultado en prueba |
|---|---|---|
| Cesárea | Random Forest (100 árboles, umbral 0.50) | AUC 0.723; reduce el costo 31.8 % frente a la regla trivial |
| Bajo peso al nacer | Regresión logística (umbral 0.16) | AUC 0.684; detecta 29.3 % de los casos; reduce el costo 13.9 % |
| Tiempo de gestación | Regresión lineal múltiple | RECM 3.089 semanas; R² 0.067 (línea base: 3.197) |

La señal de estas variables es moderada: seis modelos distintos quedan cerca de un AUC de 0.70. Es una alerta para priorizar el control prenatal, **no** una herramienta para decidir sobre una gestante en particular. El AUC en bajo peso va de 0.619 a 0.704 según el subgrupo (peor en madres de 10 a 14 años y en no aseguradas).

La variable `T_GES` (tiempo de gestación) no se conoce antes del parto, así que no se usa en los modelos finales; solo sirve como cota superior.

**Sobre el Random Forest final de cesárea.** Las tablas de validación cruzada muestran `max_features = sqrt` y `min_samples_leaf = 5` en cada pliegue. El modelo guardado en `modelos/modelo_cesarea.joblib` tiene `n_estimators = 100`, `max_features = 0.3` y `min_samples_leaf = 25`. No es una contradicción: como describe el artículo, al elegir el modelo final se repite la búsqueda sobre toda la base de validación cruzada (entrenamiento más validación, 407 020 registros) y el modelo se reentrena con esos datos, por lo que los hiperparámetros pueden diferir de los de los pliegues. Quedan registrados en `resultados/prueba_final.csv`.

## Cómo usar los modelos guardados

Cada archivo de `modelos/` es un *pipeline* de scikit-learn (preprocesamiento más modelo). `modelos/modelos_e4.json` trae las columnas y el umbral de cada modelo.

Use siempre `utils_modelos.py`: (1) el modelo de bajo peso usa una clase propia que `joblib.load` directo no encuentra, y (2) los registros deben limpiarse igual que en el artículo (los códigos de "sin información", como `99`, pasan a nulos). Con eso, las predicciones coinciden con `resultados/predicciones_prueba.csv`.

```python
from utils_modelos import cargar_modelo, cargar_meta, leer_base, preparar_datos

meta = cargar_meta()["cesarea"]
modelo = cargar_modelo("modelos/modelo_cesarea.joblib")

datos = preparar_datos(leer_base(nrows=5))      # registros con las columnas del DANE, en texto
proba = modelo.predict_proba(datos[meta["columnas"]])[:, 1]
alerta = proba >= meta["umbral"]
```

Para `bajo_peso` se usa `modelo_bajo_peso.joblib` y su umbral (0.16). El modelo de tiempo de gestación (`modelo_tiempo_gestacion.joblib`) devuelve las semanas estimadas con `predict`.

> `modelo_cesarea.joblib` está comprimido (xz) para que quepa en GitHub (13 MB). Se carga igual con `cargar_modelo`. Es el mismo modelo: las probabilidades coinciden con las originales.

## Cómo reproducir el artículo

1. Instalar Python 3.13 y [Quarto](https://quarto.org/docs/get-started/).
2. En la carpeta del repositorio:

   ```bash
   pip install -r requirements.txt
   quarto install tinytex
   quarto render Entrega_Final_Nacimientos.qmd --to pdf
   ```

3. Dejar el equipo conectado y sin suspender: **sin resultados previos en caché, el render tarda varias horas**. Al final aparece `Entrega_Final_Nacimientos.pdf`.

Usar las versiones exactas de `requirements.txt` (sobre todo scikit-learn 1.9.1) para que las cifras salgan iguales. No cambiar `MUESTRA_DESARROLLO` (debe seguir en `None`). Durante el render se crea la carpeta `resultados/cache/`, que no se incluye en el repositorio.

## Contenido de `resultados/`

| Archivo | Qué contiene |
|---|---|
| `pliegues.csv` | Asignación de cada registro a su pliegue |
| `cv_resultados.csv` | Métricas de la validación cruzada por modelo y pliegue |
| `cv_hiperparametros.csv` | Hiperparámetros elegidos en cada pliegue |
| `oof_probabilidades.csv.gz` | Probabilidades fuera de pliegue (para el umbral por costos) |
| `predicciones_prueba.csv` | Predicciones del conjunto de prueba |
| `prueba_final.csv` | Métricas finales en prueba de cada modelo elegido |

## Licencias y créditos

- **Datos:** DANE, Estadísticas Vitales, de libre acceso. Citar la fuente al reutilizarlos.
- **Código:** licencia MIT (ver el archivo `LICENSE`).

## Referencias principales

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324
- Almeida, A. V. A., Santos, A. O. R., & Nascimento, L. F. C. (2026). Predictive modeling with machine learning for low birth weight in the state of São Paulo: A retrospective population-based study. *São Paulo Medical Journal, 144*(3). https://doi.org/10.1590/1516-3180.2025.3373.04052026

La lista completa está en el artículo (`Entrega_Final_Nacimientos.pdf`) y en `referencias.bib`.
