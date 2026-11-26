"""Genera el contrato de los modelos guardados (E4) a partir de los propios archivos .joblib.

Ejecutar desde la carpeta del proyecto (donde están modelos/, resultados/, Entrega_Final_Nacimientos.qmd y la base):
    python generar_contrato.py

Escribe:
    modelos/contrato.json     -> versión para máquinas (la usan cargar_y_predecir.py y app.py)
    docs/contrato_modelo.md   -> versión para personas (va en el repositorio)

Ninguna cifra se escribe a mano: columnas, códigos, medianas, umbrales y métricas salen de los
modelos, de modelos/modelos_e4.json, de resultados/prueba_final.csv y de la base.
"""
import ast
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder

RAIZ = Path(".")
DIR_MOD, DIR_RES, DIR_DOCS = RAIZ / "modelos", RAIZ / "resultados", RAIZ / "docs"
MAESTRO = next((RAIZ / q for q in ["Entrega_Final_Nacimientos.qmd", "articulo_E3.qmd"] if (RAIZ / q).exists()),
               RAIZ / "Entrega_Final_Nacimientos.qmd")
BASES = ["BD-EEVV-Nacimientos-2023.csv", "BD-EEVV-Nacimientos-2023.csv.gz",
         "BD-EEVV-Nacimientos-2023 (1).csv", "BD-EEVV-Nacimientos-2023__1_.csv"]
SEMILLA = 82202


# El modelo de bajo peso se guardó con esta clase (definida en el .qmd). Sin ella, joblib.load falla.
class OneHotRefModa(OneHotEncoder):
    """One-hot que toma como referencia la categoría más frecuente del entrenamiento."""

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.drop = [Xd[c].value_counts().index[0] for c in Xd.columns]
        return super().fit(X, y)


sys.modules["__main__"].OneHotRefModa = OneHotRefModa


# ---------- Lo que se toma del artículo maestro (sin copiarlo a mano) ----------
def _dict_del_maestro(nombre):
    """Lee un diccionario literal definido en el .qmd (p. ej. codigos_sin_info, DESC)."""
    txt = MAESTRO.read_text(encoding="utf-8")
    m = re.search(rf"^{nombre}\s*=\s*\{{", txt, flags=re.M)
    if not m:
        raise KeyError(nombre)
    i, prof = m.end() - 1, 0
    for j in range(i, len(txt)):
        prof += {"{": 1, "}": -1}.get(txt[j], 0)
        if prof == 0:
            return ast.literal_eval(txt[i:j + 1])
    raise ValueError(nombre)


CODIGOS_SIN_INFO = _dict_del_maestro("codigos_sin_info")
DESC = _dict_del_maestro("DESC")
META = json.loads((DIR_MOD / "modelos_e4.json").read_text(encoding="utf-8"))
FIN = pd.read_csv(DIR_RES / "prueba_final.csv").set_index("objetivo")
REQ = dict(l.strip().split("==") for l in (RAIZ / "requirements.txt").read_text().splitlines()
           if "==" in l)

NOMBRE = {"cesarea": "Cesárea", "bajo_peso": "Bajo peso al nacer (< 2.500 g)",
          "tiempo_gestacion": "Tiempo de gestación (semanas aproximadas)"}
ALG = {"rf": "Random Forest", "logit": "Regresión logística"}


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------- 1) Lo que cada modelo espera, leído del Pipeline ajustado ----------
def describir(clave):
    ruta = DIR_MOD / f"modelo_{clave}.joblib"
    pipe = joblib.load(ruta)
    if isinstance(pipe, dict):          # formato con contrato: {"pipeline": ..., "umbral": ..., ...}
        pipe = pipe["pipeline"]
    pre, mod = pipe.named_steps["prep"], pipe.named_steps["modelo"]
    num, cat = {}, {}
    for nombre, tr, cols in pre.transformers_:
        if nombre == "num":
            imp = tr.named_steps["imp"] if hasattr(tr, "named_steps") else tr
            esc = tr.named_steps.get("esc") if hasattr(tr, "named_steps") else None
            for c, med in zip(cols, imp.statistics_):
                num[c] = {"mediana_imputacion": float(med), "escalado": esc is not None}
        elif nombre == "cat":
            imp, ohe = tr.named_steps["imp"], tr.named_steps["ohe"]
            relleno = imp.fill_value
            drop = getattr(ohe, "drop_idx_", None)
            for i, (c, cats) in enumerate(zip(cols, ohe.categories_)):
                cats = [str(x) for x in cats]
                ref = cats[int(drop[i])] if drop is not None and drop[i] is not None else None
                cat[c] = {"codigos": [x for x in cats if x != relleno],
                          "acepta_faltante": relleno in cats, "referencia": ref}
    orden = list(META[clave]["columnas"])
    assert set(orden) == set(num) | set(cat), clave
    tipo = "clasificacion" if hasattr(mod, "predict_proba") else "regresion"
    d = {"archivo": f"modelos/{ruta.name}", "sha256": sha256(ruta),
         "tamano_mb": round(ruta.stat().st_size / 2**20, 2),
         "estimador": type(mod).__name__, "parametros": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                                         for k, v in mod.get_params().items()},
         "tipo": tipo, "columnas": orden, "numericas": num, "categoricas": cat,
         "necesita_clase_OneHotRefModa": any(isinstance(t.named_steps.get("ohe"), OneHotRefModa)
                                             for n, t, _ in pre.transformers_ if n == "cat"),
         "categoria_desconocida": ("vector de ceros (igual que la categoría de referencia)"
                                   if any(v["referencia"] for v in cat.values())
                                   else "vector de ceros (ninguna categoría marcada)")}
    if tipo == "clasificacion":
        d.update({"umbral": float(META[clave]["umbral"]), "modelo": META[clave]["modelo"],
                  "clase_positiva": 1})
    else:
        d.update({"semanas_por_intervalo": META[clave]["semanas_por_intervalo"]})
    return pipe, d


PIPES, CONTRATO = {}, {}
for k in ["cesarea", "bajo_peso", "tiempo_gestacion"]:
    PIPES[k], CONTRATO[k] = describir(k)
COLS = CONTRATO["cesarea"]["columnas"]
assert all(CONTRATO[k]["columnas"] == COLS for k in CONTRATO), "Los modelos piden columnas distintas."


# ---------- 2) Rangos observados y verificación con la prueba (necesita la base) ----------
def leer_base():
    ruta = next((b for b in BASES if (RAIZ / b).exists()), None)
    if ruta is None:
        return None
    df = pd.read_csv(RAIZ / ruta, encoding="cp1252", dtype=str)
    for v, cods in CODIGOS_SIN_INFO.items():
        if v in df:
            df[v] = df[v].mask(df[v].isin(cods))
    return df[df["TIPO_PARTO"].notna() & df["PESO_NAC"].notna()]


def preparar(df):
    """Lo mismo que debe hacer cargar_y_predecir.py: texto, códigos 'sin información' -> nulo,
    numéricas a número."""
    X = df[COLS].astype(object).copy()
    for v, cods in CODIGOS_SIN_INFO.items():
        if v in X:
            X[v] = X[v].mask(X[v].astype(str).isin(cods))
    for c in CONTRATO["cesarea"]["numericas"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


BASE = leer_base()
VERIF = {}
if BASE is not None:
    PRED = pd.read_csv(DIR_RES / "predicciones_prueba.csv")
    ids_prueba = PRED["id_fila"].unique()
    entren = BASE.drop(index=ids_prueba)                 # entrenamiento + validación
    for c in CONTRATO["cesarea"]["numericas"]:
        s = pd.to_numeric(entren[c], errors="coerce")
        rango = {"min": float(s.min()), "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                 "max": float(s.max()), "faltantes": float(s.isna().mean())}
        for k in CONTRATO:
            CONTRATO[k]["numericas"][c]["rango_entrenamiento"] = rango
    for k in ["cesarea", "bajo_peso"]:
        d = PRED[PRED["objetivo"] == k]
        Xp = preparar(BASE.loc[d["id_fila"]])
        p = PIPES[k].predict_proba(Xp)[:, 1]
        yh = (p >= CONTRATO[k]["umbral"]).astype(int)
        VERIF[k] = {"registros": int(len(d)),
                    "dif_max_probabilidad": float(np.abs(p - d["probabilidad"].to_numpy()).max()),
                    "clases_iguales": float((yh == d["prediccion"].to_numpy()).mean())}
    Xg = preparar(BASE.loc[ids_prueba])
    sg = PIPES["tiempo_gestacion"].predict(Xg)
    VERIF["tiempo_gestacion"] = {"registros": int(len(Xg)), "min": float(sg.min()), "max": float(sg.max())}
    EJEMPLO = preparar(BASE.loc[ids_prueba[:3]])


# ---------- 3) Desempeño de referencia (la única evaluación en prueba) ----------
for k in ["cesarea", "bajo_peso"]:
    f = FIN.loc[k]
    CONTRATO[k]["desempeno_prueba"] = {m: float(f[m]) for m in
                                       ["roc_auc", "accuracy", "precision", "recall", "f1"]}
    CONTRATO[k]["hiperparametros"] = json.loads(f["hiperparametros"])
    CONTRATO[k]["n_ajuste_final"] = int(f["n_ajuste_final"])
    CONTRATO[k]["prevalencia_entrenamiento"] = (
        float((BASE.drop(index=ids_prueba)["TIPO_PARTO"] == "2").mean()) if k == "cesarea" and BASE is not None
        else float(BASE.drop(index=ids_prueba)["PESO_NAC"].isin(["1", "2", "3", "4"]).mean())
        if BASE is not None else None)

GENERAL = {"version_contrato": "1.0", "generado": date.today().isoformat(),
           "semilla": SEMILLA, "versiones": {k: REQ.get(k) for k in ["scikit-learn", "pandas", "numpy", "joblib"]},
           "version_sklearn_usada_al_generar": sklearn.__version__,
           "momento_de_uso": "última consulta prenatal (antes del parto)",
           "codigos_sin_informacion": {c: v for c, v in CODIGOS_SIN_INFO.items() if c in COLS},
           "columnas": COLS, "verificacion": VERIF}
(DIR_MOD / "contrato.json").write_text(json.dumps({"general": GENERAL, "modelos": CONTRATO},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- 4) Versión en Markdown ----------
def md_tabla(filas, cab):
    s = ["| " + " | ".join(cab) + " |", "|" + "|".join([":--"] * len(cab)) + "|"]
    s += ["| " + " | ".join(str(x) for x in f) + " |" for f in filas]
    return "\n".join(s)


def num_fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ")


L = []
a = L.append
a("# Contrato de los modelos guardados (E4)\n")
a(f"> Generado automáticamente por `generar_contrato.py` el {GENERAL['generado']} a partir de los archivos "
  "`.joblib`, `modelos/modelos_e4.json`, `resultados/prueba_final.csv` y la base. No editar a mano: si cambia "
  "un modelo, se vuelve a ejecutar el script. La versión para programas es `modelos/contrato.json`.\n")
a("Este contrato dice qué recibe cada modelo, en qué formato, qué devuelve y qué pasa con los datos faltantes "
  "o desconocidos. Quien use los modelos (`cargar_y_predecir.py`, `app.py`) debe cumplirlo.\n")

a("## 1. Archivos\n")
a(md_tabla([[NOMBRE[k], f"`{c['archivo']}`", ALG.get(c.get("modelo"), "Regresión lineal múltiple"),
             c["tamano_mb"], f"`{c['sha256'][:16]}…`"] for k, c in CONTRATO.items()],
           ["Objetivo", "Archivo", "Modelo", "MB", "SHA-256 (inicio)"]))
a("\nCada archivo es un diccionario (ver sección 2) cuyo `pipeline` es un `Pipeline` completo de scikit-learn "
  "(preprocesamiento + modelo): recibe los datos crudos ya limpios de códigos y hace solo la imputación y la "
  "codificación. El SHA-256 completo está en `modelos/contrato.json` y sirve para comprobar que el archivo no cambió.\n")

a("## 2. Cómo cargarlos\n")
a("Cada `.joblib` guarda un **diccionario con el contrato del modelo**. Se carga con solo `joblib` y `pandas`, "
  "sin definir ninguna clase propia:\n")
a("```python\nimport joblib\nimport pandas as pd\n\n"
  "m = joblib.load(\"modelos/modelo_bajo_peso.joblib\")\n"
  "pipe, umbral = m[\"pipeline\"], m[\"umbral\"]\n"
  "proba = pipe.predict_proba(datos[m[\"columnas\"]])[:, 1]      # datos: texto, con 99/999/9 ya en nulo\n"
  "clase = (proba >= umbral).astype(int)\n```\n")
a(md_tabla([["`pipeline`", "Pipeline ajustado (preprocesamiento + modelo)"],
            ["`version_modelo`", "Versión del conjunto de modelos"],
            ["`versions`", "Versiones de Python, scikit-learn, pandas, numpy y joblib con que se guardó"],
            ["`columnas`", "Las 16 columnas de entrada, en el orden del entrenamiento"],
            ["`umbral`", "Umbral de clasificación fijado en la CV (`None` en la regresión)"],
            ["`metrica_prueba`", "Métricas de la única evaluación en prueba (AUC, exactitud, precisión, recall y F1; RECM, MAE y R² en la regresión)"],
            ["`prevalencia_entrenamiento`", "Proporción de la clase positiva en entrenamiento más validación (`None` en la regresión)"],
            ["`entrenado_el`", "Fecha del modelo"],
            ["`objetivo`, `tipo`", "Objetivo (`cesarea`, `bajo_peso`, `tiempo_gestacion`) y tipo de problema"],
            ["`hiperparametros`, `n_ajuste_final`, `clase_positiva`", "Solo en los clasificadores"],
            ["`semanas_por_intervalo`", "Solo en la regresión: semanas usadas por código de `T_GES`"]],
           ["Clave", "Contenido"]))
a("\nVersiones con las que se guardaron (deben ser las mismas para cargar sin advertencias): "
  + ", ".join(f"{k} {v}" for k, v in GENERAL["versiones"].items()) + ". "
  "`utils_modelos.py` ofrece `cargar_modelo`, `leer_base` y `preparar_datos` (la misma limpieza que el artículo).\n")

a("## 3. Entrada\n")
a(f"**Momento de uso.** {GENERAL['momento_de_uso'].capitalize()}. Por eso los modelos no usan `T_GES` "
  "(edad gestacional al nacer) ni `MES` (mes del nacimiento), ni ninguna variable del parto o del recién nacido.\n")
a("**Formato.** Una tabla (CSV o `DataFrame`) con una fila por gestante y, como mínimo, las "
  f"{len(COLS)} columnas de abajo con estos nombres exactos (en mayúsculas). Las columnas de más se ignoran; "
  "si falta una, el modelo da error. Los tres modelos piden las mismas columnas. **Todo se lee como texto** "
  "(`dtype=str`, codificación `cp1252` como la base del DANE): varios códigos llevan cero a la izquierda "
  "(`NIV_EDUM` = `01`, `CODPTORE` = `05`) y si se leen como número (`1`, `5`) el modelo no los reconoce. "
  "Después, solo las numéricas se convierten a número.\n")
a("### 3.1 Numéricas\n")
filas = []
for c, v in CONTRATO["cesarea"]["numericas"].items():
    r = v.get("rango_entrenamiento", {})
    filas.append([f"`{c}`", DESC.get(c, ("",))[0], ", ".join(CODIGOS_SIN_INFO.get(c, [])) or "—",
                  num_fmt(v["mediana_imputacion"], 0),
                  f"{num_fmt(r['min'], 0)} a {num_fmt(r['max'], 0)} ({num_fmt(r['p01'], 0)} a {num_fmt(r['p99'], 0)})" if r else "—"])
a(md_tabla(filas, ["Columna", "Descripción", "Código «sin información»", "Mediana de imputación",
                   "Rango en entrenamiento: mín. a máx. (p1 a p99)"]))
a("\nUn valor fuera del rango de entrenamiento no da error, pero el modelo extrapola: `cargar_y_predecir.py` "
  "debe avisarlo. La logística y la regresión lineal estandarizan estas columnas con la media y la desviación "
  "del entrenamiento (guardadas dentro del Pipeline); el bosque no.\n")
a("### 3.2 Categóricas: códigos que el modelo conoce\n")
filas = []
for c, v in CONTRATO["cesarea"]["categoricas"].items():
    ref = CONTRATO["bajo_peso"]["categoricas"][c]["referencia"]
    filas.append([f"`{c}`", DESC.get(c, ("",))[0], " ".join(f"`{x}`" for x in v["codigos"]),
                  ", ".join(CODIGOS_SIN_INFO.get(c, [])) or "—", f"`{ref}`"])
a(md_tabla(filas, ["Columna", "Descripción", "Códigos aceptados", "Código «sin información»",
                   "Referencia (logística y RLM)"]))
a("\nLos códigos son los del diccionario de datos del DANE (Estadísticas Vitales 2023, catálogo 876 de "
  "microdatos). La referencia es la categoría más frecuente del entrenamiento: los coeficientes de la "
  "logística y de la regresión lineal se leen frente a ella.\n")

a("## 4. Datos faltantes y códigos desconocidos\n")
a("1. **Códigos de «sin información».** Antes de predecir se convierten en nulo los códigos de la tabla "
  "anterior (por ejemplo `99` en `NUMCONSUL` o `999` en `EDAD_PADRE`). Si no se convierten, el modelo los toma "
  "como un valor real: 99 consultas o un padre de 999 años.\n"
  "2. **Nulos.** Se envían como celda vacía o `NaN`, nunca como texto (`\"sin_info\"`, `\"NA\"`). El Pipeline "
  "imputa la mediana del entrenamiento en las numéricas y una categoría propia de «sin información» en las "
  "categóricas, aprendida en el entrenamiento.\n"
  "3. **Códigos que el modelo no vio.** No dan error: `handle_unknown=\"ignore\"` los codifica como un vector "
  "de ceros. En el bosque eso es «ninguna categoría»; en la logística y en la regresión lineal equivale, sin "
  "aviso, a la categoría de referencia. Por eso `cargar_y_predecir.py` debe revisar cada código contra la "
  "lista de la sección 3.2 y marcar la fila (columna `advertencias`) si alguno no está.\n"
  "4. **Filas con muchos faltantes.** Se predicen igual, pero la predicción se apoya en las medianas y en la "
  "categoría «sin información»; conviene informar cuántas columnas venían vacías.\n")

a("## 5. Salidas\n")
filas = []
for k in ["cesarea", "bajo_peso"]:
    c = CONTRATO[k]
    filas.append([NOMBRE[k], "probabilidad entre 0 y 1 de la clase 1", f"{c['umbral']:.2f}",
                  "1 si probabilidad ≥ umbral; 0 en otro caso",
                  "cesárea" if k == "cesarea" else "bajo peso (< 2.500 g)"])
c = CONTRATO["tiempo_gestacion"]
filas.append([NOMBRE["tiempo_gestacion"], "semanas aproximadas (número continuo)", "—", "—", "—"])
a(md_tabla(filas, ["Objetivo", "Salida", "Umbral", "Clase", "Clase 1 significa"]))
a("\nLos umbrales se fijaron con los costos de los errores en la validación cruzada "
  "(cesárea: falso positivo y falso negativo cuestan lo mismo; bajo peso: un caso no detectado cuesta 5 veces "
  "una falsa alarma) y **no se cambian** al usar el modelo. Para planear recursos de cesárea se debe usar la "
  "suma de las probabilidades (número esperado), no el conteo de casos marcados (ver la auditoría de equidad).\n")
a("La regresión lineal se entrenó con un valor representativo por intervalo de `T_GES`, así que su salida no "
  "son semanas exactas:\n")
a(md_tabla([[k, v] for k, v in c["semanas_por_intervalo"].items()],
           ["Código de `T_GES`", "Semanas usadas en el entrenamiento"]))
a("\nSalida mínima que debe escribir `cargar_y_predecir.py` por fila: `id` (si la entrada lo trae), "
  "`prob_cesarea`, `clase_cesarea`, `prob_bajo_peso`, `clase_bajo_peso`, `semanas_aprox`, `advertencias`, y "
  "la versión del contrato.\n")

a("## 6. Desempeño de referencia\n")
a("Medido una sola vez en la prueba (20 % de la base, no usada para elegir nada), con el umbral de la tabla "
  "anterior:\n")
a(md_tabla([[NOMBRE[k], ALG[CONTRATO[k]["modelo"]],
             ", ".join(f"{a_}={b}" for a_, b in CONTRATO[k]["hiperparametros"].items()) or "sin hiperparámetros",
             num_fmt(CONTRATO[k]["n_ajuste_final"], 0),
             f"{CONTRATO[k]['desempeno_prueba']['roc_auc']:.3f}", f"{CONTRATO[k]['desempeno_prueba']['recall']:.3f}",
             f"{CONTRATO[k]['desempeno_prueba']['precision']:.3f}",
             f"{CONTRATO[k]['prevalencia_entrenamiento'] * 100:.1f} %" if CONTRATO[k]["prevalencia_entrenamiento"] else "—"]
            for k in ["cesarea", "bajo_peso"]],
           ["Objetivo", "Modelo", "Hiperparámetros", "Registros de ajuste", "AUC", "Recall", "Precisión",
            "Prevalencia (entren. + valid.)"]))
a("\nSi en datos nuevos el AUC cae claramente por debajo de estas cifras o la prevalencia cambia mucho, el "
  "modelo deja de cumplir lo que promete (ver el plan de monitoreo). Las métricas de la regresión lineal "
  "están en el artículo.\n")

a("## 7. Verificación del contrato\n")
if VERIF:
    a("Al generar este documento, el script cargó los tres modelos guardados, preparó los registros de prueba "
      "con las reglas de la sección 4 y comparó con `resultados/predicciones_prueba.csv`:\n")
    for k in ["cesarea", "bajo_peso"]:
        v = VERIF[k]
        a(f"- {NOMBRE[k]}: {num_fmt(v['registros'], 0)} registros; diferencia máxima de probabilidad "
          f"{v['dif_max_probabilidad']:.2e}; clases iguales en {v['clases_iguales'] * 100:.2f} % de los casos.")
    v = VERIF["tiempo_gestacion"]
    a(f"- {NOMBRE['tiempo_gestacion']}: {num_fmt(v['registros'], 0)} registros; predicciones entre "
      f"{v['min']:.1f} y {v['max']:.1f} semanas.\n")
    a("Ejemplo de entrada ya preparada (tres registros de prueba) y su salida:\n")
    ej = EJEMPLO.copy()
    sal = pd.DataFrame({"prob_cesarea": PIPES["cesarea"].predict_proba(ej)[:, 1],
                        "prob_bajo_peso": PIPES["bajo_peso"].predict_proba(ej)[:, 1],
                        "semanas_aprox": PIPES["tiempo_gestacion"].predict(ej)})
    sal["clase_cesarea"] = (sal["prob_cesarea"] >= CONTRATO["cesarea"]["umbral"]).astype(int)
    sal["clase_bajo_peso"] = (sal["prob_bajo_peso"] >= CONTRATO["bajo_peso"]["umbral"]).astype(int)
    a(md_tabla([[("" if pd.isna(x) else (f"{x:g}" if isinstance(x, float) else x)) for x in r]
                for r in ej.itertuples(index=False)], list(ej.columns)))
    a("")
    a(md_tabla([[f"{r.prob_cesarea:.3f}", r.clase_cesarea, f"{r.prob_bajo_peso:.3f}", r.clase_bajo_peso,
                 f"{r.semanas_aprox:.1f}"] for r in sal.itertuples()],
               ["prob_cesarea", "clase_cesarea", "prob_bajo_peso", "clase_bajo_peso", "semanas_aprox"]))
else:
    a("No se encontró la base, así que no se pudo verificar contra la prueba.")
a("\n## 8. Fuera del contrato\n")
a("- No sirve para datos de otro año sin volver a validarlo, ni para registros tomados después del parto.\n"
  "- No es una herramienta para decidir sobre una paciente en particular: es una alerta (bajo peso) y un "
  "insumo de planeación (cesárea). El uso previsto y no recomendado está en la ficha del modelo.\n"
  "- `IDPERTET` (pertenencia étnica) es uno de los predictores; la auditoría de equidad documenta lo que eso "
  "implica.\n")

DIR_DOCS.mkdir(exist_ok=True)
(DIR_DOCS / "contrato_modelo.md").write_text("\n".join(L), encoding="utf-8")
print("Escritos: modelos/contrato.json y docs/contrato_modelo.md")
for k, v in VERIF.items():
    print(k, v)
