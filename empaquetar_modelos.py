"""Guarda cada modelo de modelos/ como un diccionario con su contrato (sin reentrenar nada).

    python empaquetar_modelos.py

Cada .joblib queda como un dict con:
    pipeline, version_modelo, versions, columnas, umbral, metrica_prueba,
    prevalencia_entrenamiento, entrenado_el      (más objetivo, tipo y, según el caso, hiperparámetros)

El pipeline se carga con solo joblib: la clase propia OneHotRefModa del artículo se cambia por el
OneHotEncoder de scikit-learn que ya traía ajustado (mismas categorías y referencias), de modo que
los resultados son idénticos. Se comprueba contra la predicción anterior antes de escribir.
Se puede volver a ejecutar sin problema (si el archivo ya es un dict, lo reescribe igual).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder

RAIZ = Path(__file__).resolve().parent
DIR_MOD, DIR_RES = RAIZ / "modelos", RAIZ / "resultados"
VERSION_MODELO = "1.0"
ENTRENADO_EL = "2026-11-18"          # fecha de los modelos (la misma de la ficha del modelo)
COMPRESION = {"cesarea": ("xz", 3), "bajo_peso": 3, "tiempo_gestacion": 3}   # el de cesárea pesaba ~110 MB


class OneHotRefModa(OneHotEncoder):
    """Solo para poder leer los .joblib antiguos (se guardaron con esta clase en __main__)."""

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.drop = [Xd[c].value_counts().index[0] for c in Xd.columns]
        return super().fit(X, y)


sys.modules["__main__"].OneHotRefModa = OneHotRefModa


def a_encoder_estandar(pipe):
    """Cambia OneHotRefModa por OneHotEncoder (en los pasos ajustados y en los sin ajustar)."""
    n = 0
    pre = pipe.named_steps["prep"]
    for lista in (getattr(pre, "transformers", []), getattr(pre, "transformers_", [])):
        for _, tr, _ in lista:
            ohe = tr.named_steps.get("ohe") if hasattr(tr, "named_steps") else None
            if isinstance(ohe, OneHotRefModa):
                ohe.__class__ = OneHotEncoder
                n += 1
    return n


def preparar(df, columnas, numericas, sin_info):
    X = df[columnas].astype(object).copy()
    for v, cods in sin_info.items():
        if v in X:
            X[v] = X[v].mask(X[v].astype(str).isin(cods))
    for c in numericas:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


C = json.loads((DIR_MOD / "contrato.json").read_text(encoding="utf-8"))
SIN_INFO = C["general"]["codigos_sin_informacion"]
COLS = C["general"]["columnas"]
NUM = [c for c in COLS if c in C["modelos"]["cesarea"]["numericas"]]
PRED = pd.read_csv(DIR_RES / "predicciones_prueba.csv")
ids = PRED["id_fila"].unique()
BASE = None
for nombre in ["BD-EEVV-Nacimientos-2023.csv", "BD-EEVV-Nacimientos-2023.csv.gz"]:
    if (RAIZ / nombre).exists():
        BASE = pd.read_csv(RAIZ / nombre, encoding="cp1252", dtype=str)
        break
assert BASE is not None, "Falta la base del DANE en la carpeta del proyecto."
BASE = BASE[BASE["TIPO_PARTO"].ne("9") & BASE["PESO_NAC"].ne("9") & BASE["TIPO_PARTO"].notna() & BASE["PESO_NAC"].notna()]
versions = {"python": ".".join(map(str, sys.version_info[:3])), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}

for clave in ["cesarea", "bajo_peso", "tiempo_gestacion"]:
    ruta = DIR_MOD / f"modelo_{clave}.joblib"
    obj = joblib.load(ruta)
    pipe = obj["pipeline"] if isinstance(obj, dict) else obj
    m = C["modelos"][clave]
    Xp = preparar(BASE.loc[PRED.loc[PRED["objetivo"] == "cesarea", "id_fila"]] if clave != "tiempo_gestacion"
                  else BASE.loc[ids], COLS, NUM, SIN_INFO)
    ref = pipe.predict_proba(Xp)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(Xp)
    cambios = a_encoder_estandar(pipe)
    nuevo = pipe.predict_proba(Xp)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(Xp)
    assert np.allclose(ref, nuevo, atol=1e-12), f"{clave}: el cambio de clase altera la predicción"
    d = {"pipeline": pipe, "version_modelo": VERSION_MODELO, "versions": versions, "columnas": list(COLS),
         "objetivo": clave, "tipo": m["tipo"], "entrenado_el": ENTRENADO_EL}
    if m["tipo"] == "clasificacion":
        d.update(umbral=float(m["umbral"]), metrica_prueba=m["desempeno_prueba"],
                 prevalencia_entrenamiento=m["prevalencia_entrenamiento"],
                 hiperparametros=m["hiperparametros"], n_ajuste_final=m["n_ajuste_final"], clase_positiva=1)
    else:
        sem = {k: float(v) for k, v in m["semanas_por_intervalo"].items()}
        s = BASE.loc[ids]
        y = s["T_GES"].map(sem)
        ok = y.notna().to_numpy()
        p = pipe.predict(preparar(s, COLS, NUM, SIN_INFO))[ok]
        y = y[ok].to_numpy(float)
        sce = float(np.sum((y - p) ** 2))
        d.update(umbral=None, semanas_por_intervalo=sem,
                 metrica_prueba={"recm": float(np.sqrt(sce / len(y))), "mae": float(np.mean(np.abs(y - p))),
                                 "r2": float(1 - sce / np.sum((y - y.mean()) ** 2)), "registros": int(len(y))},
                 prevalencia_entrenamiento=None)
    joblib.dump(d, ruta, compress=COMPRESION[clave])
    print(f"{clave}: ok ({cambios} codificadores cambiados, {ruta.stat().st_size / 2**20:.1f} MB)")
