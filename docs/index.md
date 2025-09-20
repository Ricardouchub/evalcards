# evalcards — Guía completa

`evalcards` genera **reportes de evaluación** en **Markdown** con **métricas** y **gráficos** para:
- **Clasificación**: binaria y **multiclase (One-vs-Rest)** con métricas `accuracy`, `balanced_accuracy`, `mcc`, `log_loss` (si hay probabilidades), `roc_auc`/`pr_auc` y curvas **ROC/PR** por clase.
- **Regresión**: `MAE`, `MSE`, `RMSE`, `R²`, `MedAE`, `MAPE`, `RMSLE`.
- **Forecasting** (series de tiempo): `MAE`, `MSE`, `RMSE`, `MedAE`, `MAPE`, `RMSLE`, **sMAPE (%)** y **MASE**.
- **Clasificación multi-label**: métricas, matriz de confusión por etiqueta y curvas **ROC/PR por etiqueta** si se pasan probabilidades.

Los reportes incluyen tablas con métricas y PNGs listos para insertar en informes o PRs.

---

## Índice
- [Instalación](#instalación)
- [Quickstart (Python)](#quickstart-python)
- [Conceptos y salidas](#conceptos-y-salidas)
- [Casos de uso](#casos-de-uso)
  - [Clasificación binaria](#clasificación-binaria)
  - [Clasificación multiclase (OvR)](#clasificación-multiclase-ovr)
  - [Clasificación multi-label](#clasificación-multi-label)
  - [Regresión](#regresión)
  - [Forecasting](#forecasting)
- [Export JSON (integración)](#export-json-integración)
- [Soporte de idioma](#soporte-de-idioma)
- [Buenas prácticas y troubleshooting](#buenas-prácticas-y-troubleshooting)
- [Limitaciones actuales](#limitaciones-actuales)
- [Versionado y compatibilidad](#versionado-y-compatibilidad)
- [Licencia](#licencia)
- [Referencia de API](#referencia-de-api)

---

## Instalación

```bash
pip install evalcards
```

Requisitos:
- **Python ≥ 3.9**
- Dependencias principales (instaladas automáticamente): `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jinja2`.

Verifica versión instalada:
```bash
python -c "from importlib.metadata import version; print(version('evalcards'))"
# o
python -c "import evalcards; print(getattr(evalcards, '__version__', 'unknown'))"
```

---

## Quickstart (Python)

```python
from evalcards import make_report

# y_true: etiquetas/valores reales
# y_pred: etiquetas/valores predichos
# y_proba (opcional):
#   - binaria: vector 1D con prob. de la clase positiva
#   - multiclase: matriz (n_samples, n_classes) con prob. por clase

path = make_report(
    y_true, y_pred,
    y_proba=proba,     # opcional
    path="reporte.md",
    export_json="info.json"  # << opcional: genera también info.json y devuelve (path, info)
)
print(path)  # -> ruta del Markdown generado
```

---

## Conceptos y salidas

- **Entrada mínima**: arrays `y_true` y `y_pred` (mismo largo).  
- **Probabilidades** (`y_proba`) opcionales:
  - **Binaria**: vector 1D con prob. de la clase positiva.
  - **Multiclase**: matriz `(n_samples, n_classes)` con una columna por clase (mismo orden que tu modelo).
  - **Multi-label**: matriz `(n_samples, n_labels)` con probabilidad de cada etiqueta.
- **Salida**:
  - Un archivo **Markdown** (por defecto `report.md`) con la tabla de métricas y referencias a imágenes.
  - **PNGs**:
    - Clasificación: `confusion.png` y (si hay probabilidades) `roc*.png`, `pr*.png`.  
      Multiclase: `roc_class_<clase>.png`, `pr_class_<clase>.png` **por clase**.
      Multi-label: `confusion_<etiqueta>.png`, `roc_label_<etiqueta>.png`, `pr_label_<etiqueta>.png` **por etiqueta**.
    - Regresión/Forecasting: `fit.png` (y vs ŷ) y `resid.png` (residuales).
- **Ubicación**:
  - Si `path` **no** incluye carpeta, todo se guarda en `./evalcards_reports/`.  
  - Puedes fijar carpeta con `out_dir` o pasando una **ruta completa** en `path`.
- **JSON** (opcional): 
Estructura típica del JSON:
```json
{
  "metrics": { ... },
  "charts": { "confusion": "confusion.png", "roc": [...], "pr": [...] },
  "markdown": "report.md"
}
```
---

## Casos de uso

### Clasificación binaria

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = make_classification(n_samples=600, n_features=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
y_pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]  # prob. de la clase positiva

make_report(yte, y_pred, y_proba=proba, path="rep_bin.md", title="Clasificación binaria")
```

Incluye: `accuracy`, `precision/recall/F1` (macro/weighted), `balanced_accuracy`, `mcc`, `log_loss` (si hay probabilidades), **AUC ROC** (`roc_auc`) y **AUPRC** (`pr_auc`), además de curvas **ROC/PR**.

---

### Clasificación multiclase (OvR)

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RandomForestClassifier(random_state=0).fit(Xtr, ytr)
y_pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)  # (n_samples, n_classes)

make_report(
    yte, y_pred, y_proba=proba,
    labels=[f"Clase_{c}" for c in clf.classes_],  # opcional
    path="rep_multi.md", title="Multiclase OvR"
)
```

Incluye: métricas macro/weighted, `balanced_accuracy`, `mcc`, `log_loss` (si hay probabilidades), **AUC macro OvR** (`roc_auc_ovr_macro`), **AUPRC macro** (`pr_auc_macro`) y **curvas ROC/PR por clase**.

---

### Clasificación multi-label
```python
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from evalcards import make_report

X, y = make_multilabel_classification(n_samples=300, n_features=12, n_classes=4, n_labels=2, random_state=42)
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
y_pred = clf.predict(X)
# Probabilidad por etiqueta (matriz n_samples x n_labels)
y_proba = np.stack([m.predict_proba(X)[:,1] for m in clf.estimators_], axis=1)

make_report(y, y_pred, y_proba=y_proba, path="rep_multilabel.md", title="Multi-label Example", lang="en",
            labels=[f"Tag_{i}" for i in range(y.shape[1])])
```
Genera una tabla de métricas multi-label (subset accuracy, hamming loss, F1/precision/recall macro y micro), una matriz de confusión por etiqueta y, si se pasan probabilidades (`y_proba` 2D), curvas ROC y PR por etiqueta (`roc_label_<etiqueta>.png`, `pr_label_<etiqueta>.png`).

---

### Regresión

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = make_regression(n_samples=600, n_features=8, noise=10, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

reg = RandomForestRegressor(random_state=0).fit(Xtr, ytr)
y_pred = reg.predict(Xte)

make_report(yte, y_pred, path="rep_reg.md", title="Regresión")
```

Incluye: `MAE`, `MSE`, `RMSE`, `R²`, `MedAE`, `MAPE`, `RMSLE` + gráficos de **ajuste** y **residuales**.

---

### Forecasting

```python
import numpy as np
from evalcards import make_report

rng = np.random.default_rng(0)
t = np.arange(360)
y = 10 + 0.05*t + 5*np.sin(2*np.pi*t/12) + rng.normal(0,1,360)

y_train, y_test = y[:300], y[300:]
y_hat = y_test + rng.normal(0, 1.2, y_test.size)  # ejemplo de predicción

make_report(
    y_test, y_hat,
    task="forecast", season=12, insample=y_train,
    path="rep_forecast.md", title="Forecast"
)
```

Incluye: `MAE`, `MSE`, `RMSE`, `MedAE`, `MAPE`, `RMSLE`, **sMAPE (%)**, **MASE** + gráficos.

---

## Export JSON (integración)

- Para pipelines y CI, puedes solicitar un JSON con las métricas y las rutas de los PNGs generados.
- Ejemplo en Python:
```python
md_path, info = make_report(y_true, y_pred, path="rep.md", export_json="rep.json")
# info["metrics"], info["charts"], info["markdown"]
```
- Ejemplo CLI:
```bash
evalcards --y_true y_true.csv --y_pred y_pred.csv --out report.md --export-json report.json
```
- Nota: si pasas `export_json` como un path absoluto, el JSON se escribirá allí; si pasas solo un nombre, se colocará en la carpeta de salida resuelta.

---


## Soporte de idioma

Puedes generar reportes en español o inglés con el parámetro `lang`:

```python
from evalcards import make_report

make_report(y_true, y_pred, path="reporte.md", title="Mi modelo", lang="es")
make_report(y_true, y_pred, path="report_en.md", title="My Model", lang="en")
```

También en CLI:

```bash
evalcards --y_true y_true.csv --y_pred y_pred.csv --lang en --out report_en.md
```
---

## Buenas prácticas y troubleshooting

- **Probabilidades**:
  - Binaria: `y_proba` como vector 1D (prob. de la clase positiva).
  - Multiclase: matriz `(n_samples, n_classes)`; cuida el **orden de columnas** (usa `clf.classes_` en scikit-learn).
  - Multi-label: matriz `(n_samples, n_labels)`; cada columna corresponde a la probabilidad de cada etiqueta.
- **Gráficos sin GUI**:
  - Guardado a PNG no requiere GUI. Si tu entorno no tiene backend gráfico, puedes forzar:
    - Variable: `MPLBACKEND=Agg`
    - O antes de importar `pyplot`:
      ```python
      import matplotlib
      matplotlib.use("Agg")
      ```
- **Rendimiento**:
  - En datasets muy grandes, los *scatter* pueden ser pesados. Considera muestrear puntos.
- **Errores típicos**:
  - *Shape mismatch*: `y_true` y `y_pred` deben tener la misma longitud.
  - Probabilidades inválidas: deben estar en `[0,1]`; en multiclase, las filas deberían sumar ~1.
  - Clases/nombres: si pasas `labels`, su longitud debe ser `n_classes`.

---

## Limitaciones actuales

- En multiclase: AUC macro OvR y curvas por clase (no micro/macro PR/ROC agregadas por ahora).
- Sin métricas de ranking (MAP/NDCG) ni calibración (Brier/curva de calibración) por ahora.

---

## Versionado y compatibilidad

- Soporta **Python 3.9 – 3.13**.
- Sigue SemVer de forma aproximada (patch = fixes, minor = features, major = *breaking changes*).
- Consulta `CHANGELOG.md` para detalles por versión.

---

## Licencia

**MIT** — © Ricardo Urdaneta.

---

## Referencia de API

Consulta la [Referencia de API](api.md) para firmas, tipos y parámetros detallados.