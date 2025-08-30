evalcards
================
[![PyPI version](https://img.shields.io/pypi/v/evalcards?logo=pypi&label=PyPI)](https://pypi.org/project/evalcards/)
[![Python versions](https://img.shields.io/pypi/pyversions/evalcards?logo=python&label=Python)](https://pypi.org/project/evalcards/)
[![Wheel](https://img.shields.io/pypi/wheel/evalcards?label=wheel)](https://pypi.org/project/evalcards/#files)
[![License](https://img.shields.io/pypi/l/evalcards?label=License)](https://pypi.org/project/evalcards/)
[![CI](https://github.com/Ricardouchub/evalcards/actions/workflows/ci.yml/badge.svg)](https://github.com/Ricardouchub/evalcards/actions/workflows/ci.yml)
[![Publish](https://github.com/Ricardouchub/evalcards/actions/workflows/release.yml/badge.svg)](https://github.com/Ricardouchub/evalcards/actions/workflows/release.yml)

`evalcards` genera reportes de evaluación para **modelos supervisados** en **Markdown**, con **métricas** y **gráficos** listos para pegar en informes. Soporta:
- **Clasificación**: binaria y **multiclase (OvR)** con curvas **ROC/PR** por clase.
- **Regresión**.
- **Forecasting** (series de tiempo): **sMAPE (%)** y **MASE**.

Documentación
-----------

- 📚 **Guía completa**: [docs/index.md](docs/index.md)
- 🔧 **Referencia de API**: [docs/api.md](docs/api.md)
- 📝 **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- 🐞 **Incidencias / ideas**: [Issues](https://github.com/Ricardouchub/evalcards/issues)

Instalación
-----------
```bash
pip install evalcards
```

Uso rápido (Python)
-------------------
```python
from evalcards import make_report

# y_true: etiquetas/valores reales
# y_pred: etiquetas/valores predichos
# y_proba (opcional):
#   - binaria: vector 1D con prob. de la clase positiva
#   - multiclase: matriz (n_samples, n_classes) con prob. por clase

path = make_report(
    y_true, y_pred,
    y_proba=proba,                 # opcional
    path="reporte.md",             # nombre del archivo Markdown
    title="Mi modelo"              # título del reporte
)
print(path)  # ruta del reporte generado
```

Qué puedes evaluar
------------------
- **Clasificación (binaria/multiclase)**  
  Métricas: `accuracy`, `precision/recall/F1` (macro/weighted),  
  AUC: `roc_auc` (binaria) y `roc_auc_ovr_macro` (multiclase).  
  Gráficos: **matriz de confusión**, **ROC** y **PR** (por clase en multiclase).

- **Regresión**  
  Métricas: `MAE`, `MSE`, `RMSE`, `R²`.  
  Gráficos: **Ajuste (y vs ŷ)** y **Residuales**.

- **Forecasting**  
  Métricas: `MAE`, `MSE`, `RMSE`, **sMAPE (%)**, **MASE**.  
  Parámetros extra: `season` (p.ej. 12) e `insample` (serie de entrenamiento para MASE).  
  Gráficos: **Ajuste** y **Residuales**.

Ejemplos breves
---------------
**1) Clasificación binaria (scikit-learn)**
```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = make_classification(n_samples=600, n_features=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
proba = clf.predict_proba(X_te)[:, 1]

make_report(y_te, y_pred, y_proba=proba, path="rep_bin.md", title="Clasificación binaria")
```

**2) Multiclase (OvR)**
```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
proba = clf.predict_proba(X_te)  # (n_samples, n_classes)

make_report(
    y_te, y_pred, y_proba=proba,
    labels=[f"Clase_{c}" for c in clf.classes_],   # opcional (nombres por clase)
    path="rep_multi.md", title="Multiclase OvR"
)
```

**3) Regresión**
```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = make_regression(n_samples=600, n_features=8, noise=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

reg = RandomForestRegressor(random_state=0).fit(X_tr, y_tr)
y_pred = reg.predict(X_te)

make_report(y_te, y_pred, path="rep_reg.md", title="Regresión")
```

**4) Forecasting (sMAPE/MASE)**
```python
import numpy as np
from evalcards import make_report

rng = np.random.default_rng(0)
t = np.arange(360)
y = 10 + 0.05*t + 5*np.sin(2*np.pi*t/12) + rng.normal(0,1,360)

y_train, y_test = y[:300], y[300:]
y_hat = y_test + rng.normal(0, 1.2, y_test.size)  # predicción de ejemplo

make_report(
    y_test, y_hat,
    task="forecast", season=12, insample=y_train,
    path="rep_forecast.md", title="Forecast"
)
```

Salidas y ubicación
-------------------
- Un archivo **Markdown** con las métricas y referencias a imágenes.
- Imágenes **PNG** (confusión, ROC/PR, ajuste, residuales).
- Por defecto, si `path` no incluye carpeta, todo se guarda en `./evalcards_reports/`.  
  Puedes cambiar la carpeta con el argumento `out_dir` o usando una ruta en `path`.

Entradas esperadas (formas comunes)
-----------------------------------
- **Clasificación**
  - `y_true`: enteros 0..K-1 (o etiquetas string).
  - `y_pred`: del mismo tipo/espacio de clases que `y_true`.
  - `y_proba` (opcional):
    - **Binaria**: vector 1D con prob. de la clase positiva.
    - **Multiclase**: matriz `(n_samples, n_classes)` con una columna por clase (mismo orden que tu modelo).
- **Regresión / Forecast**
  - `y_true`, `y_pred`: arrays 1D de floats.
  - `insample` (forecast): serie de entrenamiento para MASE; `season` según la estacionalidad (ej. 12 mensual/anual).

Compatibilidad (modelos)
------------------------
Funciona con **cualquier modelo** que produzca `predict` (y opcionalmente `predict_proba`):
- scikit-learn, XGBoost/LightGBM/CatBoost, statsmodels, Prophet/NeuralProphet, Keras/PyTorch (si pasas tus arrays).
- Multiclase: pasa `y_proba` como matriz (una columna por clase) y, si quieres, `labels` para nombres.

Notas técnicas
--------------
- El backend de Matplotlib se fuerza a **Agg** (no requiere GUI).
- Reportes en **Markdown** para fácil copia/pegado en documentos.

Licencia
--------
MIT

Autor
-----
**Ricardo Urdaneta**

**[Linkedin](https://www.linkedin.com/in/ricardourdanetacastro)**

