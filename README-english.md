# evalcards

[![PyPI version](https://img.shields.io/pypi/v/evalcards?logo=pypi&label=PyPI)](https://pypi.org/project/evalcards/)
[![Python versions](https://img.shields.io/pypi/pyversions/evalcards?logo=python&label=Python)](https://pypi.org/project/evalcards/)
[![Wheel](https://img.shields.io/pypi/wheel/evalcards?label=wheel)](https://pypi.org/project/evalcards/#files)
[![License](https://img.shields.io/pypi/l/evalcards?label=License)](https://pypi.org/project/evalcards/)
[![CI](https://github.com/Ricardouchub/evalcards/actions/workflows/ci.yml/badge.svg)](https://github.com/Ricardouchub/evalcards/actions/workflows/ci.yml)
[![Publish](https://github.com/Ricardouchub/evalcards/actions/workflows/release.yml/badge.svg)](https://github.com/Ricardouchub/evalcards/actions/workflows/release.yml)

evalcards is a small Python library that generates evaluation reports for supervised models in Markdown, including metrics and ready-to-use plots. It supports:
- Classification: binary, multiclass (OvR) with metrics such as `accuracy`, `balanced_accuracy`, `mcc`, `log_loss` (if probabilities supplied), `roc_auc`/`pr_auc`, and ROC/PR curves per class.
- Regression: `MAE`, `MSE`, `RMSE`, `R²`, `MedAE`, `MAPE`, `RMSLE`.
- Forecasting (time series): `MAE`, `MSE`, `RMSE`, `MedAE`, `MAPE`, `RMSLE`, `sMAPE (%)`, and `MASE`.
- Multi-label classification: confusion matrices and ROC/PR curves per label if probabilities are provided.
- JSON export of metrics and image paths for CI/pipeline integration (new in v0.2.11).

## Install

```bash
pip install evalcards
```

## Quick start

```python
from evalcards import make_report

# y_true: true labels/values
# y_pred: predicted labels/values
# y_proba (optional):
#   - binary: 1D vector with positive-class probability
#   - multiclass: matrix (n_samples, n_classes) with class probabilities
#   - multi-label: matrix (n_samples, n_labels) with probabilities per label

path = make_report(
    y_true, y_pred,
    y_proba=proba,                 # optional
    path="report.md",              # markdown filename
    title="My model"               # report title
)
print(path)  # path to generated report
```

## What evaluates

- Classification (binary / multiclass / multi-label)
  Metrics: `accuracy`, `precision/recall/F1` (macro/weighted), `balanced_accuracy`, `mcc`, `log_loss` (if probabilities).
  AUC / AUPRC: `roc_auc` and `pr_auc` (binary), `roc_auc_ovr_macro` and `pr_auc_macro` (multiclass), `roc_auc_macro` (multi-label).
  Plots: confusion matrix, ROC and PR curves (per class in multiclass, per label in multi-label).

- Regression
  Metrics: `MAE`, `MSE`, `RMSE`, `R²`, `MedAE`, `MAPE`, `RMSLE`.
  Plots: fit (y vs ŷ) and residuals.

- Forecasting
  Metrics: `MAE`, `MSE`, `RMSE`, `MedAE`, `MAPE`, `RMSLE`, `sMAPE (%)`, `MASE`.
  Extra params: `season` (e.g. 12) and `insample` (training series for MASE).
  Plots: fit and residuals.

## Examples

**1) Binary classification (scikit-learn)**

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

make_report(y_te, y_pred, y_proba=proba, path="rep_bin.md", title="Binary classification")
```

**2) Multiclass (OvR)**

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
    labels=[f"Class_{c}" for c in clf.classes_],  # optional
    path="rep_multi.md", title="Multiclass OvR"
)
```

**3) Multi-label**

```python
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from evalcards import make_report

X, y = make_multilabel_classification(n_samples=300, n_features=12, n_classes=4, n_labels=2, random_state=42)
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(X, y)
y_pred = clf.predict(X)
# Probabilities per label (n_samples x n_labels)
y_proba = np.stack([m.predict_proba(X)[:,1] for m in clf.estimators_], axis=1)

make_report(y, y_pred, y_proba=y_proba, path="rep_multilabel.md", title="Multi-label Example", lang="en",
            labels=[f"Tag_{i}" for i in range(y.shape[1])])
```

**4) Regression**

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from evalcards import make_report

X, y = make_regression(n_samples=600, n_features=8, noise=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

reg = RandomForestRegressor(random_state=0).fit(X_tr, y_tr)
y_pred = reg.predict(X_te)

make_report(y_te, y_pred, path="rep_reg.md", title="Regression")
```

**5) Forecasting (sMAPE / MASE + extra metrics)**

```python
import numpy as np
from evalcards import make_report

rng = np.random.default_rng(0)
t = np.arange(360)
y = 10 + 0.05*t + 5*np.sin(2*np.pi*t/12) + rng.normal(0,1,360)

y_train, y_test = y[:300], y[300:]
y_hat = y_test + rng.normal(0, 1.2, y_test.size)  # example prediction

make_report(
    y_test, y_hat,
    task="forecast", season=12, insample=y_train,
    path="rep_forecast.md", title="Forecast"
)
```

## Outputs and PATH

- A Markdown file with metrics and references to generated images.
- PNG images depending on the task:
  - Binary classification:
    - `confusion.png` (global confusion matrix)
    - `roc.png` (ROC curve)
    - `pr.png` (Precision-Recall curve)
  - Multiclass (OvR):
    - `confusion.png` (global confusion matrix)
    - `roc_class_<class>.png` (ROC per class, One-vs-Rest)
    - `pr_class_<class>.png` (PR per class)
  - Multi-label:
    - `confusion_<label>.png` (confusion per label)
    - `roc_label_<label>.png` (ROC per label, if probabilities provided)
    - `pr_label_<label>.png` (PR per label, if probabilities provided)
  - Regression / Forecasting:
    - `fit.png` (scatter y vs ŷ)
    - `resid.png` (residuals)

- File location:
  - By default, files are saved into `./evalcards_reports/` when `path` has no directory.
  - You can change target directory with `out_dir` or by providing a path that includes a directory.

- JSON export (optional):
  If you request `export_json`, a `.json` file with `metrics` and `charts` (image names/paths) will also be generated.

- Example multi-label naming:
  With `labels=["A","B","C"]` you will get:
  - `confusion_A.png`, `roc_label_A.png`, `pr_label_A.png`, etc.

- JSON structure: contains `metrics`, `charts` and `markdown`.

## Expected inputs

- Classification
  - `y_true`: integers 0..K-1 (or string labels).
  - `y_pred`: same type/space as `y_true`.
  - `y_proba` (optional):
    - Binary: 1D vector with positive-class probability.
    - Multiclass: matrix `(n_samples, n_classes)` with one column per class (same order as your model).
    - Multi-label: matrix `(n_samples, n_labels)` with one column per label.
- Regression / Forecast
  - `y_true`, `y_pred`: 1D float arrays.
  - `insample` (forecast): training series for MASE; `season` for seasonality (e.g. 12 monthly).

## Model compatibility

Works with any model that exposes `predict` (and optionally `predict_proba`):
- scikit-learn, XGBoost/LightGBM/CatBoost, statsmodels, Prophet/NeuralProphet, Keras/PyTorch, etc.
- For multiclass, supply `y_proba` as a matrix (one column per class) and use `labels` to provide class names.

## Roadmap

### v0.3 — Output & key metrics
- [ ] Self-contained HTML report (`format="md|html"`)
- [x] JSON export of metrics/paths (`--export-json`)
- [x] New classification metrics: AUPRC, Balanced Accuracy, MCC, Log Loss
- [x] New regression metrics: MAPE, MedAE, RMSLE

### v0.4 — Multiclass & thresholds
- [ ] ROC/PR micro & macro (multiclass) + `roc_auc_macro`, `average_precision_macro`
- [ ] Threshold analysis (precision/recall/F1 vs threshold + best threshold per metric)
- [ ] Normalized confusion matrix (global and per-class)

### v0.5 — Probabilities & comparisons
- [ ] Calibration: Brier score + reliability diagram
- [ ] Multi-model comparison in a single report ("best per metric" table)
- [ ] Gain / lift curves

### v0.6 — DX, formats & docs
- [ ] New input formats: Parquet/Feather/NPZ
- [ ] Project config (`.evalcards.toml`) for defaults (outdir, title, language)
- [ ] Docs with MkDocs + GitHub Pages (guide, API, runnable examples)
- [ ] Jinja2 templates / themes (branding)

### Ideas
- [x] Multi-label support (completed)
- [ ] Ranking metrics (MAP/NDCG)
- [ ] Calibration curves with configurable bins
- [ ] QQ-plot and residual histogram (regression)
- [x] i18n EN/ES (completed)

## Documentation

**[Guide](docs/index.md)** | **[API reference](docs/api.md)** | **[Changelog](CHANGELOG.md)**

## License

MIT

## Author

**Ricardo Urdaneta**

**[Linkedin](https://www.linkedin.com/in/ricardourdanetacastro)**
