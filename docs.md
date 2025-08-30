evalcards — Documentación completa
====================================

[![PyPI version](https://img.shields.io/pypi/v/evalcards?logo=pypi&label=PyPI)](https://pypi.org/project/evalcards/)
[![Python versions](https://img.shields.io/pypi/pyversions/evalcards?logo=python&label=Python)](https://pypi.org/project/evalcards/)
[![CI](https://github.com/Ricardouchub/evalcards/actions/workflows/ci.yml/badge.svg)](https://github.com/Ricardouchub/evalcards/actions/workflows/ci.yml)

`evalcards` genera **reportes de evaluación** en **Markdown** con **métricas** y **gráficos** para:
- **Clasificación**: binaria y **multiclase (One-vs-Rest)** con curvas **ROC** y **PR** por clase.
- **Regresión**.
- **Forecasting** (series de tiempo): **sMAPE (%)** y **MASE**.

Los reportes incluyen tablas con métricas y PNGs listos para insertar en informes o PRs.

Índice
------
1. Instalación
2. Conceptos y salidas
3. Uso rápido (Python)
4. Casos de uso
   - Clasificación binaria
   - Clasificación multiclase (OvR)
   - Regresión
   - Forecasting
5. Referencia de API
6. Detalles de métricas
7. Buenas prácticas y troubleshooting
8. Limitaciones actuales
9. Versionado y compatibilidad
10. Contribuir
11. Licencia

Instalación
-----------
    pip install evalcards

Requisitos:
- Python ≥ 3.9
- Dependencias principales (instaladas automáticamente): numpy, pandas, scikit-learn, matplotlib, jinja2.

Verifica versión instalada:
    python -c "from importlib.metadata import version; print(version('evalcards'))"
    # o
    python -c "import evalcards; print(getattr(evalcards, '__version__', 'unknown'))"

Conceptos y salidas
-------------------
- Entrada mínima: arrays y_true y y_pred.
- Probabilidades (y_proba):
  - Binaria: vector 1D con prob. de la clase positiva.
  - Multiclase: matriz (n_samples, n_classes) con una columna por clase.
- Salida:
  - Un archivo Markdown (por defecto report.md) con la tabla de métricas y referencias a imágenes.
  - PNGs:
    - Clasificación: confusion.png y (si hay probabilidades) roc*.png, pr*.png.
      - Multiclase: roc_class_<clase>.png, pr_class_<clase>.png por clase.
    - Regresión/Forecasting: fit.png (y vs ŷ) y resid.png (residuales).
- Ubicación:
  - Si path no incluye carpeta, se usa ./evalcards_reports/.
  - Puedes fijar carpeta con out_dir o pasar una ruta completa en path.

Uso rápido (Python)
-------------------
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
        title="Mi modelo"
    )
    print(path)  # -> ruta del Markdown generado

Casos de uso
------------

Clasificación binaria
~~~~~~~~~~~~~~~~~~~~~
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

Incluye: accuracy, precision/recall/F1 (macro/weighted), AUC ROC y curvas ROC/PR.

Clasificación multiclase (OvR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

Incluye: métricas macro/weighted, AUC macro OvR (roc_auc_ovr_macro), y curvas ROC/PR por clase.

Regresión
~~~~~~~~~
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from evalcards import make_report

    X, y = make_regression(n_samples=600, n_features=8, noise=10, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    reg = RandomForestRegressor(random_state=0).fit(Xtr, ytr)
    y_pred = reg.predict(Xte)

    make_report(yte, y_pred, path="rep_reg.md", title="Regresión")

Incluye: MAE, MSE, RMSE, R² + gráficos de ajuste y residuales.

Forecasting
~~~~~~~~~~~
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

Incluye: MAE, MSE, RMSE, sMAPE (%), MASE + gráficos.

Referencia de API
-----------------

make_report(...)
~~~~~~~~~~~~~~~~
Firma:
    make_report(
        y_true,
        y_pred,
        y_proba: Optional[Sequence[float] | np.ndarray] = None,
        *,
        path: str = "report.md",
        title: str = "Reporte de Evaluación",
        labels: Optional[Sequence] = None,
        task: Literal["auto","classification","regression","forecast"] = "auto",
        out_dir: Optional[str] = None,
        # Forecast:
        season: int = 1,
        insample: Optional[Sequence[float]] = None,
    ) -> str

Parámetros clave:
- y_true, y_pred: arrays 1D del mismo largo.
- y_proba (opcional):
  - Binaria: vector 1D con prob. de la clase positiva.
  - Multiclase: matriz (n_samples, n_classes); columnas en el mismo orden que tus clases.
- labels: nombres legibles por clase (longitud = n_classes). Si no se pasa, se usan las clases numéricas.
- task:
  - "auto" (por defecto): detecta clasificación si la cantidad de valores únicos en y_true es “pequeña”; si no, usa regresión.
  - "classification", "regression", "forecast": forzar modo.
- path: ruta del Markdown. Si no incluye carpeta, se usa ./evalcards_reports/.
- out_dir: carpeta de salida. Si se pasa, tiene prioridad sobre la lógica por defecto.
- season / insample (forecast): estacionalidad (p.ej. 12) y serie de entrenamiento para MASE.

Retorna:
- Ruta (string) del archivo Markdown generado.

Efectos colaterales:
- Escribe PNGs en la carpeta de salida (confusion.png, roc*.png, pr*.png, fit.png, resid.png).

Detalles de métricas
--------------------

Clasificación
~~~~~~~~~~~~~
- accuracy
- precision_macro, recall_macro, f1_macro
- precision_weighted, recall_weighted, f1_weighted
- Binaria (si hay y_proba): roc_auc + curvas ROC y PR.
- Multiclase (si hay y_proba 2D): roc_auc_ovr_macro + curvas ROC/PR por clase (OvR).

Regresión
~~~~~~~~~
- MAE, MSE, RMSE, R².

Forecasting
~~~~~~~~~~~
- sMAPE (%):
      sMAPE = 100 * (2/n) * sum( |y - ŷ| / (|y| + |ŷ| + ε) )
- MASE (con estacionalidad m y serie insample):
      MASE = MAE(y, ŷ) / MAE(naive_m)
  donde naive_m usa |x_t - x_{t-m}| sobre insample (o sobre y_true si no se pasó insample).

Buenas prácticas y troubleshooting
----------------------------------
- Probabilidades:
  - Binaria: y_proba como vector 1D (prob. de la clase positiva).
  - Multiclase: matriz (n_samples, n_classes); cuida el orden de columnas (usa clf.classes_ en scikit-learn).
- Gráficos sin GUI:
  - Guardado a PNG no requiere GUI. Si tu entorno no tiene backend gráfico, puedes forzar:
      MPLBACKEND=Agg
  o antes de importar pyplot:
      import matplotlib; matplotlib.use("Agg")
- Rendimiento:
  - En datasets gigantes, los scatter pueden ser pesados. Muestra una muestra si lo necesitas.
- Errores típicos:
  - Shape mismatch: y_true y y_pred deben tener la misma longitud.
  - Probabilidades inválidas: deben estar en [0,1]; en multiclase, filas que sumen ~1.
  - Clases faltantes: verifica que labels tenga una entrada por clase.

Limitaciones actuales
---------------------
- No multi-label (varias etiquetas verdaderas por muestra).
- Multiclase: AUC macro OvR y curvas por clase (no micro/macro AUC/PR globales aún).
- No incluye métricas de ranking (MAP/NDCG) ni calibración (Brier/Curva de calibración).

Versionado y compatibilidad
---------------------------
- Soporta Python 3.9 – 3.13.
- Sigue SemVer a grandes rasgos: parches para fixes, minor para nuevas funciones, major para cambios incompatibles.
- Consulta el CHANGELOG.md para novedades por versión.

Contribuir
----------
- Abre un issue describiendo tu propuesta.
- Añade tests en tests/.
- Ejecuta pytest -q localmente y verifica CI.

Licencia
--------
MIT — © Ricardo Urdaneta.