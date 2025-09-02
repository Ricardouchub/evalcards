# Referencia de API — evalcards

Actualmente, el punto de entrada público es **`make_report`**.

---

### `make_report(...)`

**Módulo**: `evalcards.report`  
**Devuelve**: `str` con la **ruta** del Markdown generado.

### Firma
```python
make_report(
    y_true,
    y_pred,
    y_proba: Optional[Sequence[float] | np.ndarray] = None,
    *,
    path: str = "report.md",
    title: str = "Reporte de Evaluación",
    labels: Optional[Sequence] = None,
    task: Literal["auto","classification","regression","forecast"] = "auto",
    out_dir: Optional[Sequence] = None,
    # Forecast:
    season: int = 1,
    insample: Optional[Sequence[float]] = None,
    lang: str = "es"
) -> str
```

### Parámetros
- **`y_true`** (`array-like 1D`): valores/etiquetas reales. Soporta `list`, `np.ndarray`, `pd.Series`.
- **`y_pred`** (`array-like 1D`): valores/etiquetas predichas. Misma longitud que `y_true`.
- **`y_proba`** (opcional):  
  - **Binaria**: `array-like 1D` con prob. de la clase positiva.  
  - **Multiclase**: `array-like 2D` `(n_samples, n_classes)` con prob. por clase.
- **`path`** (`str`, por defecto `"report.md"`): nombre del archivo Markdown a generar.  
  - Si no incluye carpeta, se guardará en `./evalcards_reports/` por defecto.
- **`title`** (`str`): título mostrado en el reporte.
- **`labels`** (`Sequence`, opcional): nombres legibles por clase (longitud = `n_classes`). Si no se pasa, se usan las clases tal cual.
- **`task`** (`"auto" | "classification" | "regression" | "forecast"`):  
  - `"auto"` intenta detectar clasificación si el número de valores únicos en `y_true` es pequeño (heurística) y si no, usa regresión.
- **`out_dir`** (`str | None`): carpeta de salida. Si se indica, tiene prioridad sobre la carpeta por defecto.
- **`season`** (`int`, forecast): periodicidad estacional para MASE (p.ej. 12 para mensual con patrón anual).
- **`insample`** (`array-like 1D`, forecast): serie de entrenamiento usada para el denominador de **MASE**. Si no se pasa, se usa `y_true` como *fallback*.
- **`lang`** (`str`, default `"es"`):  
  Idioma del reporte: `"es"` (español, default), `"en"` (inglés).  
  Aplica a todos los textos, títulos y etiquetas del Markdown generado.

### Retorno
- **`str`**: ruta absoluta o relativa del **Markdown** generado.

### Efectos colaterales (archivos)
- **Clasificación**: `confusion.png`; si hay `y_proba` binaria: `roc.png`, `pr.png`.  
  Multiclase: `roc_class_<clase>.png`, `pr_class_<clase>.png` por clase.
- **Regresión/Forecast**: `fit.png`, `resid.png`.

Los archivos se escriben en la carpeta resuelta por `out_dir`/`path`.

### Métricas generadas

**Clasificación**
- `accuracy`
- `precision_macro`, `recall_macro`, `f1_macro`
- `precision_weighted`, `recall_weighted`, `f1_weighted`
- **Binaria** (con `y_proba`): `roc_auc`, curvas **ROC** y **PR**.
- **Multiclase** (con `y_proba` 2D): `roc_auc_ovr_macro`, curvas **ROC/PR por clase** (OvR).

**Regresión**
- `MAE`, `MSE`, `RMSE`, `R²`.

**Forecasting**
- `MAE`, `MSE`, `RMSE`
- **`sMAPE (%)`**
- **`MASE`** (requiere `season` y preferentemente `insample`).

### Errores y validaciones
- Longitudes incompatibles entre `y_true` y `y_pred` ⇒ error.
- `y_proba` fuera de `[0,1]` o filas que no suman ~1 en multiclase ⇒ resultados indefinidos; valida antes de llamar.
- AUC (binaria) puede no calcularse si `y_true` no contiene ambas clases; el código lo ignora silenciosamente y no rompe el reporte.

### Ejemplos breves
**Binaria**
```python
make_report(y_true, y_pred, y_proba=proba, path="rep_bin.md", title="Binaria")
```

**Multiclase**
```python
make_report(y_true, y_pred, y_proba=proba_matrix,
            labels=["A","B","C"], path="rep_multi.md", title="Multiclase OvR")
```

**Regresión**
```python
make_report(y_true, y_pred, path="rep_reg.md", title="Regresión")
```

**Forecast**
```python
make_report(y_test, y_hat, task="forecast", season=12, insample=y_train,
            path="rep_forecast.md", title="Forecast")
```

### Ejemplo de uso multilenguaje

```python
make_report(y_true, y_pred, path="reporte.md", lang="es", title="Mi reporte")
make_report(y_true, y_pred, path="report_en.md", lang="en", title="My report")
```
---

> Para una introducción paso a paso y ejemplos completos, ver la **[Guía completa](index.md)**.
