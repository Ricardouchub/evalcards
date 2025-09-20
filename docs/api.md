# Referencia de API — evalcards

Actualmente, el punto de entrada público es **`make_report`**.

---

### `make_report(...)`

**Módulo**: `evalcards.report`  
**Devuelve**:  
- Si no se solicita export JSON: `str` con la **ruta** del Markdown generado.  
- Si se pasa `export_json`: `Tuple[str, dict]` -> `(ruta_markdown, info_dict)` donde `info_dict` contiene las métricas y rutas de los gráficos.

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
    task: Literal["auto","classification","regression","forecast","multi-label"] = "auto",
    out_dir: Optional[str] = None,
    # Forecast:
    season: int = 1,
    insample: Optional[Sequence[float]] = None,
    lang: str = "es",
    metrics: Optional[List[str]] = None,
    export_json: Optional[str] = None,  # << nuevo parámetro
) -> Union[str, Tuple[str, Dict[str, Any]]]
```

### Parámetros
- **`y_true`** (`array-like 1D` o `2D` en multi-label): valores/etiquetas reales. Soporta `list`, `np.ndarray`, `pd.Series`.
- **`y_pred`** (`array-like 1D` o `2D` en multi-label): valores/etiquetas predichas. Misma longitud/forma que `y_true`.
- **`y_proba`** (opcional):  
  - **Binaria**: `array-like 1D` con prob. de la clase positiva.  
  - **Multiclase**: `array-like 2D` `(n_samples, n_classes)` con prob. por clase.  
  - **Multi-label**: `array-like 2D` `(n_samples, n_labels)` con prob. por etiqueta.
- **`path`** (`str`, por defecto `"report.md"`): nombre del archivo Markdown a generar.  
  - Si no incluye carpeta, se guardará en `./evalcards_reports/` por defecto.
- **`title`** (`str`): título mostrado en el reporte.
- **`labels`** (`Sequence`, opcional): nombres legibles por clase/etiqueta (longitud = `n_classes`/`n_labels`). Si no se pasa, se usan índices o clases originales.
- **`task`** (`"auto" | "classification" | "regression" | "forecast" | "multi-label"`):  
  - `"auto"` intenta detectar multi-label (si `y_true` y `y_pred` son 2D binarias) o clasificación/regresión por heurística.
- **`out_dir`** (`str | None`): carpeta de salida. Si se indica, tiene prioridad sobre la carpeta por defecto.
- **`season`**, **`insample`**, **`lang`**, **`metrics`**: ver sección Forecast / i18n / selección de métricas.
- **`export_json`** (`str` | `None`) — NUEVO:  
  - Si se pasa una ruta (absoluta o relativa), además del Markdown se generará un archivo JSON con las métricas calculadas y las rutas de las imágenes generadas.
  - Si `export_json` se pasa como nombre (sin carpeta), el JSON se colocará en la misma carpeta de salida resuelta (`out_dir` / carpeta por defecto).
  - Si `export_json` es `None` (por defecto), no se genera el JSON y el comportamiento previo no cambia.

### Retorno
- **`str`**: ruta al archivo Markdown generado (caso sin `export_json`).  
- **`(str, dict)`**: `(ruta_markdown, info_dict)` si `export_json` fue solicitado. `info_dict` contiene:
  - `metrics`: mapping métrica -> valor (floats/valores serializables).
  - `charts`: mapping con los nombres/paths relativos de las imágenes generadas (por tipo).
  - `markdown`: nombre del archivo Markdown generado (basename).

Ejemplo de `info_dict`:
```json
{
  "metrics": {
    "accuracy": 0.9125,
    "roc_auc": 0.94
  },
  "charts": {
    "confusion": "confusion.png",
    "roc": ["roc.png"],
    "pr": ["pr.png"]
  },
  "markdown": "report.md"
}
```

### Efectos colaterales (archivos)
- **Clasificación**: `confusion.png`; si hay `y_proba` binaria: `roc.png`, `pr.png`.  
  Multiclase: `roc_class_<clase>.png`, `pr_class_<clase>.png` por clase.
- **Multi-label**: `confusion_<etiqueta>.png`, `roc_label_<etiqueta>.png`, `pr_label_<etiqueta>.png` por etiqueta (si se pasan probabilidades).
- **Regresión/Forecast**: `fit.png`, `resid.png`.

Los archivos se escriben en la carpeta resuelta por `out_dir`/`path`. Si `export_json` se indica con ruta absoluta, el JSON se escribirá exactamente allí (creando directorio si es necesario).

### Ejemplos breves

- Uso en Python (export JSON y usar info en memoria):
```python
md_path, info = make_report(y_true, y_pred, y_proba=proba, path="rep.md", title="Mi modelo", export_json="rep.json")
print(md_path)         # ruta del markdown
print(info["metrics"]) # dict con métricas
```

- Uso desde CLI:
```bash
evalcards --y_true y_true.csv --y_pred y_pred.csv --proba proba.csv --out report.md --export-json report.json
# imprime rutas y crea report.md + report.json
```

### Errores y validaciones
- Longitudes incompatibles entre `y_true` y `y_pred` ⇒ error.
- `y_proba` fuera de `[0,1]` o filas que no suman ~1 en multiclase ⇒ resultados indefinidos; valida antes de llamar.
- AUC puede no calcularse si falta alguna clase; el código ignora silenciosamente métricas imposibles y no rompe el reporte.

Para más ejemplos y detalles (CLI, multi-label, forecasting) ver la [Guía completa](index.md).

---

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
