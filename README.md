evalcards
================

Descripción
-----------
`evalcards` genera reportes de evaluación para **modelos supervisados** en **Markdown**, con gráficos listos. Soporta:
- Clasificación (binaria y multiclase OvR)
- Regresión
- Forecasting (sMAPE/MASE)

Características
---------------
- Tabla de métricas por tarea.
- Gráficos: matriz de confusión, ROC/PR (binaria y multiclase OvR), ajuste y residuales.
- CLI y API Python.
- Carpeta de salida por defecto: `./evalcards_reports` (personalizable con `--outdir`).

Requisitos
----------
- Python 3.9+
- Dependencias se instalan automáticamente (numpy, pandas, scikit-learn, matplotlib, jinja2).

Instalación (dev/local)
-----------------------
1) Crear y activar entorno virtual (opcional):
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     . .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2) Instalar en modo editable desde la carpeta del proyecto:
   ```bash
   pip install -e .
   ```

Uso rápido (API Python)
----------------------
**Clasificación binaria (con probabilidades opcionales):**
```python
from evalcards import make_report

path = make_report(
    y_true, y_pred, y_proba=proba,         # y_proba: vector 1D (prob. clase positiva)
    path="reporte.md", title="Mi modelo"
)
print(path)
```

**Regresión:**
```python
make_report(y_true, y_pred, path="rep_reg.md", title="Regresión")
```

Uso rápido (CLI)
----------------
1) Prepara CSVs con una sola columna o con nombres estándar (`y_true`, `y_pred`, `y_proba`).
2) Ejecuta (por defecto guarda en `./evalcards_reports`):
   ```bash
   evalcards --y_true y_true.csv --y_pred y_pred.csv --proba y_proba.csv --out rep.md --title "Mi modelo"
   ```
3) Carpeta personalizada:
   ```bash
   evalcards --y_true y_true.csv --y_pred y_pred.csv --out rep.md --outdir informes_eval
   ```

Clasificación multiclase (OvR)
------------------------------
### API (Python)
```python
# y_proba: matriz (n_samples, n_classes) con probabilidades
out = make_report(
    y_true, y_pred, y_proba=proba,
    labels=["Clase_A","Clase_B","Clase_C"],  # opcional
    path="rep_multi.md", title="Multiclase OvR"
)
```

### CLI
```bash
# y_proba.csv con N columnas (una por clase)
evalcards --y_true y_true.csv --y_pred y_pred.csv --proba y_proba.csv --class-names "Clase_A,Clase_B,Clase_C" --out rep_cli_multiclase.md
```
**Salidas:** `confusion.png` + `roc_class_<clase>.png` y `pr_class_<clase>.png` por clase.  
**Métrica:** `roc_auc_ovr_macro`.

Forecasting (sMAPE/MASE)
------------------------
### API (Python)
```python
out = make_report(
    y_true_test, y_pred_test,
    task="forecast", season=12, insample=y_true_train,
    path="rep_forecast.md", title="Forecast"
)
```

### CLI
```bash
evalcards --y_true y_true_test.csv --y_pred y_pred_test.csv --forecast --season 12 --insample y_insample.csv --out rep_forecast_cli.md
```
**Métricas:** `MAE`, `MSE`, `RMSE`, `sMAPE (%)`, `MASE`.  
**Gráficos:** `fit.png` y `resid.png`.

Entradas esperadas
------------------
- **Clasificación:**
  - `y_true`: etiquetas reales (0..K-1 o nombres).
  - `y_pred`: etiquetas predichas.
  - `y_proba`:
    - **Binaria:** vector 1D (prob. clase positiva).
    - **Multiclase:** matriz (N columnas, una por clase).
- **Regresión/Forecast:**
  - `y_true`: valores reales.
  - `y_pred`: valores predichos.
  - `insample` (opcional para MASE): serie de entrenamiento; usar `--season` (ej. 12).

Salidas
-------
- Archivo Markdown con métricas y referencias a imágenes.
- Imágenes PNG en la carpeta destino (por defecto: `./evalcards_reports`):
  - **Clasificación:** `confusion.png`; si hay probabilidades binaria: `roc.png`, `pr.png`; en multiclase: `roc_class_<clase>.png` y `pr_class_<clase>.png` por clase.
  - **Regresión/Forecast:** `fit.png` (ajuste y vs ŷ) y `resid.png` (residuales).

Métricas incluidas
------------------
- **Clasificación:** `accuracy`, `precision`/`recall`/`F1` (macro y weighted).  
  - Binaria: `roc_auc`.  
  - Multiclase OvR: `roc_auc_ovr_macro`.
- **Regresión:** `MAE`, `MSE`, `RMSE`, `R²`.
- **Forecast:** `MAE`, `MSE`, `RMSE`, `sMAPE (%)`, `MASE`.

Notas técnicas
--------------
- Si solo pasas un nombre de archivo en `--out`/`path` (p. ej. `rep.md`), se usará `./evalcards_reports` automáticamente.
- Si especificas una ruta con carpeta, se respeta esa ruta.
- `--outdir` fija explícitamente la carpeta de salida.
- El paquete fuerza backend **Agg** de Matplotlib para generar PNGs en entornos sin GUI.

Solución de problemas (rápida)
------------------------------
- “No module named evalcards”: instala con `pip install -e .` desde la carpeta del proyecto y activa tu venv.
- Matplotlib/Tk en servidores: no se necesita GUI; se guardan PNGs con backend `Agg`.
- CSVs: si el archivo tiene varias columnas, usa nombres `y_true`/`y_pred`/`y_proba` o deja una sola columna.

Soporte actual
--------------
- Detección automática de tarea (`auto`): clasificación vs regresión.  
- **Multiclase:** curvas ROC/PR **OvR** y `roc_auc_ovr_macro`.  
- **Forecasting:** sMAPE/MASE con `--season` e `--insample`.

Licencia
--------
MIT (puedes cambiarla si lo prefieres).

Autor
-----
Ricardo Urdaneta (Ricardouchub)