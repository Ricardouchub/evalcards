evalcards
================

Descripción
----------
evalcards genera reportes de evaluación para **modelos supervisados** (clasificación y regresión) en formato Markdown, incluyendo gráficos listos (matriz de confusión, ROC/PR, ajuste y residuales). Pensado para usarse tanto desde Python como desde la línea de comandos (CLI).

Requisitos
----------
- Python 3.9+
- Las dependencias se instalan automáticamente al instalar el paquete (numpy, pandas, scikit-learn, matplotlib, jinja2).

Instalación (local / desarrollo)
--------------------------------
1) Crear y activar un entorno virtual (opcional pero recomendado)
   - Windows (PowerShell):
       python -m venv .venv
       . .\.venv\Scripts\Activate.ps1
   - macOS / Linux:
       python3 -m venv .venv
       source .venv/bin/activate

2) Instalar en modo editable desde la carpeta del proyecto:
       pip install -e .

Uso desde Python
----------------
Ejemplo (clasificación binaria con probabilidades opcionales):

    from evalcards import make_report

    # y_true: etiquetas reales (array-like)
    # y_pred: etiquetas predichas (array-like)
    # y_proba: probabilidad de la clase positiva (shape (n_samples,)), opcional
    path = make_report(
        y_true, y_pred, y_proba=proba,
        path="reporte.md",
        title="Mi modelo"
    )
    print(path)  # ruta al reporte Markdown

Para regresión, simplemente no pases y_proba y utiliza tus vectores reales y predichos (y_true, y_pred).

Uso desde la CLI
----------------
1) Prepara CSVs con una sola columna o con nombres estándar (y_true, y_pred, y_proba).
2) Ejecuta:

    # Por defecto guarda todo en ./evalcards_reports
    evalcards --y_true y_true.csv --y_pred y_pred.csv --proba y_proba.csv --out rep.md --title "Mi modelo"

    # Carpeta de salida personalizada
    evalcards --y_true y_true.csv --y_pred y_pred.csv --out rep.md --outdir informes_eval

Entradas esperadas
------------------
- Clasificación:
  - y_true: etiquetas reales (0/1 o multicategoría).
  - y_pred: etiquetas predichas.
  - y_proba (opcional, binaria): prob. de la clase positiva, vector 1D de longitud n_samples.

- Regresión:
  - y_true: valores reales (float).
  - y_pred: valores predichos (float).

Salidas
-------
- Un archivo Markdown (por ejemplo, rep.md) con las métricas y referencias a imágenes.
- Imágenes PNG guardadas en la carpeta destino (por defecto: ./evalcards_reports):
  - Clasificación: confusion.png; si pasas y_proba (binaria): roc.png y pr.png.
  - Regresión: fit.png (ajuste y vs ŷ) y resid.png (residuales).

Métricas incluidas
------------------
- Clasificación: accuracy, precision/recall/F1 (macro y weighted). Si hay y_proba (binaria): AUC (ROC).
- Regresión: MAE, MSE, RMSE, R².

Soporte actual
--------------
- Detección automática de tarea (classification vs regression).
- Clasificación: binaria y multiclase (métricas + matriz de confusión). Curvas ROC/PR: por ahora solo binaria.
- Regresión: métricas básicas y dos gráficos (ajuste y residuales).

Notas técnicas
--------------
- Cuando solo pasas un nombre de archivo en --out o path (por ejemplo, "rep.md"), evalcards crea por defecto la carpeta ./evalcards_reports y guarda allí el reporte y las imágenes.
- Si especificas una ruta con carpeta, se respeta esa ruta.
- Con --outdir puedes fijar explícitamente la carpeta de salida.

Solución de problemas (rápida)
------------------------------
- "No module named evalcards": asegúrate de correr 'pip install -e .' en la carpeta del proyecto y que tu entorno virtual esté activo.
- Problemas con matplotlib en servidores sin GUI: evalcards usa guardado a PNG y no requiere backend interactivo.
- CSVs: si tu archivo tiene varias columnas, incluye una llamada y_true/y_pred/y_proba o deja una sola columna.

Licencia
--------
MIT (puedes cambiarla si lo prefieres).

Autor
-----
Ricardo Urdaneta (Ricardouchub)