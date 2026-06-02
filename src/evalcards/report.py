from __future__ import annotations
import os
from typing import Optional, Sequence, Literal, List, Dict, Any, Tuple, Union
import numpy as np

from .evaluators import (
    evaluate_multilabel, evaluate_classification,
    evaluate_regression, evaluate_forecast, evaluate_ranking
)
from .generator import generate_report

Task = Literal["auto", "classification", "regression", "forecast", "multi-label", "ranking"]

DEFAULT_OUTDIR = "evalcards_reports"

def _resolve_out(path: str, out_dir: str | None):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, os.path.join(out_dir, os.path.basename(path))
    d = os.path.dirname(os.path.abspath(path))
    if d == os.path.abspath(os.getcwd()):
        out_dir = os.path.join(os.getcwd(), DEFAULT_OUTDIR)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, os.path.join(out_dir, os.path.basename(path))
    os.makedirs(d, exist_ok=True)
    return d, os.path.abspath(path)

def _is_classification(y_true) -> bool:
    y = np.asarray(y_true)
    uniq = np.unique(y).size
    return uniq <= max(20, int(0.05 * y.size))

def _is_multilabel(y_true, y_pred) -> bool:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape:
        vals = np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()]))
        return np.all(np.isin(vals, [0, 1]))
    return False

def make_report(
    y_true,
    y_pred,
    y_proba=None,
    *,
    path: str = "report.md",
    title: str = None,
    labels: Optional[Sequence] = None,
    task: Task = "auto",
    out_dir: Optional[str] = None,
    season: int = 1,
    insample: Optional[Sequence[float]] = None,
    lang: str = "es",
    metrics: Optional[List[str]] = None,
    fmt: str = "md",
    export_json: Optional[str] = None,
    sensitive_features: Optional[Sequence] = None,
    query_id: Optional[Sequence] = None,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    
    y_true = np.asarray(y_true)
    
    # Don't convert to array if it's a dict (multi-model)
    if not isinstance(y_pred, dict):
        y_pred = np.asarray(y_pred)
        
    if y_proba is not None and not isinstance(y_proba, dict):
        y_proba = np.asarray(y_proba)

    out_dir, path = _resolve_out(path, out_dir)

    if task == "auto":
        if _is_multilabel(y_true, y_pred):
            task = "multi-label"
        else:
            task = "classification" if _is_classification(y_true) else "regression"
    elif task == "multi-label":
        if not _is_multilabel(y_true, y_pred):
            raise ValueError("Para 'multi-label', y_true e y_pred deben ser arrays 2D binarios de igual forma.")

    if task == "multi-label":
        eval_result = evaluate_multilabel(y_true, y_pred, y_proba, labels, out_dir, metrics, sensitive_features, fmt)
    elif task == "classification":
        eval_result = evaluate_classification(y_true, y_pred, y_proba, labels, out_dir, metrics, sensitive_features, fmt)
    elif task == "regression":
        eval_result = evaluate_regression(y_true, y_pred, out_dir, metrics, sensitive_features, fmt)
    elif task == "ranking":
        if query_id is None:
            raise ValueError("Se requiere 'query_id' para la tarea de ranking.")
        eval_result = evaluate_ranking(y_true, y_pred, query_id, out_dir, metrics, fmt)
    else:
        eval_result = evaluate_forecast(y_true, y_pred, season, insample, out_dir, metrics, sensitive_features, fmt)

    from .insights import generate_insights
    eval_result.insights = generate_insights(eval_result)

    return generate_report(eval_result, path, out_dir, title, lang, fmt, export_json)

# Mantener main() temporalmente en caso de que alguien lo importe desde evalcards.report
# aunque pyproject.toml debe apuntar a evalcards.cli:main
def main():
    from .cli import main as cli_main
    cli_main()