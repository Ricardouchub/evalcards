from __future__ import annotations
import os
from typing import Optional, Sequence, Literal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

Task = Literal["auto","classification","regression"]

DEFAULT_OUTDIR = "evalcards_reports"

def _resolve_out(path: str, out_dir: str | None):
    """
    Devuelve (out_dir_final, path_final).
    - Si solo pasan un nombre de archivo, guardamos todo en ./evalcards_reports
    - Si pasan --outdir, usamos esa carpeta
    - Si path ya trae carpeta, respetamos esa ruta
    """
    if out_dir:  # usuario especificó carpeta
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, os.path.join(out_dir, os.path.basename(path))
    d = os.path.dirname(os.path.abspath(path))
    if d == os.path.abspath(os.getcwd()):  # solo nombre de archivo
        out_dir = os.path.join(os.getcwd(), DEFAULT_OUTDIR)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, os.path.join(out_dir, os.path.basename(path))
    os.makedirs(d, exist_ok=True)
    return d, os.path.abspath(path)

def _is_classification(y_true) -> bool:
    y = np.asarray(y_true)
    uniq = np.unique(y).size
    return uniq <= max(20, int(0.05 * y.size))

def _plot_confusion(y_true, y_pred, labels=None, path="confusion.png"):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path

def _plot_regression_fit(y_true, y_pred, path="fit.png"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=10)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("y real"); ax.set_ylabel("y predicho"); ax.set_title("Ajuste")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path

def _plot_residuals(y_true, y_pred, path="resid.png"):
    resid = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, resid, s=10)
    ax.axhline(0)
    ax.set_xlabel("y predicho"); ax.set_ylabel("residual"); ax.set_title("Residuales")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path

def _plot_roc(y_true, y_proba, path="roc.png"):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path

def _plot_pr(y_true, y_proba, path="pr.png"):
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path

def make_report(
    y_true,
    y_pred,
    y_proba: Optional[Sequence[float]] = None,
    *,
    path: str = "report.md",
    title: str = "Reporte de Evaluación",
    labels: Optional[Sequence] = None,
    task: Task = "auto",
    out_dir: Optional[str] = None,          # <- NUEVO
) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = None if y_proba is None else np.asarray(y_proba)

    # Resolver carpeta/archivo de salida
    out_dir, path = _resolve_out(path, out_dir)

    if task == "auto":
        task = "classification" if _is_classification(y_true) else "regression"

    lines = [f"# {title}", "", f"**Tarea:** {task}", ""]

    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        img_conf = _plot_confusion(y_true, y_pred, labels=labels, path=os.path.join(out_dir, "confusion.png"))

        roc_img = pr_img = None
        if y_proba is not None and y_proba.ndim == 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception:
                pass
            roc_img = _plot_roc(y_true, y_proba, path=os.path.join(out_dir, "roc.png"))
            pr_img  = _plot_pr(y_true, y_proba,  path=os.path.join(out_dir, "pr.png"))

        lines += ["## Métricas", "| métrica | valor |", "|---|---:|"]
        for k, v in metrics.items(): lines.append(f"| {k} | {v:.4f} |")
        lines += ["", "## Gráficos", f"![confusion]({os.path.basename(img_conf)})"]
        if roc_img: lines.append(f"![roc]({os.path.basename(roc_img)})")
        if pr_img:  lines.append(f"![pr]({os.path.basename(pr_img)})")
        lines.append("")

    else:  # regression
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        fit = _plot_regression_fit(y_true, y_pred, path=os.path.join(out_dir, "fit.png"))
        resid = _plot_residuals(y_true, y_pred, path=os.path.join(out_dir, "resid.png"))
        lines += ["## Métricas", "| métrica | valor |", "|---|---:|",
                  f"| MAE | {mae:.4f} |", f"| MSE | {mse:.4f} |",
                  f"| RMSE | {rmse:.4f} |", f"| R2 | {r2:.4f} |",
                  "", "## Gráficos",
                  f"![fit]({os.path.basename(fit)})",
                  f"![resid]({os.path.basename(resid)})", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def _load_vec(path):
    import pandas as pd
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy()
    for c in ("y_true", "y_pred", "y_proba"):
        if c in df.columns:
            return df[c].to_numpy()
    raise SystemExit(f"No pude inferir la columna en {path} (usa 1 columna o nómbrala y_true/y_pred/y_proba).")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Genera reporte de evaluación (Markdown)")
    p.add_argument("--y_true", required=True, help="CSV con y_true (1 columna o columna 'y_true')")
    p.add_argument("--y_pred", required=True, help="CSV con y_pred (1 columna o columna 'y_pred')")
    p.add_argument("--proba", help="CSV con y_proba (binaria)")
    p.add_argument("--out", default="report.md")
    p.add_argument("--outdir", help="Carpeta destino (por defecto ./evalcards_reports)", default=None)  # <- NUEVO
    p.add_argument("--title", default="Reporte de Evaluación")
    args = p.parse_args()

    y_true = _load_vec(args.y_true)
    y_pred = _load_vec(args.y_pred)
    y_proba = _load_vec(args.proba) if args.proba else None

    out_path = make_report(
        y_true, y_pred, y_proba=y_proba,
        path=args.out, title=args.title, out_dir=args.outdir
    )
    print(os.path.abspath(out_path))