from __future__ import annotations
import os
from typing import Optional, Sequence, Literal
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

from .lang import LANG

Task = Literal["auto", "classification", "regression", "forecast"]

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

def _sanitize(name) -> str:
    s = str(name)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

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

# ---------- Forecast metrics ----------
def _smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)

def _mase(y_true, y_pred, season: int = 1, insample: Optional[Sequence[float]] = None, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    series = y_true if insample is None else np.asarray(insample, dtype=float)
    if series.size <= season:
        season = 1
    diff = np.abs(series[season:] - series[:-season])
    if diff.size == 0:  # fallback naive-1
        diff = np.abs(series[1:] - series[:-1])
    scale = np.mean(diff) if diff.size else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae / (scale + eps))

def make_report(
    y_true,
    y_pred,
    y_proba: Optional[Sequence[float]] = None,
    *,
    path: str = "report.md",
    title: str = None,
    labels: Optional[Sequence] = None,
    task: Task = "auto",
    out_dir: Optional[str] = None,
    season: int = 1,
    insample: Optional[Sequence[float]] = None,
    lang: str = "es",  # Nuevo parámetro
) -> str:
    T = LANG.get(lang, LANG["es"])
    if title is None:
        title = T["title_default"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = None if y_proba is None else np.asarray(y_proba)

    out_dir, path = _resolve_out(path, out_dir)

    if task == "auto":
        task = "classification" if _is_classification(y_true) else "regression"

    lines = [f"# {title}", "", f"**{T['task']}:** {T.get(task, task)}", ""]

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

        roc_imgs, pr_imgs = [], []
        if y_proba is not None:
            if y_proba.ndim == 1:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                except Exception:
                    pass
                roc_imgs.append(_plot_roc(y_true, y_proba, path=os.path.join(out_dir, "roc.png")))
                pr_imgs.append(_plot_pr(y_true, y_proba, path=os.path.join(out_dir, "pr.png")))
            elif y_proba.ndim == 2:
                n_classes = y_proba.shape[1]
                try:
                    metrics["roc_auc_ovr_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                except Exception:
                    pass
                names = list(labels) if (labels is not None and len(labels) >= n_classes) else list(range(n_classes))
                for i in range(n_classes):
                    pos = (y_true == i).astype(int)
                    proba_i = y_proba[:, i]
                    cname = _sanitize(names[i] if i < len(names) else i)
                    roc_imgs.append(_plot_roc(pos, proba_i, path=os.path.join(out_dir, f"roc_class_{cname}.png")))
                    pr_imgs.append(_plot_pr(pos,  proba_i, path=os.path.join(out_dir, f"pr_class_{cname}.png")))

        lines += [f"## {T['metrics']}", f"| {T['metric']} | {T['value']} |", "|---|---:|"]
        for k, v in metrics.items():
            lines.append(f"| {k} | {v:.4f} |")

        lines += ["", f"## {T['charts']}", f"![{T['confusion']}]({os.path.basename(img_conf)})"]
        for p in roc_imgs:
            lines.append(f"![{T['roc']}]({os.path.basename(p)})")
        for p in pr_imgs:
            lines.append(f"![{T['pr']}]({os.path.basename(p)})")
        lines.append("")

    elif task == "regression":
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        fit = _plot_regression_fit(y_true, y_pred, path=os.path.join(out_dir, "fit.png"))
        resid = _plot_residuals(y_true, y_pred, path=os.path.join(out_dir, "resid.png"))
        lines += [f"## {T['metrics']}", f"| {T['metric']} | {T['value']} |", "|---|---:|",
                  f"| MAE | {mae:.4f} |", f"| MSE | {mse:.4f} |",
                  f"| RMSE | {rmse:.4f} |", f"| R2 | {r2:.4f} |",
                  "", f"## {T['charts']}",
                  f"![{T['fit']}]({os.path.basename(fit)})",
                  f"![{T['resid']}]({os.path.basename(resid)})", ""]

    else:  # forecast
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        smape = _smape(y_true, y_pred)
        mase = _mase(y_true, y_pred, season=season, insample=insample)
        fit = _plot_regression_fit(y_true, y_pred, path=os.path.join(out_dir, "fit.png"))
        resid = _plot_residuals(y_true, y_pred, path=os.path.join(out_dir, "resid.png"))
        lines += [f"## {T['metrics']}", f"| {T['metric']} | {T['value']} |", "|---|---:|",
                  f"| MAE | {mae:.4f} |", f"| MSE | {mse:.4f} |",
                  f"| RMSE | {rmse:.4f} |", f"| sMAPE (%) | {smape:.2f} |",
                  f"| MASE | {mase:.4f} |",
                  "", f"## {T['charts']}",
                  f"![{T['fit']}]({os.path.basename(fit)})",
                  f"![{T['resid']}]({os.path.basename(resid)})", ""]

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

def _load_proba(path):
    if not path:
        return None
    import pandas as pd
    df = pd.read_csv(path)
    if df.shape[1] > 1:
        return df.to_numpy()
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy()
    raise SystemExit(f"No pude leer probabilidades desde {path} (usa 1 columna o varias, una por clase).")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Genera reporte de evaluación (Markdown)")
    p.add_argument("--y_true", required=True, help="CSV con y_true (1 columna o columna 'y_true')")
    p.add_argument("--y_pred", required=True, help="CSV con y_pred (1 columna o columna 'y_pred')")
    p.add_argument("--proba", help="CSV con y_proba (binaria: 1 col; multiclase: N columnas)")
    p.add_argument("--class-names", help="Nombres de clases separados por coma (solo multiclase)", default=None)
    p.add_argument("--out", default="report.md")
    p.add_argument("--outdir", help="Carpeta destino (por defecto ./evalcards_reports)", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--lang", default="es", help="Idioma (es/en)")  # Nuevo parámetro

    # Flags forecast
    p.add_argument("--forecast", action="store_true", help="Tratar como pronóstico (usa sMAPE/MASE)")
    p.add_argument("--season", type=int, default=1, help="Periodicidad estacional para MASE (ej. 12)")
    p.add_argument("--insample", help="CSV con serie insample para MASE (opcional)")

    args = p.parse_args()

    y_true = _load_vec(args.y_true)
    y_pred = _load_vec(args.y_pred)
    y_proba = _load_proba(args.proba) if args.proba else None
    insample = _load_vec(args.insample) if args.insample else None
    labels = [s.strip() for s in args.class_names.split(",")] if args.class_names else None

    task: Task = "forecast" if args.forecast else "auto"

    out_path = make_report(
        y_true, y_pred, y_proba=y_proba,
        path=args.out, title=args.title, out_dir=args.outdir,
        task=task, season=args.season, insample=insample,
        labels=labels, lang=args.lang
    )
    print(os.path.abspath(out_path))