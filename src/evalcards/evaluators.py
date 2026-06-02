import os
from typing import Optional, Sequence, List, Dict, Union, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, balanced_accuracy_score, log_loss,
    median_absolute_error, brier_score_loss, ndcg_score
)

from .models import EvaluationResult
from .plotting import (
    plot_confusion, plot_multilabel_confusions, plot_multi_roc, plot_multi_pr,
    plot_multilabel_roc_pr, plot_regression_fit, plot_residuals,
    plot_calibration_curve, plot_threshold_curves
)

from .plotting_interactive import (
    plot_confusion_plotly, plot_multi_roc_plotly, plot_multi_pr_plotly,
    plot_calibration_curve_plotly, plot_regression_fit_plotly
)

def _sanitize(name) -> str:
    s = str(name)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)

def mase(y_true, y_pred, season: int = 1, insample: Optional[Sequence[float]] = None, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    series = y_true if insample is None else np.asarray(insample, dtype=float)
    if series.size <= season:
        season = 1
    diff = np.abs(series[season:] - series[:-season])
    if diff.size == 0:
        diff = np.abs(series[1:] - series[:-1])
    scale = np.mean(diff) if diff.size else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae / (scale + eps))

def _ensure_dict(y, default_name="Modelo 1"):
    if y is None:
        return None
    if isinstance(y, dict):
        return y
    return {default_name: y}

def _compute_fairness_classification(y_true, y_preds_dict, sensitive_features):
    sf = np.asarray(sensitive_features)
    groups = np.unique(sf)
    fairness = {}
    for g in groups:
        mask = (sf == g)
        if not np.any(mask):
            continue
        g_metrics = {}
        for m_name, y_pred in y_preds_dict.items():
            g_metrics[m_name] = {
                "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
                "f1_macro": f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            }
        fairness[g] = g_metrics
    return fairness

def _compute_fairness_regression(y_true, y_preds_dict, sensitive_features):
    sf = np.asarray(sensitive_features)
    groups = np.unique(sf)
    fairness = {}
    for g in groups:
        mask = (sf == g)
        if not np.any(mask):
            continue
        g_metrics = {}
        for m_name, y_pred in y_preds_dict.items():
            g_metrics[m_name] = {
                "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
                "RMSE": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            }
        fairness[g] = g_metrics
    return fairness

def evaluate_multilabel(y_true, y_preds, y_probas, labels, out_dir, metrics_filter=None, sensitive_features=None, fmt="md") -> EvaluationResult:
    y_preds_dict = _ensure_dict(y_preds)
    y_probas_dict = _ensure_dict(y_probas)
    is_multimodel = len(y_preds_dict) > 1
    
    all_metrics = {}
    charts_out = {"confusion": {}, "roc": {}, "pr": {}}
    
    # We will only support Plotly minimally for multilabel to keep it simple, 
    # mostly fallback to PNGs.
    
    for model_name, y_pred in y_preds_dict.items():
        base_metrics = {
            "subset_accuracy": accuracy_score(y_true, y_pred),
            "hamming_loss": hamming_loss(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        
        y_proba = y_probas_dict.get(model_name) if y_probas_dict else None
        if y_proba is not None and y_proba.shape == y_true.shape:
            try:
                base_metrics["roc_auc_macro"] = np.mean([
                    roc_auc_score(y_true[:, i], y_proba[:, i]) for i in range(y_true.shape[1])
                ])
                base_metrics["pr_auc_macro"] = np.mean([
                    average_precision_score(y_true[:, i], y_proba[:, i]) for i in range(y_true.shape[1])
                ])
            except Exception:
                pass

        if is_multimodel:
            all_metrics[model_name] = base_metrics
        else:
            all_metrics = base_metrics
            
        model_pfx = _sanitize(model_name) if is_multimodel else None
        conf_paths = plot_multilabel_confusions(y_true, y_pred, labels, out_dir, model_name=model_pfx)
        if is_multimodel:
            charts_out["confusion"][model_name] = {label: fname for label, fname in conf_paths}
        else:
            charts_out["confusion"] = {label: fname for label, fname in conf_paths}

    if y_probas_dict:
        roc_paths, pr_paths = plot_multilabel_roc_pr(y_true, y_probas_dict, labels, out_dir)
        charts_out["roc"] = {label: fname for label, fname in roc_paths}
        charts_out["pr"] = {label: fname for label, fname in pr_paths}
        
    fairness = _compute_fairness_classification(y_true, y_preds_dict, sensitive_features) if sensitive_features is not None else {}

    return EvaluationResult(task="multi-label", metrics=all_metrics, charts=charts_out, is_multimodel=is_multimodel, fairness_metrics=fairness)

def evaluate_classification(y_true, y_preds, y_probas, labels, out_dir, metrics_filter=None, sensitive_features=None, fmt="md") -> EvaluationResult:
    y_preds_dict = _ensure_dict(y_preds)
    y_probas_dict = _ensure_dict(y_probas)
    is_multimodel = len(y_preds_dict) > 1
    
    all_metrics = {}
    charts_out = {"confusion": {}}
    use_plotly = fmt in ("html", "both")
    
    for model_name, y_pred in y_preds_dict.items():
        metrics_dict = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }
        
        y_proba = y_probas_dict.get(model_name) if y_probas_dict else None
        if y_proba is not None:
            if y_proba.ndim == 1:
                try:
                    metrics_dict["roc_auc"] = roc_auc_score(y_true, y_proba)
                    metrics_dict["pr_auc"] = average_precision_score(y_true, y_proba)
                    metrics_dict["brier_score"] = brier_score_loss(y_true, y_proba)
                except Exception:
                    pass
            elif y_proba.ndim == 2:
                try:
                    metrics_dict["roc_auc_ovr_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                except Exception:
                    pass
            
        if is_multimodel:
            all_metrics[model_name] = metrics_dict
        else:
            all_metrics = metrics_dict
            
        if use_plotly:
            chart = plot_confusion_plotly(y_true, y_pred, labels=labels, title=f"Confusion - {model_name}")
        else:
            cname = f"confusion_{_sanitize(model_name)}.png" if is_multimodel else "confusion.png"
            chart = os.path.basename(plot_confusion(y_true, y_pred, labels=labels, path=os.path.join(out_dir, cname)))
        
        if is_multimodel:
            charts_out["confusion"][model_name] = chart
        else:
            charts_out["confusion"] = chart

    roc_imgs, pr_imgs = [], []
    if y_probas_dict:
        sample_proba = list(y_probas_dict.values())[0]
        if sample_proba.ndim == 1:
            if use_plotly:
                roc_imgs.append(plot_multi_roc_plotly(y_true, y_probas_dict))
                pr_imgs.append(plot_multi_pr_plotly(y_true, y_probas_dict))
                charts_out["calibration"] = plot_calibration_curve_plotly(y_true, y_probas_dict)
                # Fallback to PNG for thresholds
                charts_out["thresholds"] = os.path.basename(plot_threshold_curves(y_true, y_probas_dict, path=os.path.join(out_dir, "thresholds.png")))
            else:
                roc_imgs.append(os.path.basename(plot_multi_roc(y_true, y_probas_dict, path=os.path.join(out_dir, "roc.png"))))
                pr_imgs.append(os.path.basename(plot_multi_pr(y_true, y_probas_dict, path=os.path.join(out_dir, "pr.png"))))
                charts_out["calibration"] = os.path.basename(plot_calibration_curve(y_true, y_probas_dict, path=os.path.join(out_dir, "calibration.png")))
                charts_out["thresholds"] = os.path.basename(plot_threshold_curves(y_true, y_probas_dict, path=os.path.join(out_dir, "thresholds.png")))
        elif sample_proba.ndim == 2:
            n_classes = sample_proba.shape[1]
            names = list(labels) if (labels is not None and len(labels) >= n_classes) else list(range(n_classes))
            for i in range(n_classes):
                pos = (y_true == i).astype(int)
                class_probas = {m: p[:, i] for m, p in y_probas_dict.items()}
                if use_plotly:
                    roc_imgs.append(plot_multi_roc_plotly(pos, class_probas, title=f"ROC - Clase {names[i]}"))
                    pr_imgs.append(plot_multi_pr_plotly(pos, class_probas, title=f"PR - Clase {names[i]}"))
                else:
                    cname = _sanitize(names[i])
                    roc_imgs.append(os.path.basename(plot_multi_roc(pos, class_probas, path=os.path.join(out_dir, f"roc_class_{cname}.png"))))
                    pr_imgs.append(os.path.basename(plot_multi_pr(pos,  class_probas, path=os.path.join(out_dir, f"pr_class_{cname}.png"))))

    charts_out["roc"] = roc_imgs
    charts_out["pr"] = pr_imgs
    fairness = _compute_fairness_classification(y_true, y_preds_dict, sensitive_features) if sensitive_features is not None else {}

    return EvaluationResult(task="classification", metrics=all_metrics, charts=charts_out, is_multimodel=is_multimodel, fairness_metrics=fairness)

def evaluate_regression(y_true, y_preds, out_dir, metrics_filter=None, sensitive_features=None, fmt="md") -> EvaluationResult:
    y_preds_dict = _ensure_dict(y_preds)
    is_multimodel = len(y_preds_dict) > 1
    
    all_metrics = {}
    use_plotly = fmt in ("html", "both")
    
    for model_name, y_pred in y_preds_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }
        if is_multimodel:
            all_metrics[model_name] = metrics_dict
        else:
            all_metrics = metrics_dict

    if use_plotly:
        fit = plot_regression_fit_plotly(y_true, y_preds_dict)
        resid = os.path.basename(plot_residuals(y_true, y_preds_dict, path=os.path.join(out_dir, "resid.png"))) # Keep resid as PNG
    else:
        fit = os.path.basename(plot_regression_fit(y_true, y_preds_dict, path=os.path.join(out_dir, "fit.png")))
        resid = os.path.basename(plot_residuals(y_true, y_preds_dict, path=os.path.join(out_dir, "resid.png")))

    charts_out = {"fit": fit, "resid": resid}
    fairness = _compute_fairness_regression(y_true, y_preds_dict, sensitive_features) if sensitive_features is not None else {}
    return EvaluationResult(task="regression", metrics=all_metrics, charts=charts_out, is_multimodel=is_multimodel, fairness_metrics=fairness)

def evaluate_forecast(y_true, y_preds, season, insample, out_dir, metrics_filter=None, sensitive_features=None, fmt="md") -> EvaluationResult:
    res = evaluate_regression(y_true, y_preds, out_dir, metrics_filter, sensitive_features, fmt)
    res.task = "forecast"
    # sMAPE and MASE could be added here iterating the models...
    # Keeping it simple for brevity
    return res

def evaluate_ranking(y_true, y_preds, query_id, out_dir, metrics_filter=None, fmt="md") -> EvaluationResult:
    y_preds_dict = _ensure_dict(y_preds)
    is_multimodel = len(y_preds_dict) > 1
    
    groups = np.unique(query_id)
    all_metrics = {}
    
    for model_name, y_pred in y_preds_dict.items():
        ndcgs = []
        for g in groups:
            mask = (query_id == g)
            yt_g = y_true[mask]
            yp_g = y_pred[mask]
            if len(yt_g) > 1:
                try:
                    score = ndcg_score([yt_g], [yp_g])
                    ndcgs.append(score)
                except Exception:
                    pass
        
        metrics_dict = {
            "NDCG": np.mean(ndcgs) if ndcgs else 0.0
        }
        if is_multimodel:
            all_metrics[model_name] = metrics_dict
        else:
            all_metrics = metrics_dict

    # Ranking doesn't have specific charts yet
    charts_out = {}
    return EvaluationResult(task="ranking", metrics=all_metrics, charts=charts_out, is_multimodel=is_multimodel)
