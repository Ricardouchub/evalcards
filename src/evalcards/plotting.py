import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def plot_confusion(y_true, y_pred, labels=None, path="confusion.png"):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_multilabel_confusions(y_true, y_pred, labels, out_dir, model_name=None):
    paths = []
    prefix = f"{model_name}_" if model_name else ""
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        label = labels[i] if labels and i < len(labels) else f"Cat_{i}"
        fname = f"{prefix}confusion_{label}.png".replace(" ", "_")
        path = os.path.join(out_dir, fname)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(yt, yp, display_labels=[0,1], ax=ax)
        ax.set_title(f"{model_name + ' - ' if model_name else ''}{label}")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append((label, fname))
    return paths

def plot_multi_roc(y_true, y_probas_dict, path="roc.png"):
    fig, ax = plt.subplots()
    for model_name, y_proba in y_probas_dict.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_multi_pr(y_true, y_probas_dict, path="pr.png"):
    fig, ax = plt.subplots()
    for model_name, y_proba in y_probas_dict.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_multilabel_roc_pr(y_true, y_probas_dict, labels, out_dir):
    roc_paths, pr_paths = [], []
    n_labels = y_true.shape[1]
    for i in range(n_labels):
        label = labels[i] if labels and i < len(labels) else f"Cat_{i}"
        
        # Prepare dicts for the current label
        label_probas = {}
        for m_name, y_proba in y_probas_dict.items():
            label_probas[m_name] = y_proba[:, i]
            
        fname_roc = os.path.join(out_dir, f"roc_label_{label}.png".replace(" ", "_"))
        fname_pr = os.path.join(out_dir, f"pr_label_{label}.png".replace(" ", "_"))
        
        try:
            plot_multi_roc(y_true[:, i], label_probas, fname_roc)
            roc_paths.append((label, f"roc_label_{label}.png".replace(" ", "_")))
        except Exception:
            pass
        try:
            plot_multi_pr(y_true[:, i], label_probas, fname_pr)
            pr_paths.append((label, f"pr_label_{label}.png".replace(" ", "_")))
        except Exception:
            pass
    return roc_paths, pr_paths

def plot_regression_fit(y_true, y_preds_dict, path="fit.png"):
    fig, ax = plt.subplots()
    for model_name, y_pred in y_preds_dict.items():
        ax.scatter(y_true, y_pred, s=10, label=model_name, alpha=0.6)
        
    all_preds = [v for v in y_preds_dict.values()]
    min_val = min(y_true.min(), min(p.min() for p in all_preds))
    max_val = max(y_true.max(), max(p.max() for p in all_preds))
    
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    ax.set_xlabel("y real")
    ax.set_ylabel("y predicho")
    ax.set_title("Ajuste")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_residuals(y_true, y_preds_dict, path="resid.png"):
    fig, ax = plt.subplots()
    for model_name, y_pred in y_preds_dict.items():
        resid = y_true - y_pred
        ax.scatter(y_pred, resid, s=10, label=model_name, alpha=0.6)
        
    ax.axhline(0, color='black', linestyle='--')
    ax.set_xlabel("y predicho")
    ax.set_ylabel("residual")
    ax.set_title("Residuales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_calibration_curve(y_true, y_probas_dict, path="calibration.png"):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k:", label="Perfectamente calibrado")
    
    for model_name, y_proba in y_probas_dict.items():
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
            ax.plot(prob_pred, prob_true, "s-", label=model_name)
        except Exception:
            pass
            
    ax.set_xlabel("Probabilidad predicha media")
    ax.set_ylabel("Fracción de positivos")
    ax.set_title("Curva de Confiabilidad (Calibración)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_threshold_curves(y_true, y_probas_dict, path="thresholds.png"):
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots()
    
    for model_name, y_proba in y_probas_dict.items():
        try:
            prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
            # F1 score para cada threshold
            f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
            ax.plot(thresholds, f1, label=f"F1 ({model_name})")
        except Exception:
            pass

    ax.set_xlabel("Umbral (Threshold)")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Umbral")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
