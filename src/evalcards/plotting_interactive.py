import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_plotly(y_true, y_pred, labels=None, title="Matriz de Confusión"):
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicho",
        yaxis_title="Real",
        yaxis=dict(autorange="reversed")
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_multi_roc_plotly(y_true, y_probas_dict, title="Curva ROC"):
    from sklearn.metrics import roc_curve, auc
    fig = go.Figure()
    
    fig.add_shape(
        type='line', line=dict(dash='dash', color='black'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    for model_name, y_proba in y_probas_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={roc_auc:.3f})", mode='lines'))
        
    fig.update_layout(
        title=title,
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_multi_pr_plotly(y_true, y_probas_dict, title="Curva Precision-Recall"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig = go.Figure()
    
    for model_name, y_proba in y_probas_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        aps = average_precision_score(y_true, y_proba)
        fig.add_trace(go.Scatter(x=rec, y=prec, name=f"{model_name} (AP={aps:.3f})", mode='lines'))
        
    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain')
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_calibration_curve_plotly(y_true, y_probas_dict, title="Curva de Confiabilidad"):
    from sklearn.calibration import calibration_curve
    fig = go.Figure()
    
    fig.add_shape(
        type='line', line=dict(dash='dot', color='black'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    for model_name, y_proba in y_probas_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name=model_name))
        
    fig.update_layout(
        title=title,
        xaxis_title='Probabilidad predicha media',
        yaxis_title='Fracción de positivos'
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def plot_regression_fit_plotly(y_true, y_preds_dict, title="Ajuste de Regresión"):
    fig = go.Figure()
    
    all_preds = [v for v in y_preds_dict.values()]
    min_val = min(y_true.min(), min(p.min() for p in all_preds))
    max_val = max(y_true.max(), max(p.max() for p in all_preds))
    
    fig.add_shape(
        type='line', line=dict(dash='dash', color='black'),
        x0=min_val, x1=max_val, y0=min_val, y1=max_val
    )
    
    for model_name, y_pred in y_preds_dict.items():
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name=model_name, opacity=0.6))
        
    fig.update_layout(
        title=title,
        xaxis_title='y real',
        yaxis_title='y predicho'
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
