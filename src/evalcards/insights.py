from typing import List, Dict, Any
from .models import EvaluationResult

def generate_insights(eval_result: EvaluationResult) -> List[str]:
    insights = []
    
    if not eval_result.metrics:
        return insights
        
    metrics = eval_result.metrics
    is_multi = eval_result.is_multimodel
    task = eval_result.task
    
    if is_multi:
        models = list(metrics.keys())
        if task in ("classification", "multi-label"):
            key = "f1_macro" if "f1_macro" in metrics[models[0]] else "accuracy"
            best_model = max(models, key=lambda m: metrics[m].get(key, 0))
            best_val = metrics[best_model].get(key, 0)
            insights.append(f"🏆 **{best_model}** es el mejor modelo según {key} ({best_val:.3f}).")
            
            # Check calibration
            briers = {m: metrics[m].get("brier_score") for m in models if "brier_score" in metrics[m]}
            if briers:
                best_calibrated = min(briers, key=briers.get)
                if best_calibrated != best_model:
                    insights.append(f"⚖️ Aunque {best_model} tiene mejor rendimiento puro, **{best_calibrated}** está mejor calibrado (Brier Score más bajo).")
        
        elif task in ("regression", "forecast"):
            key = "RMSE" if "RMSE" in metrics[models[0]] else "MAE"
            best_model = min(models, key=lambda m: metrics[m].get(key, float('inf')))
            best_val = metrics[best_model].get(key, 0)
            insights.append(f"🏆 **{best_model}** tiene el menor error según {key} ({best_val:.3f}).")
            
    else:
        # Single model insights
        if task == "classification":
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_macro", 0)
            if acc - f1 > 0.1:
                insights.append("⚠️ Hay una diferencia significativa entre Accuracy y F1-Score Macro. Esto sugiere un **desbalance importante en las clases**.")
            
            auc = metrics.get("roc_auc", metrics.get("roc_auc_ovr_macro"))
            if auc is not None:
                if auc < 0.6:
                    insights.append("🚨 El valor de ROC AUC es bastante bajo (< 0.6). El modelo tiene un bajo poder predictivo discriminatorio.")
                elif auc > 0.95:
                    insights.append("✨ El valor de ROC AUC es sobresaliente (> 0.95). Verifica que no haya *data leakage* (fuga de datos).")
                    
        elif task == "ranking":
            ndcg = metrics.get("NDCG", 0)
            if ndcg > 0.9:
                insights.append("🌟 Excelente ranking (NDCG > 0.9). Los elementos relevantes están apareciendo en las primeras posiciones.")
                
    if eval_result.fairness_metrics:
        insights.append("🔍 **Análisis de Equidad (Fairness)** detectado: revisa la tabla segmentada para asegurar que ningún subgrupo esté en desventaja.")
        
    return insights
