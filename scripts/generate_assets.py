import os
import sys
sys.path.insert(0, os.path.abspath('src'))
import numpy as np
from evalcards.report import make_report
import shutil

def generate_sample_assets():
    out_dir = "docs/assets"
    os.makedirs(out_dir, exist_ok=True)
    
    rng = np.random.default_rng(42)
    
    # Classification Multi-class
    y_true_clf = rng.integers(0, 3, 200)
    y_pred_clf = y_true_clf.copy()
    noise = rng.random(200) < 0.2
    y_pred_clf[noise] = rng.integers(0, 3, sum(noise))
    
    y_proba_clf = rng.dirichlet(np.ones(3)*0.5, 200)
    # Hacer que las probabilidades coincidan en su mayoría con y_true
    for i in range(200):
        if not noise[i]:
            y_proba_clf[i, :] *= 0.1
            y_proba_clf[i, y_true_clf[i]] = 0.8
        y_proba_clf[i, :] /= y_proba_clf[i, :].sum()
        
    y_pred_dict = {
        "RandomForest": y_pred_clf,
        "XGBoost": y_pred_clf.copy() # solo para el ejemplo
    }
    mask = rng.random(200) < 0.1
    y_pred_dict["XGBoost"][mask] = rng.integers(0, 3, sum(mask))
    
    y_proba_dict = {
        "RandomForest": y_proba_clf,
        "XGBoost": y_proba_clf.copy()
    }
    
    groups = np.where(rng.random(200) > 0.5, "Adulto", "Joven")
    
    # Generate the report in a temporary dir and copy the images
    make_report(
        y_true=y_true_clf,
        y_pred=y_pred_dict,
        y_proba=y_proba_dict,
        task="classification",
        labels=["Clase A", "Clase B", "Clase C"],
        sensitive_features=groups,
        out_dir="docs/assets/temp_clf",
        path="report.md",
        title="Clasificación"
    )
    
    # Regression
    y_true_reg = np.linspace(0, 100, 200)
    y_pred_reg1 = y_true_reg + rng.normal(0, 10, 200)
    y_pred_reg2 = y_true_reg + rng.normal(0, 5, 200)
    
    make_report(
        y_true=y_true_reg,
        y_pred={"Modelo A": y_pred_reg1, "Modelo B": y_pred_reg2},
        task="regression",
        out_dir="docs/assets/temp_reg",
        path="report.md",
        title="Regresión"
    )
    
    # Copy images to assets
    shutil.copy("docs/assets/temp_clf/confusion_RandomForest.png", "docs/assets/sample_confusion.png")
    shutil.copy("docs/assets/temp_clf/roc_class_Clase_A.png", "docs/assets/sample_roc.png")
    shutil.copy("docs/assets/temp_reg/fit.png", "docs/assets/sample_fit.png")
    
    # Clean temp dirs
    shutil.rmtree("docs/assets/temp_clf")
    shutil.rmtree("docs/assets/temp_reg")
    
    print("Assets generated successfully at docs/assets/")

if __name__ == "__main__":
    generate_sample_assets()
