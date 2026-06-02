import os
import sys
sys.path.insert(0, os.path.abspath('src'))
import numpy as np
from evalcards.report import make_report
from playwright.sync_api import sync_playwright

def generate_html_and_screenshot():
    out_dir = "docs/assets/temp_html"
    os.makedirs(out_dir, exist_ok=True)
    
    rng = np.random.default_rng(42)
    
    y_true_clf = rng.integers(0, 3, 200)
    y_pred_clf = y_true_clf.copy()
    noise = rng.random(200) < 0.2
    y_pred_clf[noise] = rng.integers(0, 3, sum(noise))
    
    y_proba_clf = rng.dirichlet(np.ones(3)*0.5, 200)
    for i in range(200):
        if not noise[i]:
            y_proba_clf[i, :] *= 0.1
            y_proba_clf[i, y_true_clf[i]] = 0.8
        y_proba_clf[i, :] /= y_proba_clf[i, :].sum()
        
    groups = np.where(rng.random(200) > 0.5, "Adulto", "Joven")
    
    html_path = make_report(
        y_true=y_true_clf,
        y_pred={"RandomForest": y_pred_clf},
        y_proba={"RandomForest": y_proba_clf},
        task="classification",
        labels=["Clase A", "Clase B", "Clase C"],
        sensitive_features=groups,
        out_dir=out_dir,
        path="report.html",
        fmt="html",
        title="Reporte de Clasificación"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 1200})
        # Use file URL
        file_url = f"file:///{os.path.abspath(html_path).replace(chr(92), '/')}"
        page.goto(file_url)
        
        # Take screenshot of insights section
        insights = page.locator("#insights")
        if insights.count() > 0:
            insights.screenshot(path="docs/assets/sample_insights.png")
            print("Captured insights.")
            
        # Take screenshot of fairness table section
        fairness = page.locator("#fairness")
        if fairness.count() > 0:
            fairness.screenshot(path="docs/assets/sample_fairness.png")
            print("Captured fairness table.")
            
        # Take screenshot of overall metrics
        metrics = page.locator("#metrics")
        if metrics.count() > 0:
            metrics.screenshot(path="docs/assets/sample_metrics.png")
            print("Captured metrics table.")
            
        browser.close()
        
    import shutil
    shutil.rmtree(out_dir)

if __name__ == "__main__":
    generate_html_and_screenshot()
