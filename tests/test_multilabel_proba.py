import numpy as np, os, tempfile
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from evalcards import make_report

def test_multilabel_with_probabilities():
    X, y = make_multilabel_classification(n_samples=60, n_features=8, n_classes=3, random_state=2)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=500)).fit(X, y)
    y_pred = clf.predict(X)
    y_proba = np.stack([m.predict_proba(X)[:,1] for m in clf.estimators_], axis=1)
    labels = [f"Cat_{i}" for i in range(y.shape[1])]
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, y_pred, y_proba=y_proba, path=os.path.join(d, "mlrep.md"),
                          title="MultiLabelTest", labels=labels, task="multi-label")
        assert os.path.exists(out)
        # Verifica que los PNGs ROC/PR se generan
        for i in range(y.shape[1]):
            for chart_type in ["roc_label", "pr_label", "confusion"]:
                fname = f"{chart_type}_{labels[i]}.png".replace(" ", "_")
                assert os.path.exists(os.path.join(d, fname))
        # Verifica que las métricas ROC/PR están en el Markdown
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "roc_auc_macro" in content
        assert "pr_auc_macro" in content