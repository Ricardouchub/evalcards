import numpy as np, os, tempfile
from evalcards import make_report

def test_multiclass_with_custom_labels():
    rng = np.random.default_rng(42)
    n, k = 80, 4
    logits = rng.normal(size=(n, k))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp/exp.sum(axis=1, keepdims=True)
    y_true = rng.integers(0, k, size=n)
    y_pred = proba.argmax(axis=1)
    labels = ["Perro", "Gato", "Ave", "Pez"]
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y_true, y_pred, y_proba=proba,
                          labels=labels, path=os.path.join(d, "multiclass.md"), title="Multiclase Custom Labels")
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        # Las etiquetas personalizadas aparecen en el reporte
        for lbl in labels:
            assert lbl in content
        # Los gráficos por clase existen
        for lbl in labels:
            for chart_type in ["roc_class", "pr_class"]:
                fname = f"{chart_type}_{lbl}.png".replace(" ", "_")
                assert os.path.exists(os.path.join(d, fname))