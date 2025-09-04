import numpy as np, os, tempfile
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from evalcards import make_report

def test_multilabel_basic():
    X, y = make_multilabel_classification(n_samples=40, n_features=10, n_classes=3, random_state=1)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=500)).fit(X, y)
    y_pred = clf.predict(X)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, y_pred, path=os.path.join(d, "rep.md"), title="multilabel",
                          labels=[f"tag_{i}" for i in range(y.shape[1])])
        assert os.path.exists(out)
        # Check Markdown contents
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "subset_accuracy" in content
        assert "hamming_loss" in content
        assert "Confusion tag_0" in content
        assert "Confusion tag_1" in content