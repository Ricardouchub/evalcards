import numpy as np, os, tempfile, json
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from evalcards import make_report

def test_export_json_classification_binary():
    y = np.array([0,1,0,1,1,0])
    p = np.array([0.2,0.8,0.4,0.6,0.7,0.3])
    yp = (p>=0.5).astype(int)
    with tempfile.TemporaryDirectory() as d:
        md_path = os.path.join(d, "rep.md")
        json_path = os.path.join(d, "rep.json")
        res = make_report(y, yp, y_proba=p, path=md_path, title="bin", export_json=json_path)
        # API returns (md_path, info) when export_json is provided
        assert isinstance(res, tuple)
        md, info = res
        assert os.path.exists(md)
        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "metrics" in data and "charts" in data and "markdown" in data
        assert "accuracy" in data["metrics"]

def test_export_json_multilabel():
    X, y = make_multilabel_classification(n_samples=40, n_features=6, n_classes=3, random_state=1)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=500)).fit(X, y)
    y_pred = clf.predict(X)
    y_proba = np.stack([m.predict_proba(X)[:,1] for m in clf.estimators_], axis=1)
    with tempfile.TemporaryDirectory() as d:
        md_path = os.path.join(d, "ml.md")
        json_path = os.path.join(d, "ml.json")
        res = make_report(y, y_pred, y_proba=y_proba, path=md_path, title="ml", labels=[f"tag_{i}" for i in range(y.shape[1])], export_json=json_path)
        assert isinstance(res, tuple)
        md, info = res
        assert os.path.exists(md)
        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "subset_accuracy" in data["metrics"] or "hamming_loss" in data["metrics"]
        assert "confusion" in data["charts"]
        # Confusion mapping should contain our tags
        assert any(k.startswith("tag_") for k in data["charts"]["confusion"].keys())

def test_export_json_regression():
    rng = np.random.default_rng(0)
    y = np.linspace(0,100,50); yp = y + rng.normal(0,5,50)
    with tempfile.TemporaryDirectory() as d:
        md_path = os.path.join(d, "reg.md")
        json_path = os.path.join(d, "reg.json")
        res = make_report(y, yp, path=md_path, title="reg", export_json=json_path)
        assert isinstance(res, tuple)
        md, info = res
        assert os.path.exists(md)
        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "MAE" in data["metrics"]
        assert "fit" in data["charts"] and "resid" in data["charts"]