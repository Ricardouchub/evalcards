import numpy as np, os, tempfile
from evalcards import make_report

def test_outdir_and_labels():
    y = np.array([0,1,2,1,0,2,2])
    yp = np.array([0,1,1,1,0,2,2])
    proba = np.eye(3)[yp]
    with tempfile.TemporaryDirectory() as d:
        labels = ["A", "B", "C"]
        out = make_report(y, yp, y_proba=proba, labels=labels, path="rep.md", out_dir=d, lang="en")
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        # Verifica que los nombres personalizados aparecen
        for lbl in labels:
            assert lbl in content
        # Verifica que los gráficos se generan en el out_dir
        for fname in ["confusion.png", "roc_class_A.png", "pr_class_A.png"]:
            assert os.path.exists(os.path.join(d, fname))