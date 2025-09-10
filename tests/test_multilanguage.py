import numpy as np, os, tempfile
from evalcards import make_report

def test_report_in_english():
    y = np.array([0,1,1,0,1])
    yp = np.array([0,1,0,0,1])
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, yp, path=os.path.join(d, "report_en.md"), lang="en", title="Test English")
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        # El texto clave debe estar en inglés
        assert "Task" in content or "**Task:**" in content
        assert "Confusion" in content or "Confusion matrix" in content