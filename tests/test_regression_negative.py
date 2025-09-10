import numpy as np, os, tempfile
from evalcards import make_report

def test_regression_with_negatives():
    # RMSLE no está definido para negativos, pero el reporte no debe romperse
    rng = np.random.default_rng(1)
    y = rng.normal(-10, 2, 40)   # valores negativos
    yp = y + rng.normal(0, 1, 40)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, yp, path=os.path.join(d, "regneg.md"), title="Regresión Negativa")
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        # El reporte contiene métricas principales
        assert "MAE" in content
        assert "RMSLE" in content