import numpy as np, os, tempfile
from evalcards import make_report

def test_forecast_mase_no_insample():
    # Testea MASE con fallback (sin insample, season=1)
    rng = np.random.default_rng(0)
    y = 10 + 0.1*np.arange(30) + rng.normal(0, 2, 30)
    yhat = y + rng.normal(0, 2, 30)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, yhat, task="forecast", season=1, path=os.path.join(d, "forecast.md"), title="Forecast fallback")
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "MASE" in content
        assert "sMAPE" in content