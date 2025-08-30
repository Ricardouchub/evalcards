import numpy as np, os, tempfile
from evalcards import make_report

def test_classification_binary():
    y = np.array([0,1,0,1,1,0])
    p = np.array([0.2,0.8,0.4,0.6,0.7,0.3])
    yp = (p>=0.5).astype(int)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, yp, y_proba=p, path=os.path.join(d, "rep.md"), title="bin")
        assert os.path.exists(out)

def test_regression():
    rng = np.random.default_rng(0)
    y = np.linspace(0,100,200); yp = y + rng.normal(0,10,200)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y, yp, path=os.path.join(d, "rep.md"), title="reg")
        assert os.path.exists(out)

def test_multiclase_ovr():
    rng = np.random.default_rng(0); n,k=100,3
    logits = rng.normal(size=(n,k))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp/exp.sum(axis=1, keepdims=True)
    y_true = rng.integers(0,k,size=n); y_pred = proba.argmax(axis=1)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y_true, y_pred, y_proba=proba, path=os.path.join(d, "rep.md"), title="multi")
        assert os.path.exists(out)

def test_forecast_smape_mase():
    rng = np.random.default_rng(0); t = np.arange(240)
    y = 10 + 0.1*t + 5*np.sin(2*np.pi*t/12) + rng.normal(0,1,240)
    y_train, y_test = y[:180], y[180:]; yhat = y_test + rng.normal(0,1.2,y_test.size)
    with tempfile.TemporaryDirectory() as d:
        out = make_report(y_test, yhat, task="forecast", season=12, insample=y_train,
                          path=os.path.join(d, "rep.md"), title="fc")
        assert os.path.exists(out)