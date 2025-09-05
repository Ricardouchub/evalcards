import os, tempfile, numpy as np, pandas as pd, subprocess, shutil

def test_cli_invocation():
    if shutil.which("evalcards") is None:
        import pytest
        pytest.skip("evalcards CLI not found in PATH")
    y = np.array([0,1,1,0])
    yp = np.array([0,1,0,0])
    df_y = pd.DataFrame({"y_true": y})
    df_yp = pd.DataFrame({"y_pred": yp})
    with tempfile.TemporaryDirectory() as d:
        y_path = os.path.join(d, "y.csv")
        yp_path = os.path.join(d, "yp.csv")
        df_y.to_csv(y_path, index=False)
        df_yp.to_csv(yp_path, index=False)
        out_path = os.path.join(d, "cli_report.md")
        proc = subprocess.run([
            "evalcards",
            "--y_true", y_path, "--y_pred", yp_path, "--out", out_path, "--lang", "en"
        ], capture_output=True)
        print("STDOUT:", proc.stdout.decode())
        print("STDERR:", proc.stderr.decode())
        assert os.path.exists(out_path)
        with open(out_path, encoding="utf-8") as f:
            content = f.read()
        assert "accuracy" in content