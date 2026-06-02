import os
import argparse
from .report import make_report
from .config import load_config

def _load_vec(path):
    if not path:
        return None
    import pandas as pd
    
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".feather", ".arrow"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)
        
    if df.shape[1] > 1:
        return df.to_numpy()
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy()
    for c in ("y_true", "y_pred", "y_proba"):
        if c in df.columns:
            return df[c].to_numpy()
    raise SystemExit(f"No pude inferir la columna en {path} (usa 1 columna o nómbrala y_true/y_pred/y_proba).")

def _load_proba(path):
    if not path:
        return None
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".feather", ".arrow"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)
        
    if df.shape[1] > 1:
        return df.to_numpy()
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy()
    raise SystemExit(f"No pude leer probabilidades desde {path} (usa 1 columna o varias, una por clase).")

def main():
    conf = load_config()

    p = argparse.ArgumentParser(description="Genera reporte de evaluación (Markdown/HTML)")
    p.add_argument("--y_true", required=True, help="CSV/Parquet con y_true")
    p.add_argument("--y_pred", required=True, help="CSV/Parquet con y_pred")
    p.add_argument("--proba", help="CSV/Parquet con y_proba")
    p.add_argument("--class-names", help="Nombres de clases/etiquetas separados por coma", default=None)
    
    p.add_argument("--out", default=conf.get("out", "report.md"))
    p.add_argument("--outdir", help="Carpeta destino", default=conf.get("outdir", None))
    p.add_argument("--title", default=conf.get("title", None))
    p.add_argument("--lang", default=conf.get("lang", "es"), help="Idioma (es/en)")
    p.add_argument("--format", choices=["md", "html", "both"], default=conf.get("format", "md"), help="Formato de salida")
    
    p.add_argument("--task", choices=["auto", "classification", "regression", "forecast", "multi-label"],
                   default=conf.get("task", "auto"), help="Tipo de tarea forzada")
    p.add_argument("--metrics", help="Lista de métricas separadas por coma", default=conf.get("metrics", None))
    p.add_argument("--forecast", action="store_true", help="Tratar como pronóstico")
    p.add_argument("--season", type=int, default=conf.get("season", 1), help="Periodicidad estacional para MASE")
    p.add_argument("--insample", help="CSV/Parquet con serie insample para MASE (opcional)")
    p.add_argument("--export-json", help="Ruta del JSON a generar", default=conf.get("export_json", None))

    args = p.parse_args()

    y_true = _load_vec(args.y_true)
    y_pred = _load_vec(args.y_pred)
    y_proba = _load_proba(args.proba) if args.proba else None
    insample = _load_vec(args.insample) if args.insample else None
    labels = [s.strip() for s in args.class_names.split(",")] if args.class_names else None
    
    metrics = args.metrics
    if isinstance(metrics, str):
        metrics = [s.strip() for s in metrics.split(",")]

    task = args.task
    if args.forecast and args.task == "auto":
        task = "forecast"

    result = make_report(
        y_true, y_pred, y_proba=y_proba,
        path=args.out, title=args.title, out_dir=args.outdir,
        task=task, season=args.season, insample=insample,
        labels=labels, lang=args.lang, metrics=metrics,
        fmt=args.format,
        export_json=args.export_json
    )
    
    if isinstance(result, tuple):
        md_path, info = result
        print(os.path.abspath(md_path))
        if args.export_json:
            json_path = args.export_json if os.path.isabs(args.export_json) else os.path.join(os.path.dirname(os.path.abspath(md_path)), args.export_json)
            print(os.path.abspath(json_path))
    else:
        print(os.path.abspath(result))

if __name__ == "__main__":
    main()
