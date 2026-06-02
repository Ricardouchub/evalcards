import os
import json
from typing import Dict, Any, Tuple, Union
from jinja2 import Environment, FileSystemLoader
import base64

from .models import EvaluationResult
from .lang import LANG

def _convert_charts_to_b64(charts: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    def _encode(p):
        if not p:
            return ""
        if p.strip().startswith("<div"):
            return p  # It's raw HTML from Plotly
        full = os.path.join(out_dir, p)
        if not os.path.exists(full):
            return ""
        with open(full, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    b64_charts = {}
    for key, value in charts.items():
        if isinstance(value, str):
            b64_charts[key] = _encode(value)
        elif isinstance(value, list):
            b64_charts[key] = [_encode(v) for v in value]
        elif isinstance(value, dict):
            b64_charts[key] = {k: _encode(v) for k, v in value.items()}
    return b64_charts

def generate_report(
    eval_result: EvaluationResult,
    path: str,
    out_dir: str,
    title: str = None,
    lang: str = "es",
    fmt: str = "md",
    export_json: str = None,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    
    if title is None:
        title = "Reporte de Evaluación"
        
    T = LANG.get(lang, LANG["es"])
    
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
    
    render_kwargs = {
        "title": title,
        "task": eval_result.task,
        "metrics": eval_result.metrics,
        "charts": eval_result.charts,
        "T": T,
        "is_multimodel": eval_result.is_multimodel,
        "insights": eval_result.insights,
        "fairness": eval_result.fairness_metrics,
        "ranking": eval_result.ranking_metrics
    }
    
    full_path = os.path.join(out_dir, path)
    
    if fmt in ("html", "both"):
        html_template = env.get_template("report.html.j2")
        charts_b64 = _convert_charts_to_b64(eval_result.charts, out_dir)
        html_kwargs = render_kwargs.copy()
        html_kwargs["charts_b64"] = charts_b64
        html_out = html_template.render(**html_kwargs)
        
        html_path = full_path
        if not html_path.endswith(".html"):
            html_path = os.path.splitext(html_path)[0] + ".html"
            
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_out)
            
        if fmt == "html":
            full_path = html_path
            
    if fmt in ("md", "both"):
        md_template = env.get_template("report.md.j2")
        md_out = md_template.render(**render_kwargs)
        
        md_path = full_path
        if not md_path.endswith(".md"):
            md_path = os.path.splitext(md_path)[0] + ".md"
            
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_out)
            
        if fmt == "md":
            full_path = md_path

    info_dict = None
    if export_json:
        if os.path.isabs(export_json) or os.path.dirname(export_json):
            json_path = export_json
        else:
            json_path = os.path.join(out_dir, export_json)
            
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        
        info_dict = {
            "metrics": eval_result.metrics,
            "charts": eval_result.charts,
            "markdown": os.path.basename(full_path),
            "fairness_metrics": eval_result.fairness_metrics,
            "ranking_metrics": eval_result.ranking_metrics
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(info_dict, f, indent=2, ensure_ascii=False)

    if export_json:
        return full_path, info_dict
    return full_path
