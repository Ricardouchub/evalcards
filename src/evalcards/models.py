from dataclasses import dataclass, field
from typing import Dict, Any, List, Union, Optional

@dataclass
class EvaluationResult:
    """Contiene los resultados de la evaluación de un modelo."""
    task: str
    # metrics can be a flat dict for single model, or Dict[model_name, Dict[metric_name, value]] for multi-model
    metrics: Dict[str, Any] = field(default_factory=dict)
    # charts can be filenames, or dicts grouping them by model
    charts: Dict[str, Any] = field(default_factory=dict)
    
    is_multimodel: bool = False
    
    # New features
    insights: List[str] = field(default_factory=list)
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)
    ranking_metrics: Dict[str, Any] = field(default_factory=dict)
