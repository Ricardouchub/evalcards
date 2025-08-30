from .report import make_report

# Exponer versión desde el metadata instalado 
try:
    from importlib.metadata import version
    __version__ = version("evalcards")
except Exception:
    __version__ = "0.0.0"

__all__ = ["make_report", "__version__"]