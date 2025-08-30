from .report import make_report

# versión tomada del metadata instalado (sin duplicar con pyproject.toml)
try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
    __version__ = version("evalcards")
except Exception:
    __version__ = "0.0.0"

__all__ = ["make_report", "__version__"]