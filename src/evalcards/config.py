import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

def load_config(cwd=None):
    if cwd is None:
        cwd = os.getcwd()
    
    config_path = os.path.join(cwd, ".evalcards.toml")
    if not os.path.exists(config_path):
        return {}
    
    if tomllib is None:
        print("Warning: .evalcards.toml found but 'tomli' is not installed (required for Python < 3.11). Ignorado.", file=sys.stderr)
        return {}

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("evalcards", data) # Apoya tanto [evalcards] como llaves planas
    except Exception as e:
        print(f"Error loading .evalcards.toml: {e}", file=sys.stderr)
        return {}
