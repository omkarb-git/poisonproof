import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cleanlab_root = ROOT / "cleanlab-master"
if cleanlab_root.exists() and str(cleanlab_root) not in sys.path:
    sys.path.insert(0, str(cleanlab_root))

import importlib.util

app_path = ROOT / "poisonshield" / "app.py"
spec = importlib.util.spec_from_file_location("poisonshield_app", app_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
