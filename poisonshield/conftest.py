"""Make the project importable when running pytest from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANLAB_ROOT = ROOT.parent / "cleanlab-master"
if CLEANLAB_ROOT.exists() and str(CLEANLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANLAB_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
