"""Compatibility shim that exposes the packaged implementation under the historical module name."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_module = importlib.import_module("asd_motion_clusters.utils_cleaned")
sys.modules[__name__] = _module
