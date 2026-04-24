"""Utility functions for MoRe-AST."""

import sys
from pathlib import Path


def load_toml(path: Path) -> dict:
    """Load TOML file. Uses tomllib (3.11+) or tomli."""
    with open(path, "rb") as f:
        if sys.version_info >= (3, 11):
            import tomllib
            return tomllib.load(f)
        try:
            import tomli
            return tomli.load(f)
        except ImportError:
            raise ImportError("Install tomli: pip install tomli (for Python < 3.11)")


def ensure_crispo_path():
    """Add CriSPO to sys.path so crispo imports work."""
    root = Path(__file__).resolve().parent.parent
    crispo_parent = root / "CriSPO"
    if crispo_parent.exists() and str(crispo_parent) not in sys.path:
        sys.path.insert(0, str(crispo_parent))
