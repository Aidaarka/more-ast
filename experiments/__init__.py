"""MoRe-AST experiments package."""

import os
import sys
from pathlib import Path

# Workspace root (parent of more_ast/)
_MORE_AST_DIR = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _MORE_AST_DIR.parent

root = str(_WORKSPACE_ROOT)


def cdroot():
    """Change working directory to workspace root so relative paths resolve correctly."""
    os.chdir(root)


def setup_paths():
    """Ensure CriSPO and workspace root are on sys.path."""
    crispo_parent = _WORKSPACE_ROOT / "CriSPO"
    if crispo_parent.exists() and str(crispo_parent) not in sys.path:
        sys.path.insert(0, str(crispo_parent))
    if str(_WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(_WORKSPACE_ROOT))
