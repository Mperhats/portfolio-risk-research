"""Project commands, exposed through ``uv run``.

    uv run render [html|pdf|typst|all]   render the memo (default: html)
    uv run data                          regenerate the synthetic price file
    uv run pytest                        unit tests
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as cfg
from .data import write_synthetic_prices

REPORT = cfg.PROJECT_ROOT / "reports" / "portfolio_risk_review.qmd"
POSITRON_QUARTO = Path("/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto")


def _quarto() -> str:
    """Quarto binary: $QUARTO_BIN, then PATH, then the copy bundled with Positron."""
    for candidate in (os.environ.get("QUARTO_BIN"), shutil.which("quarto"), str(POSITRON_QUARTO)):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("quarto not found: install it from https://quarto.org or set QUARTO_BIN")


def data() -> None:
    """Write data/raw/synthetic_adjusted_prices.csv (deterministic for the configured seed)."""
    path = write_synthetic_prices()
    print(f"wrote {path.relative_to(cfg.PROJECT_ROOT)}")


def render() -> None:
    """Render the memo with Quarto, always using this environment's Python kernel."""
    target = sys.argv[1] if len(sys.argv) > 1 else "html"
    formats = {"html": ["html"], "pdf": ["pdf"], "typst": ["typst"], "all": ["html", "typst", "pdf"]}.get(target)
    if formats is None:
        sys.exit(__doc__)
    env = os.environ | {
        "QUARTO_PYTHON": sys.executable,  # kernel = this venv
        "JUPYTER_PATH": str(Path(sys.prefix) / "share" / "jupyter"),  # its kernelspec first
    }
    if not cfg.SYNTHETIC_PRICES_PATH.exists():
        data()
    for fmt in formats:
        if fmt == "pdf" and not (shutil.which("lualatex") or shutil.which("xelatex")):
            print("skipping pdf: no LaTeX engine (use `uv run render typst`)")
            continue
        subprocess.run([_quarto(), "render", str(REPORT), "--to", fmt], check=True, env=env, cwd=cfg.PROJECT_ROOT)
