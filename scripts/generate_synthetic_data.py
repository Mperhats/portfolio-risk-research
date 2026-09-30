#!/usr/bin/env python
"""Regenerate the deterministic synthetic price dataset.

Usage (from the project root):

    python scripts/generate_synthetic_data.py [--seed N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402
from src.data import generate_synthetic_prices, write_synthetic_prices  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=cfg.SEED, help="random seed")
    parser.add_argument("--out", type=Path, default=cfg.SYNTHETIC_PRICES_PATH, help="output CSV path")
    args = parser.parse_args()

    path = write_synthetic_prices(args.out, seed=args.seed)
    frame = generate_synthetic_prices(seed=args.seed)
    print(f"wrote {path.relative_to(cfg.PROJECT_ROOT) if path.is_relative_to(cfg.PROJECT_ROOT) else path}")
    print(f"rows: {len(frame)}  range: {frame['date'].iloc[0].date()} to {frame['date'].iloc[-1].date()}")
    print(frame.set_index("date").iloc[[0, -1]].round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
