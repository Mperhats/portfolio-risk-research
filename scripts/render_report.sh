#!/usr/bin/env bash
# Render the research memo. HTML always; PDF via Typst (bundled with Quarto)
# when it works; PDF via LaTeX only on request (needs a TeX distribution).
#
# Usage, from the project root:
#   scripts/render_report.sh            # HTML, then reports/portfolio_risk_review-typst.pdf
#   scripts/render_report.sh --html     # HTML only
#   scripts/render_report.sh --pdf      # Typst PDF only
#   scripts/render_report.sh --latex    # LaTeX PDF (reports/portfolio_risk_review.pdf)
#
# Quarto lookup order: $QUARTO_BIN, `quarto` on PATH, then the copy bundled
# with Positron Desktop on macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPORT="reports/portfolio_risk_review.qmd"
MODE="${1:-all}"

find_quarto() {
  if [[ -n "${QUARTO_BIN:-}" && -x "${QUARTO_BIN}" ]]; then echo "$QUARTO_BIN"; return; fi
  if command -v quarto >/dev/null 2>&1; then command -v quarto; return; fi
  local positron="/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto"
  if [[ -x "$positron" ]]; then echo "$positron"; return; fi
  echo ""
}

QUARTO="$(find_quarto)"
if [[ -z "$QUARTO" ]]; then
  cat >&2 <<'MSG'
quarto was not found.

Install it from https://quarto.org/docs/get-started/ (macOS: download the .pkg,
or `brew install --cask quarto`), or point QUARTO_BIN at a quarto binary, e.g.
the one bundled with Positron:

  export QUARTO_BIN=/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto
MSG
  exit 1
fi

# Prefer the project virtual environment for the Jupyter kernel.
if [[ -z "${QUARTO_PYTHON:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then export QUARTO_PYTHON="$ROOT/.venv/bin/python"; fi
fi
# Quarto starts the first Python kernelspec Jupyter lists; put the project
# venv's kernelspec first so stale user-level kernels cannot hijack the render.
if [[ -d ".venv/share/jupyter" ]]; then
  export JUPYTER_PATH="$ROOT/.venv/share/jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
fi

# Make sure the synthetic data exists.
if [[ ! -f data/raw/synthetic_adjusted_prices.csv ]]; then
  "${QUARTO_PYTHON:-python3}" scripts/generate_synthetic_data.py
fi

echo "quarto: $QUARTO ($("$QUARTO" --version))"
echo "python: ${QUARTO_PYTHON:-<quarto default>}"

if [[ "$MODE" == "all" || "$MODE" == "--html" ]]; then
  "$QUARTO" render "$REPORT" --to html
  echo "HTML: reports/portfolio_risk_review.html"
fi

if [[ "$MODE" == "all" || "$MODE" == "--pdf" ]]; then
  if "$QUARTO" typst --version >/dev/null 2>&1; then
    if "$QUARTO" render "$REPORT" --to typst; then
      echo "PDF (Typst): reports/portfolio_risk_review-typst.pdf"
    else
      echo "PDF render failed; the HTML report is unaffected." >&2
      [[ "$MODE" == "--pdf" ]] && exit 1
    fi
  else
    echo "Typst is not available through this Quarto install; skipping PDF." >&2
    [[ "$MODE" == "--pdf" ]] && exit 1
  fi
fi

if [[ "$MODE" == "--latex" ]]; then
  if command -v lualatex >/dev/null 2>&1 || command -v xelatex >/dev/null 2>&1; then
    "$QUARTO" render "$REPORT" --to pdf
    echo "PDF (LaTeX): reports/portfolio_risk_review.pdf"
  else
    echo "No LaTeX engine found; use --pdf for the Typst route or install TinyTeX (quarto install tinytex)." >&2
    exit 1
  fi
fi
