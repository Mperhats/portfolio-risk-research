# Tail Risk in a Concentrated Equity Portfolio

A reproducible [Quarto](https://quarto.org) + Python research memo: historical,
parametric, bootstrap and simulation-based tail-risk estimates for a
deliberately concentrated four-asset portfolio, with Tufte-inspired figures.

**Synthetic data, educational only.** The assets and benchmark are invented;
nothing here is market performance, a forecast, or investment advice.

## Run

Needs Python ≥ 3.11, [uv](https://docs.astral.sh/uv/), and Quarto (on PATH, or
the copy bundled with [Positron](https://positron.posit.co) is found automatically).

```bash
uv sync                 # create .venv with every dependency
uv run pytest           # unit tests
uv run render           # reports/portfolio_risk_review.html
```

`uv run render all` also writes `reports/portfolio_risk_review-typst.pdf`
(Typst, bundled with Quarto) and `reports/portfolio_risk_review.pdf` (LaTeX,
skipped if no TeX engine is installed). `uv run data` regenerates the synthetic
prices; the seed and every modelling assumption live in `src/config.py`.

In Positron: open the folder, open `reports/portfolio_risk_review.qmd`, press
Preview. The workspace settings select `.venv` automatically after `uv sync`.

## Your own prices

Save a CSV with columns `date, asset_a, asset_b, asset_c, asset_d, benchmark`
(ascending ISO dates, positive adjusted prices, no gaps) as
`data/raw/adjusted_prices.csv` and re-render. It is git-ignored and takes
precedence over the synthetic file.

## Layout

```
src/        config · data · returns · risk · simulation · figures · cli
reports/    portfolio_risk_review.qmd   (the memo; code hidden in output)
tests/      pytest suite
_quarto.yml · styles.scss · references.bib
```

Limitations are discussed in the memo's *Interpretation, model risk, and
limitations* section. MIT licence.
