# Tail Risk in a Concentrated Equity Portfolio

A local, reproducible [Quarto](https://quarto.org) research project that renders
an investment-research-style memo on financial risk and probability:
*historical, parametric, bootstrap, and simulation-based* estimates of tail
risk for a deliberately concentrated four-asset portfolio, with Tufte-inspired
figures. Everything runs from a Python virtual environment; nothing needs
cloud services, credentials, or paid data.

> **Disclaimer.** This is an educational example built on **synthetic data**.
> The assets are hypothetical, the "benchmark" is invented, and no number in
> the report describes real market performance. Nothing here is investment
> advice or a recommendation. Past synthetic behaviour is not a forecast.

Primary artefact: `reports/portfolio_risk_review.html` (self-contained HTML).
Optional: `reports/portfolio_risk_review.pdf` via Quarto's bundled Typst.

## Prerequisites (macOS)

| Tool | Why | Check |
|---|---|---|
| Python ≥ 3.11 | analysis and Jupyter kernel | `python3 --version` |
| [`uv`](https://docs.astral.sh/uv/) *(optional)* | fastest environment setup | `uv --version` |
| [Quarto](https://quarto.org/docs/get-started/) ≥ 1.4 | renders the report | `quarto --version` |
| [Positron Desktop](https://positron.posit.co) *(optional)* | IDE with a bundled Quarto | — |

Positron ships its own Quarto binary at
`/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto`. If
`quarto` is not on your `PATH`, either install Quarto (`brew install --cask
quarto`, or the `.pkg` from quarto.org) or export
`QUARTO_BIN=/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto`
before running `scripts/render_report.sh`. Typst is bundled with Quarto, so no
LaTeX installation is required for the PDF.

## Setup with `uv`

```bash
cd portfolio-risk-research
uv sync                       # creates .venv with Python 3.11 and all dependencies
```

## Setup with `venv` and `pip`

```bash
cd portfolio-risk-research
python3.11 -m venv .venv      # any Python >= 3.11 works
source .venv/bin/activate
pip install -r requirements.txt
```

Quarto automatically uses the Jupyter kernel from a `.venv` in the project
root, so no kernel registration is needed.

> **Troubleshooting: "Starting <other-name> kernel…" or a `FileNotFoundError`
> for a Python path you do not recognise.** Quarto starts the *first* Python
> kernelspec that Jupyter lists, and user-level kernelspecs in
> `~/Library/Jupyter/kernels/` come before the project venv. Stale entries
> pointing at deleted environments will hijack the render. Either remove them
> (`jupyter kernelspec list`, then `jupyter kernelspec remove <name>`), or use
> `scripts/render_report.sh`, which puts the project venv first via
> `JUPYTER_PATH`.

## Generate the synthetic data

```bash
.venv/bin/python scripts/generate_synthetic_data.py
```

Writes `data/raw/synthetic_adjusted_prices.csv` with columns
`date, asset_a, asset_b, asset_c, asset_d, benchmark` (about six years of
business days). The file is byte-for-byte reproducible for a given seed
(`--seed N` to change it). The report regenerates the file automatically if
it is missing.

## Render the HTML report

```bash
quarto render reports/portfolio_risk_review.qmd --to html
```

or, with automatic Quarto detection and an optional PDF afterwards:

```bash
scripts/render_report.sh          # HTML, then PDF if Typst works
scripts/render_report.sh --html   # HTML only
```

Output: `reports/portfolio_risk_review.html`. Rendering runs the full analysis
(bootstrap and 60,000 simulated paths) and takes roughly a minute.

## Render the PDF (optional, Typst)

```bash
quarto render reports/portfolio_risk_review.qmd --to typst
```

Output: `reports/portfolio_risk_review.pdf`. If this fails on your machine,
the HTML render is unaffected. Check `quarto check` for the Typst line.

## Open and run in Positron

1. **File → Open Folder…** and choose `portfolio-risk-research`.
2. The interpreter: `.vscode/settings.json` points Positron at
   `.venv/bin/python`, so it is selected automatically once the environment
   exists (`uv sync` or `pip install`). If Positron shows a different
   interpreter, click the interpreter selector (top right) and choose the
   project `.venv`.
3. Open `reports/portfolio_risk_review.qmd`.
4. Click **Render** (or press `⇧⌘K`) and pick **HTML**; the preview opens in
   the Viewer pane. Code cells can also be run interactively with the
   *Run Cell* buttons.
5. Tests: open the Terminal and run `.venv/bin/python -m pytest`, or use the
   Testing pane after selecting the interpreter.

If Positron cannot find Quarto, it is bundled with the app and needs no
configuration; if the render fails with a kernel error, confirm the selected
interpreter is the project `.venv`.

## Use your own adjusted prices

1. Save a CSV in the same wide format, **same column names**, ascending ISO
   dates, strictly positive prices, no gaps, at
   `data/raw/adjusted_prices.csv` (this path is git-ignored).
2. Re-render. `src.data.load_prices()` prefers that file automatically; delete
   or rename it to return to the synthetic data. Details and validation rules
   are in `data/raw/README.md`.
3. Revisit the prose in the report: it describes the *synthetic* design.

## Run the tests

```bash
.venv/bin/python -m pytest
```

Covers return calculations, weighted aggregation, historical VaR/ES on a
known small series, bootstrap shape and seed reproducibility, simulation
shape and sanity properties, and maximum drawdown on a known path.

## Project structure

```
portfolio-risk-research/
  README.md
  pyproject.toml            # project metadata + dependencies (uv / pip)
  requirements.txt          # pip alternative
  .python-version           # 3.11 (used by uv)
  _quarto.yml               # project + HTML/Typst format configuration
  references.bib            # cited literature
  styles.scss               # Tufte-inspired HTML theme
  data/
    raw/README.md           # provenance; how to substitute your own CSV
    raw/synthetic_adjusted_prices.csv   # generated
    derived/.gitkeep
  src/
    config.py               # weights, seeds, DGP parameters, stress scenarios
    data.py                 # synthetic generator, loader, validation
    returns.py              # returns, annualisation, drawdowns, rolling stats
    risk.py                 # VaR / ES (historical, Gaussian, Student-t), bootstrap, stress
    simulation.py           # lognormal, Student-t and block-bootstrap Monte Carlo
    figures.py              # matplotlib figures used by the report
  reports/
    portfolio_risk_review.qmd
  scripts/
    generate_synthetic_data.py
    render_report.sh
  tests/
    test_returns.py  test_risk.py  test_simulation.py
```

## Known modelling limitations

- **Unconditional estimators on non-stationary data.** Volatility and
  correlation vary through the sample; the VaR/ES figures average over calm
  and turbulent regimes. No GARCH-type conditional model is fitted.
- **Univariate portfolio-level tail model.** The four assets are not modelled
  jointly in the tail; rising stress correlations are only addressed through
  scenarios.
- **Constant weights, no costs.** Daily rebalancing at zero cost is assumed
  everywhere.
- **Tail-index and sampling uncertainty.** The Student-t degrees of freedom
  and the 99% quantiles depend on a small number of extreme days; percentile
  bootstrap intervals for quantiles tend to be too narrow, and the bootstrap
  cannot generate losses beyond the worst observed day.
- **Horizon aggregation.** One-year simulations extrapolate daily estimates
  under i.i.d. or block-resampled returns; drift is taken from the sample and
  is itself highly uncertain.
- **Synthetic data.** Parameters were chosen to exhibit fat tails, a
  high-volatility regime and jump days, not to match any market. All results
  change when a real price file is substituted.

## License

MIT. Cited works remain the property of their authors and publishers.
