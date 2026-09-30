# Raw data

## `synthetic_adjusted_prices.csv` (default)

Generated deterministically by `scripts/generate_synthetic_data.py` from the
parameters in `src/config.py` (fixed seed). It is **entirely synthetic**:
four hypothetical assets (`asset_a` … `asset_d`) and a synthetic `benchmark`,
roughly six years of business days, fat-tailed Student-t innovations, a
higher-volatility regime, and occasional negative jump shocks.

No column corresponds to a real security, index, or fund. Nothing here is a
record of actual market performance and nothing here is investable.

Regenerate at any time:

```bash
python scripts/generate_synthetic_data.py
```

Columns: `date, asset_a, asset_b, asset_c, asset_d, benchmark`.

## `adjusted_prices.csv` (optional, user-supplied, git-ignored)

To run the same analysis on your own adjusted prices, place a CSV **with the
same wide format and the same column names** at:

```
data/raw/adjusted_prices.csv
```

`src.data.load_prices()` prefers this file automatically when it exists; delete
or rename it to fall back to the synthetic dataset. Requirements:

- one row per trading day, ascending dates in ISO format (`YYYY-MM-DD`);
- strictly positive prices, adjusted for splits and distributions;
- no missing values (forward-fill before saving if necessary);
- the four asset columns and `benchmark` must all be present.

The report text still describes the *synthetic* design; adjust the prose in
`reports/portfolio_risk_review.qmd` if you substitute real data.
