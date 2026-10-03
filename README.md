# portfolio-lab

Monte Carlo projections for an index-fund portfolio. Given your assets, a monthly contribution and a
horizon, it resamples blocks of real historical monthly returns (block bootstrap) to simulate
thousands of possible futures, then summarises the spread of outcomes: percentile bands over time,
the distribution of the final value, and the gain over what you actually paid.

Prices come from Yahoo Finance via `yfinance`. Nothing is sent anywhere: your portfolio config
stays on your machine.

## What it does

- Loads and validates the portfolio and simulation settings from a TOML file (`pydantic`).
- Downloads daily adjusted prices for every ticker, caches them as Parquet and converts them to monthly returns.
- Extends short ETF histories with proxy series (e.g. SPY/EFA for MSCI World), converted from USD to
  EUR, so the resampled window reaches back to 2005 and includes 2008-09
  (see [Extending the history with proxies](#extending-the-history-with-proxies)).
- Simulates `n_paths` portfolios over `horizon_years`. Each path stitches together `block_size`-month
  blocks of historical returns, which keeps some of the real autocorrelation between months. The
  monthly contribution is split across assets by `contribution_weight` and added at the start of each month.
- Reports percentiles (P5 to P95) of the final value, the probability of ending below what you put in,
  and gain per asset against its cost basis.
- Stress-tests the result: compares ETF-only against extended history, and re-runs with an annual
  return haircut (e.g. -2%/yr, -4%/yr).
- Demonstrates sequence risk: replays a horizon-long stretch of historical returns (the one containing
  the worst 12-month block) reversed and with that block first or last. The returns are identical
  but, with monthly contributions, the final values are not (a late crash hurts far more than an early one).
- Plots fan charts, a final-value histogram, a gain fan chart, asset weights over time and the
  sequence-risk scenarios, saved as PNGs in `figures/`.

Rebalancing is not implemented yet: `rebalance` must be `"none"`.

## Project structure

```
portfolio-lab/
├── config/
│   └── portfolio_example.toml   # template; copy to portfolio.toml (git-ignored)
├── notebooks/
│   └── 01_monte_carlo.ipynb     # end-to-end walkthrough: config -> data -> simulation -> charts
├── src/portfolio_lab/
│   ├── config.py                # pydantic models + load_config() (falls back to the example file)
│   ├── data.py                  # download/cache prices, monthly returns, USD->EUR, blend, splice
│   ├── simulate.py              # block-bootstrap simulation engine (numpy)
│   ├── analysis.py              # proxy history, run_simulation, summaries, sensitivity, plots
│   └── utils/logger.py
├── tests/unit/                  # pytest suite
├── data/raw/                    # cached prices.parquet (git-ignored, created on first run)
├── figures/                     # exported charts (git-ignored)
├── .github/workflows/ci.yml     # ruff + pytest on every PR
└── pyproject.toml
```

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) (it installs the right Python for you; the project
needs Python 3.14).

```bash
git clone git@github.com:idontknowmoney/portfolio-lab.git
cd portfolio-lab
uv sync                      # creates .venv and installs runtime + dev dependencies
uv run pre-commit install    # optional: lint/format hooks on commit and push
```

Set up your portfolio:

```bash
cp config/portfolio_example.toml config/portfolio.toml
```

Then edit `config/portfolio.toml`. If the file is missing, the example config is used and a
warning is logged.

```toml
[[portfolio.assets]]
name = "MSCI World"
isin = "IE00B4L5Y983"
ticker = "EUNL.DE"            # Yahoo Finance ticker
contribution_weight = 0.70    # weights across assets must sum to 1
initial_value = 0             # what the asset is worth today
cost_basis = 0                # what you actually paid for it
proxy = [                     # optional: stand-in history from before the ETF launched
  { ticker = "SPY", currency = "USD", weight = 0.7 },
  { ticker = "EFA", currency = "USD", weight = 0.3 },
]

[simulation]
monthly_contribution = 200
horizon_years = 30
n_paths = 10000
block_size = 12               # months per bootstrap block
seed = 42
rebalance = "none"
use_proxies = true            # false: use each ETF's own history only
```

Run the simulation by opening the notebook (in VS Code, or `uv run jupyter lab` if you add Jupyter):
`notebooks/01_monte_carlo.ipynb`. The first run downloads prices into `data/raw/prices.parquet`;
later runs reuse them (pass `refresh=True` to `load_prices` to re-download).

Or use the library directly:

```python
from portfolio_lab.analysis import (
    build_returns,
    contributed_path,
    final_value_summary,
    required_tickers,
    run_simulation,
)
from portfolio_lab.config import load_config
from portfolio_lab.data import load_prices

cfg = load_config()
assets = cfg.portfolio.assets
prices = load_prices(required_tickers(assets))
returns, _ = build_returns(prices, assets, cfg.simulation.use_proxies)

paths = run_simulation(cfg, returns)  # (n_paths, n_months + 1, n_assets)
total = paths.sum(axis=2)
print(final_value_summary(total, contributed_path(cfg)))
```

## Extending the history with proxies

Each ETF only has a short history, and the youngest one would otherwise set the window for all of
them (a recent, mostly bull-market window flatters the results). Each asset can list a `proxy`: one
or more tickers with weights summing to 1 and a quote currency (`EUR` or `USD`). Before the ETF
launched, the proxy's returns are used instead (USD converted to EUR with `EURUSD=X`, components
blended with monthly rebalancing); from launch onward the ETF's own returns are used. The two never
overlap. Set `use_proxies = false` under `[simulation]` to use ETF history only.

The notebook shows the window and the proxy share of each asset (`history_window`), how closely each
proxy tracked its ETF (`proxy_quality`: correlation, tracking difference and error), and compares
ETF-only against extended history (`history_comparison`).

## Limitations

- Proxies are not the real funds: a different index (e.g. EEM is standard EM, not EM IMI), no fund
  costs, different currency effects. Tracking differences are reported, not corrected.
- The past is a sample, not a forecast. Use `sensitivity_to_haircut` to see how the results move
  with lower returns.
- No inflation (results are nominal), taxes, fees or rebalancing.
- The latest month in the data may be incomplete if prices were downloaded mid-month.
- The bootstrap only reuses months that happened. It cannot produce a crash worse than anything in
  the window.

## Development

```bash
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format .         # format
```

CI runs the same checks on every pull request.

## Branching & contributing

```
feature/*, fix/*  --PR--> develop --PR (release or single feature)--> main
hotfix/*  (from main) --PR--> main --back-merge PR--> develop
```

- `main` is production, `develop` is the dev environment. Both are protected: no direct pushes, no force pushes, no deletion.
- Every change goes through a pull request with a passing `lint-test` check and 1 approval (repo admins can bypass the approval).
- Branch names: `feature/<x>`, `fix/<x>`, `hotfix/<x>`, `release/<x>`.
- PRs into `main` must come from `develop`, `hotfix/*` or `release/*` (enforced by the `source-branch` check).
- Use a **merge commit** for `develop -> main` PRs so the two branches don't drift. Squash or rebase is fine for `feature/* -> develop`.
- After a hotfix lands on `main`, open a back-merge PR `main -> develop`.
