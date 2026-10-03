from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from portfolio_lab.config import Asset, Config
from portfolio_lab.data import blend, series_monthly_returns, splice, to_eur
from portfolio_lab.simulate import simulate

QUANTILES = (5, 10, 25, 50, 75, 95)
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


FX_TICKER = "EURUSD=X"


def required_tickers(assets: list[Asset]) -> list[str]:
    """
    Every ticker `build_returns` needs: the ETFs, their proxy components and, if any proxy is
    quoted in USD, the EURUSD rate. Duplicates are removed, order is kept.
    """
    tickers = [a.ticker for a in assets]
    tickers += [c.ticker for a in assets for c in a.proxy]
    if any(c.currency == "USD" for a in assets for c in a.proxy):
        tickers.append(FX_TICKER)
    return list(dict.fromkeys(tickers))


def _monthly_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Per-series monthly returns from raw yfinance prices (adjusted closes)."""
    return series_monthly_returns(prices.xs("Close", axis=1, level=1)[tickers])


def etf_returns(prices: pd.DataFrame, assets: list[Asset]) -> pd.DataFrame:
    """
    Monthly returns of each asset's own ETF, one column per ticker, NaN before its launch.

    Parameters:
    prices (pd.DataFrame): Raw prices with (ticker, field) columns, as returned by `load_prices`.
    assets (list[Asset]): The portfolio's assets.
    """
    return _monthly_returns(prices, [a.ticker for a in assets])


def proxy_returns(prices: pd.DataFrame, assets: list[Asset]) -> pd.DataFrame:
    """
    Monthly EUR returns of each asset's proxy blend, one column per asset ticker.

    USD components are converted with the EURUSD rate, then the components are blended with
    their weights. Assets without a proxy give an all-NaN column.

    Parameters:
    prices (pd.DataFrame): Raw prices with (ticker, field) columns, as returned by `load_prices`.
    assets (list[Asset]): The portfolio's assets.
    """
    monthly = _monthly_returns(prices, required_tickers(assets))
    out = {}
    for asset in assets:
        if not asset.proxy:
            out[asset.ticker] = pd.Series(np.nan, index=monthly.index)
            continue
        parts = pd.DataFrame(
            {
                c.ticker: (
                    to_eur(monthly[c.ticker], monthly[FX_TICKER])
                    if c.currency == "USD"
                    else monthly[c.ticker]
                )
                for c in asset.proxy
            }
        )
        out[asset.ticker] = blend(parts, [c.weight for c in asset.proxy]).reindex(monthly.index)
    return pd.DataFrame(out)


def spliced_returns(etf: pd.DataFrame, proxy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per asset, proxy returns before the ETF launched and the ETF's own returns after.

    Parameters:
    etf (pd.DataFrame): Output of `etf_returns`.
    proxy (pd.DataFrame): Output of `proxy_returns`.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: The spliced returns (NaN where an asset has no data) and
    a boolean frame of the same shape that is True where the month came from a proxy.
    """
    spliced = pd.DataFrame({t: splice(etf[t], proxy[t]) for t in etf.columns})
    is_proxy = spliced.notna() & ~etf.reindex(spliced.index).notna()
    return spliced, is_proxy


def build_returns(
    prices: pd.DataFrame, assets: list[Asset], use_proxies: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly returns for the simulation: each asset's history (extended with its proxy if
    `use_proxies`), restricted to the months where every asset has data, so the youngest
    series sets the window.

    Parameters:
    prices (pd.DataFrame): Raw prices with (ticker, field) columns, as returned by `load_prices`.
    assets (list[Asset]): The portfolio's assets.
    use_proxies (bool): Splice proxies in before each ETF's launch.

    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: Returns with one column per ticker, and a boolean frame
    marking which months come from a proxy.
    """
    etf = etf_returns(prices, assets)
    if use_proxies:
        spliced, is_proxy = spliced_returns(etf, proxy_returns(prices, assets))
    else:
        spliced, is_proxy = etf, pd.DataFrame(False, index=etf.index, columns=etf.columns)
    common = spliced.dropna()
    return common, is_proxy.loc[common.index]


def history_window(spliced: pd.DataFrame, is_proxy: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """
    Where each asset's history starts, how much of it is proxy, and the common window.

    Parameters:
    spliced (pd.DataFrame): Spliced returns before intersecting assets, from `spliced_returns`.
    is_proxy (pd.DataFrame): Which of those months are proxy, from `spliced_returns`.
    block_size (int): Months per bootstrap block.

    Returns:
    pd.DataFrame: One row per asset, then a "Common window" row, with the first ETF month, the
    first month used, ETF and proxy month counts, and the distinct blocks of `block_size` months
    (months - block_size + 1) the bootstrap can draw from.
    """

    def row(used: pd.Series, from_proxy: pd.Series) -> dict[str, object]:
        etf_months = used & ~from_proxy
        first = lambda m: f"{m.index[m][0]:%Y-%m}" if m.any() else "-"  # noqa: E731
        return {
            "First ETF month": first(etf_months),
            "First month used": first(used),
            "ETF months": int(etf_months.sum()),
            "Proxy months": int((used & from_proxy).sum()),
            "Distinct blocks": max(int(used.sum()) - block_size + 1, 0),
        }

    valid = spliced.notna()
    rows = {t: row(valid[t], is_proxy[t]) for t in spliced.columns}
    rows["Common window"] = row(valid.all(axis=1), is_proxy.any(axis=1))
    return pd.DataFrame(rows).T


def proxy_quality(etf: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """
    How closely each proxy tracked its ETF over the months where both exist.

    Parameters:
    etf (pd.DataFrame): Output of `etf_returns`.
    proxy (pd.DataFrame): Output of `proxy_returns`.

    Returns:
    pd.DataFrame: One row per asset with the overlap months, the monthly return correlation, the
    annualised tracking difference (mean of ETF minus proxy, times 12) and tracking error (std of
    the difference, times sqrt(12)).
    """
    rows = {}
    for t in etf.columns:
        both = pd.concat([etf[t], proxy[t]], axis=1, keys=["etf", "proxy"]).dropna()
        diff = both["etf"] - both["proxy"]
        rows[t] = {
            "Overlap months": len(both),
            "Correlation": both["etf"].corr(both["proxy"]) if len(both) > 1 else np.nan,
            "Tracking diff": diff.mean() * 12,
            "Tracking error": diff.std() * np.sqrt(12),
        }
    return pd.DataFrame(rows).T.astype({"Overlap months": int})


def history_comparison(cfg: Config, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Final-value distribution with only the ETFs' own history versus with proxies spliced in.

    Parameters:
    cfg (Config): The portfolio and simulation configuration (its `use_proxies` is ignored).
    prices (pd.DataFrame): Raw prices with (ticker, field) columns, as returned by `load_prices`.

    Returns:
    pd.DataFrame: One row per history with its first month, length, P5/P50/P95 final value and
    P(loss), the share of paths ending below the total contributed.
    """
    contributed = contributed_path(cfg)
    rows = {}
    for label, use_proxies in {"ETFs only": False, "With proxies": True}.items():
        returns, _ = build_returns(prices, cfg.portfolio.assets, use_proxies)
        total = run_simulation(cfg, returns).sum(axis=2)
        summary = final_value_summary(total, contributed)
        rows[label] = {
            "From": f"{returns.index[0]:%Y-%m}",
            "Months": len(returns),
            **summary[["P5", "P50", "P95", "P(loss)"]],
        }
    return pd.DataFrame(rows).T


def haircut_returns(returns: pd.DataFrame, annual_haircut: float) -> pd.DataFrame:
    """
    Subtract a flat annual haircut (spread evenly over 12 months) from every monthly return.

    Parameters:
    returns (pd.DataFrame): Monthly returns, one column per asset.
    annual_haircut (float): Annual return to remove, e.g. 0.02 for two percentage points.

    Returns:
    pd.DataFrame: Haircut monthly returns.
    """
    return returns - annual_haircut / 12


def sensitivity_to_haircut(
    cfg: Config, returns: pd.DataFrame, haircuts: tuple[float, ...] = (0.0, 0.02, 0.04)
) -> pd.DataFrame:
    """
    Re-run the simulation with progressively lower historical returns.

    Parameters:
    cfg (Config): The portfolio and simulation configuration.
    returns (pd.DataFrame): Historical monthly returns with columns in the config's asset order.
    haircuts (tuple[float, ...]): Annual haircuts to apply; 0.0 is the unadjusted baseline.

    Returns:
    pd.DataFrame: One row per haircut with the P5/P50/P95 final value and P(loss), the share of
    paths ending below the total contributed.
    """
    contributed = contributed_path(cfg)
    rows = {}
    for h in haircuts:
        total = run_simulation(cfg, haircut_returns(returns, h)).sum(axis=2)
        summary = final_value_summary(total, contributed)
        rows[f"-{h:.0%}/yr" if h else "Baseline"] = summary[["P5", "P50", "P95", "P(loss)"]]
    return pd.DataFrame(rows).T


def annualised_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised mean and volatility of monthly returns, per asset.

    Parameters:
    returns (pd.DataFrame): Monthly returns, one column per asset.

    Returns:
    pd.DataFrame: Columns "ann. mean" and "ann. vol", indexed by asset.
    """
    return pd.DataFrame({"ann. mean": returns.mean() * 12, "ann. vol": returns.std() * np.sqrt(12)})


def run_simulation(cfg: Config, returns: pd.DataFrame) -> np.ndarray:
    """
    Run the Monte Carlo simulation described by the config.

    Parameters:
    cfg (Config): The portfolio and simulation configuration.
    returns (pd.DataFrame): Historical monthly returns with columns in the config's asset order.

    Returns:
    np.ndarray: Simulated value per asset, shape (n_paths, n_months + 1, n_assets).
    """
    assets = cfg.portfolio.assets
    sim = cfg.simulation
    return simulate(
        initial_values=np.array([a.initial_value for a in assets]),
        contribution_weights=np.array([a.contribution_weight for a in assets]),
        monthly_contribution=sim.monthly_contribution,
        monthly_returns=returns[[a.ticker for a in assets]].to_numpy(),
        n_months=sim.horizon_years * 12,
        n_paths=sim.n_paths,
        block_size=sim.block_size,
        rebalance=sim.rebalance,
        seed=sim.seed,
    )


def contributions_by_asset(
    cfg: Config, start: Literal["initial_value", "cost_basis"] = "initial_value"
) -> np.ndarray:
    """
    Cumulative amount put into each asset per month: starting amount plus its share of the
    monthly contributions so far.

    Parameters:
    cfg (Config): The portfolio and simulation configuration.
    start (str): Starting amount per asset. "initial_value" is what the asset is worth today,
    "cost_basis" is what was actually paid for it.

    Returns:
    np.ndarray: Shape (n_months + 1, n_assets).
    """
    assets = cfg.portfolio.assets
    n_months = cfg.simulation.horizon_years * 12
    starts = np.array([getattr(a, start) for a in assets])
    weights = np.array([a.contribution_weight for a in assets])
    months = np.arange(n_months + 1)[:, None]
    return starts + cfg.simulation.monthly_contribution * weights * months


def contributed_path(
    cfg: Config, start: Literal["initial_value", "cost_basis"] = "initial_value"
) -> np.ndarray:
    """
    Cumulative amount contributed to the whole portfolio per month (see `contributions_by_asset`).

    Returns:
    np.ndarray: Shape (n_months + 1,).
    """
    return contributions_by_asset(cfg, start).sum(axis=1)


def final_value_summary(
    total: np.ndarray, contributed: np.ndarray, cost: np.ndarray | None = None
) -> pd.Series:
    """
    Percentiles of the final portfolio value alongside the total contributed.

    Parameters:
    total (np.ndarray): Simulated total values, shape (n_paths, n_months + 1).
    contributed (np.ndarray): Cumulative contributions from initial values, shape (n_months + 1,).
    cost (np.ndarray | None): Cumulative contributions from the cost basis, same shape.

    Returns:
    pd.Series: P5..P95 of the final value, plus "Contributed" and "P(loss)" (share of paths
    ending below it). If `cost` is given, also "Cost basis + contrib." and "P(loss vs cost)".
    """
    final = total[:, -1]
    summary = pd.Series(np.percentile(final, QUANTILES), index=[f"P{q}" for q in QUANTILES])
    summary["Contributed"] = contributed[-1]
    summary["P(loss)"] = (final < contributed[-1]).mean()
    if cost is not None:
        summary["Cost basis + contrib."] = cost[-1]
        summary["P(loss vs cost)"] = (final < cost[-1]).mean()
    return summary


def gain_summary(total: np.ndarray, cost: np.ndarray) -> pd.DataFrame:
    """
    Distribution of the gain over cost (final value minus cost basis plus contributions).

    Parameters:
    total (np.ndarray): Simulated total values, shape (n_paths, n_months + 1).
    cost (np.ndarray): Cumulative contributions from the cost basis, shape (n_months + 1,).

    Returns:
    pd.DataFrame: One row per percentile, columns "Gain" (absolute) and "Return" (gain / cost).
    """
    gain = np.percentile(total[:, -1] - cost[-1], QUANTILES)
    return pd.DataFrame(
        {"Gain": gain, "Return": gain / cost[-1] if cost[-1] else np.nan},
        index=[f"P{q}" for q in QUANTILES],
    )


def gain_by_asset(paths: np.ndarray, cost: np.ndarray, names: list[str]) -> pd.DataFrame:
    """
    Final gain over cost per asset.

    Parameters:
    paths (np.ndarray): Simulated values per asset, shape (n_paths, n_months + 1, n_assets).
    cost (np.ndarray): Cumulative contributions per asset, shape (n_months + 1, n_assets).
    names (list[str]): Asset names.

    Returns:
    pd.DataFrame: One row per asset with the final amount invested, the gain at each QUANTILES
    percentile and the median return (NaN where nothing was invested).
    """
    invested = cost[-1]
    gain = paths[:, -1, :] - invested
    gains = pd.DataFrame(
        np.percentile(gain, QUANTILES, axis=0).T,
        index=names,
        columns=[f"P{q} gain" for q in QUANTILES],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        median_return = np.where(invested > 0, gains["P50 gain"] / invested, np.nan)

    df = pd.DataFrame({"Invested": invested, "P50 return": median_return}, index=names)
    return df.join(gains)


def percentiles_over_time(total: np.ndarray) -> pd.DataFrame:
    """
    Percentiles of the total portfolio value at each month.

    Parameters:
    total (np.ndarray): Simulated total values, shape (n_paths, n_months + 1).

    Returns:
    pd.DataFrame: One row per month, columns "P5".."P95".
    """
    return pd.DataFrame(
        np.percentile(total, QUANTILES, axis=0).T, columns=[f"P{q}" for q in QUANTILES]
    )


def _plot_bands(ax: Axes, total: np.ndarray) -> np.ndarray:
    """Draw the P5-P95 and P25-P75 bands and the median of `total` over time, in years."""
    years = np.arange(total.shape[1]) / 12
    pct = percentiles_over_time(total)
    ax.fill_between(years, pct["P5"], pct["P95"], alpha=0.2, label="P5-P95")
    ax.fill_between(years, pct["P25"], pct["P75"], alpha=0.4, label="P25-P75")
    ax.plot(years, pct["P50"], label="Median")
    ax.set_xlabel("Years")
    return years


def plot_fan_chart(
    total: np.ndarray,
    contributed: np.ndarray,
    currency: str,
    log_scale: bool = False,
    cost: np.ndarray | None = None,
) -> Figure:
    """
    Percentile bands of simulated portfolio value over time, with contributions overlaid.

    The log scale keeps the upper paths from flattening everything else over long horizons.
    If `cost` is given, the cost basis + contributions line is drawn too.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    years = _plot_bands(ax, total)
    ax.plot(years, contributed, "k--", label="Starting value + contributions")
    if cost is not None:
        ax.plot(years, cost, "k:", label="Cost basis + contributions")
    ax.set_ylabel(f"Portfolio value ({currency})")
    ax.set_title("Simulated portfolio value" + (" (log scale)" if log_scale else ""))
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    return fig


def plot_gain_fan_chart(total: np.ndarray, cost: np.ndarray, currency: str) -> Figure:
    """Percentile bands of the gain over cost basis + contributions over time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_bands(ax, total - cost)
    ax.axhline(0, color="k", linestyle=":", label="Break-even")
    ax.set_ylabel(f"Gain over cost ({currency})")
    ax.set_title("Simulated gain over cost basis")
    ax.legend()
    return fig


def weights_summary(paths: np.ndarray, names: list[str]) -> pd.DataFrame:
    """
    Each asset's share of the portfolio value at the start and the P5/P50/P95 at the end.

    Parameters:
    paths (np.ndarray): Simulated asset values, shape (n_paths, n_months + 1, n_assets).
    names (list[str]): Asset names, in the same order as the last axis of `paths`.

    Returns:
    pd.DataFrame: One row per asset, columns "Start", "P5", "P50", "P95" (shares, 0-1).
    """
    weights = paths / paths.sum(axis=2, keepdims=True)
    p5, p50, p95 = np.percentile(weights[:, -1], [5, 50, 95], axis=0)
    return pd.DataFrame({"Start": weights[0, 0], "P5": p5, "P50": p50, "P95": p95}, index=names)


def plot_weights_over_time(paths: np.ndarray, names: list[str]) -> Figure:
    """
    Median and P5-P95 band of each asset's share of the portfolio value over time.

    Parameters:
    paths (np.ndarray): Simulated asset values, shape (n_paths, n_months + 1, n_assets).
    names (list[str]): Asset names, in the same order as the last axis of `paths`.
    """
    weights = paths / paths.sum(axis=2, keepdims=True)
    p5, p50, p95 = np.percentile(weights, [5, 50, 95], axis=0)
    years = np.arange(weights.shape[1]) / 12
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        (line,) = ax.plot(years, p50[:, i], label=name)
        ax.fill_between(years, p5[:, i], p95[:, i], color=line.get_color(), alpha=0.2)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Years")
    ax.set_ylabel("Share of portfolio value")
    ax.set_title("Asset weights over time (median and P5-P95)")
    ax.legend()
    return fig


def plot_final_histogram(
    total: np.ndarray, contributed: np.ndarray, currency: str, cost: np.ndarray | None = None
) -> Figure:
    """Histogram of final portfolio values, with total contributed (and cost, if given) marked."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(total[:, -1], bins=60)
    ax.axvline(contributed[-1], color="k", linestyle="--", label="Starting value + contributions")
    if cost is not None:
        ax.axvline(cost[-1], color="k", linestyle=":", label="Cost basis + contributions")
    ax.set_xlabel(f"Final value ({currency})")
    ax.set_ylabel("Paths")
    ax.legend()
    return fig


def portfolio_return_series(cfg: Config, returns: pd.DataFrame) -> pd.Series:
    """
    Single monthly return series for the whole portfolio, weighting each asset by its share of
    the starting value (constant weights, i.e. rebalanced monthly).

    Parameters:
    cfg (Config): The portfolio and simulation configuration.
    returns (pd.DataFrame): Historical monthly returns with columns in the config's asset order.

    Returns:
    pd.Series: Weighted monthly returns, same index as `returns`.
    """
    assets = cfg.portfolio.assets
    values = np.array([a.initial_value for a in assets])
    weights = values / values.sum() if values.sum() else np.full(len(assets), 1 / len(assets))
    return pd.Series(
        returns[[a.ticker for a in assets]].to_numpy() @ weights, index=returns.index, name="return"
    )


def final_value_of_sequence(
    series: np.ndarray, initial_value: float, monthly_contribution: float
) -> np.ndarray:
    """
    Portfolio value over time when replaying `series` in order, contributing at the start of
    each month (same timing as `simulate`).

    Returns:
    np.ndarray: Shape (len(series) + 1,).
    """
    values = np.empty(len(series) + 1)
    values[0] = v = initial_value
    for t, r in enumerate(series):
        v = (v + monthly_contribution) * (1 + r)
        values[t + 1] = v
    return values


def worst_window_start(series: np.ndarray, window: int = 12) -> int:
    """Index where the `window`-month block with the lowest compounded return starts."""
    growth = np.array(
        [np.prod(1 + series[i : i + window]) for i in range(len(series) - window + 1)]
    )
    return int(growth.argmin())


def sequence_risk_scenarios(
    series: pd.Series, initial_value: float, monthly_contribution: float, window: int = 12
) -> dict[str, np.ndarray]:
    """
    Replay the same monthly returns in different orders and track the portfolio value.

    The returns are identical in every scenario, so without contributions the final value would
    be too. With contributions, a bad stretch late in the horizon hits a bigger pot than an
    early one, so the final values differ.

    Parameters:
    series (pd.Series): Monthly returns, in historical order.
    initial_value (float): Starting portfolio value.
    monthly_contribution (float): Total contribution added at the start of each month.
    window (int): Length in months of the "worst block" that is moved around.

    Returns:
    dict[str, np.ndarray]: Value path per scenario: "Historical order", "Reversed",
    "Worst block first" and "Worst block last".
    """
    r = series.to_numpy()
    if not 1 <= window < len(r):
        raise ValueError("window must be between 1 and the number of months minus one.")
    i = worst_window_start(r, window)
    block, rest = r[i : i + window], np.concatenate([r[:i], r[i + window :]])
    orders = {
        "Historical order": r,
        "Reversed": r[::-1],
        "Worst block first": np.concatenate([block, rest]),
        "Worst block last": np.concatenate([rest, block]),
    }
    return {
        name: final_value_of_sequence(o, initial_value, monthly_contribution)
        for name, o in orders.items()
    }


def sequence_risk_summary(scenarios: dict[str, np.ndarray]) -> pd.Series:
    """Final value of each sequence-risk scenario."""
    return pd.Series({name: path[-1] for name, path in scenarios.items()}, name="Final value")


def plot_sequence_risk(scenarios: dict[str, np.ndarray], currency: str) -> Figure:
    """Portfolio value over time for each ordering of the same returns."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, path in scenarios.items():
        ax.plot(np.arange(len(path)) / 12, path, label=name)
    ax.set_xlabel("Years")
    ax.set_ylabel(f"Portfolio value ({currency})")
    ax.set_title("Sequence risk: same returns, different order")
    ax.legend()
    return fig


def save_figure(fig: Figure, name: str, dpi: int = 200) -> Path:
    """
    Save a figure as a PNG in the project's figures/ directory.

    Parameters:
    fig (Figure): The figure to save.
    name (str): File name without extension.
    dpi (int): Resolution of the exported image.

    Returns:
    Path: The written file.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
