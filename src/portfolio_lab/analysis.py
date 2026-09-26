from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from portfolio_lab.config import Config
from portfolio_lab.data import to_monthly_returns
from portfolio_lab.simulate import simulate

QUANTILES = (5, 25, 50, 75, 95)
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


def monthly_returns_from_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Compute monthly returns from raw yfinance prices.

    Only dates where every ticker has data are kept, so the shortest history sets the window.

    Parameters:
    prices (pd.DataFrame): Raw prices with (ticker, field) columns, as returned by `load_prices`.
    tickers (list[str]): Tickers to include, in the desired column order.

    Returns:
    pd.DataFrame: Monthly returns with one column per ticker.
    """
    closes = prices.xs("Close", axis=1, level=1)[tickers].dropna()
    return to_monthly_returns(closes)


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
    pd.DataFrame: One row per asset with the final amount invested, P5/P50/P95 gain and the
    median return (NaN where nothing was invested).
    """
    invested = cost[-1]
    gain = paths[:, -1, :] - invested
    p5, p50, p95 = np.percentile(gain, [5, 50, 95], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        median_return = np.where(invested > 0, p50 / invested, np.nan)
    return pd.DataFrame(
        {
            "Invested": invested,
            "P5 gain": p5,
            "P50 gain": p50,
            "P95 gain": p95,
            "P50 return": median_return,
        },
        index=names,
    )


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
