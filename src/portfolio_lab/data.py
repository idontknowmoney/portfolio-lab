from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_lab.utils.logger import get_logger

logger = get_logger(__name__)

PRICES_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices.parquet"


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Download historical daily stock prices for the given tickers.

    Parameters:
    tickers (list[str]): A list of stock ticker symbols.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock prices.
    """
    data = yf.download(tickers, period="max", group_by="ticker", auto_adjust=True)
    return data


def load_prices(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """
    Load historical stock prices for the given tickers from a local file or
    download them if not available.

    Parameters:
    tickers (list[str]): A list of stock ticker symbols.
    refresh (bool): If True, download the prices even if they are already saved locally.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock prices.
    """
    data = None
    if not refresh:
        try:
            data = pd.read_parquet(PRICES_PATH)
        except FileNotFoundError:
            pass
        else:
            missing = set(tickers) - set(data.columns.get_level_values(0))
            if missing:
                logger.info("Local prices lack %s, re-downloading.", sorted(missing))
                data = None
            else:
                logger.info("Loaded prices from local file.")

    if data is None:
        logger.info("Downloading prices...")
        data = download_prices(tickers)
        PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(PRICES_PATH)
        logger.info("Prices downloaded and saved to local file.")
    return data


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily stock prices to monthly returns.

    Parameters:
    prices (pd.DataFrame): A DataFrame containing daily stock prices.

    Returns:
    pd.DataFrame: A DataFrame containing monthly returns.
    """
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def series_monthly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly returns computed independently per column.

    Unlike `to_monthly_returns`, nothing is dropped across columns, so series with different
    start dates (or exchange calendars) keep all of their own months. Months without a price
    are NaN.

    Parameters:
    closes (pd.DataFrame): Daily adjusted closes, one column per series.

    Returns:
    pd.DataFrame: Month-end indexed returns, NaN before each series starts.
    """
    monthly_prices = closes.resample("ME").last()
    return monthly_prices.pct_change(fill_method=None)


def to_eur(usd_returns: pd.Series, eurusd_returns: pd.Series) -> pd.Series:
    """
    Convert USD returns to EUR returns.

    `eurusd_returns` are the returns of EURUSD (USD per EUR): when it rises the dollar weakens,
    which lowers the EUR return of a USD asset.

    Parameters:
    usd_returns (pd.Series): Monthly returns in USD.
    eurusd_returns (pd.Series): Monthly returns of the EURUSD rate.

    Returns:
    pd.Series: Monthly returns in EUR, indexed like `usd_returns` (NaN where FX is missing).
    """
    fx = eurusd_returns.reindex(usd_returns.index)
    return (1 + usd_returns) / (1 + fx) - 1


def blend(returns: pd.DataFrame, weights: Sequence[float]) -> pd.Series:
    """
    Weighted sum of monthly returns (i.e. rebalanced monthly), using only the months where
    every component has data.

    Parameters:
    returns (pd.DataFrame): Monthly returns, one column per component.
    weights (Sequence[float]): Weight per column, in column order.

    Returns:
    pd.Series: Blended monthly returns.
    """
    return returns.dropna() @ np.asarray(weights, dtype=float)


def splice(primary: pd.Series, proxy: pd.Series) -> pd.Series:
    """
    Use `proxy` returns strictly before the first valid month of `primary`, and `primary` from
    then on. The two never overlap, so there is no duplicated month and no gap at the seam
    (as long as the proxy covers the month before `primary` starts).

    Parameters:
    primary (pd.Series): Monthly returns of the real asset.
    proxy (pd.Series): Monthly returns of the stand-in.

    Returns:
    pd.Series: The spliced series, valid months only.
    """
    first = primary.first_valid_index()
    if proxy.empty:
        return primary.dropna()
    if first is None:
        return proxy.dropna()
    head = proxy[proxy.index < first].dropna()
    tail = primary[primary.index >= first].dropna()
    return pd.concat([head, tail]).sort_index()
