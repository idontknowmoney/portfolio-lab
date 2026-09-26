from pathlib import Path

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
    data = yf.download(tickers, period="max", group_by="ticker")
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
