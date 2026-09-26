import pandas as pd
import yfinance as yf

from portfolio_lab.utils.logger import get_logger

logger = get_logger(__name__)


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Download historical daily stock prices for the given tickers.

    Parameters:
    tickers (list[str]): A list of stock ticker symbols.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock prices.
    """
    data = yf.download(tickers, group_by="ticker")
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
    try:
        if refresh:
            raise FileNotFoundError
        data = pd.read_parquet("data/raw/prices.parquet")
        logger.info("Loaded prices from local file.")
    except FileNotFoundError:
        logger.info("Downloading prices...")
        data = download_prices(tickers)
        data.to_parquet("data/raw/prices.parquet")
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
