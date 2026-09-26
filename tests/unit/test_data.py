import pandas as pd
import pytest

from portfolio_lab import data


@pytest.fixture
def prices():
    idx = pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29", "2024-03-29"])
    return pd.DataFrame({"AAA": [100.0, 110.0, 120.0, 121.0, 133.1]}, index=idx)


def test_download_prices_calls_yfinance(monkeypatch, prices):
    calls = {}

    def fake_download(tickers, **kwargs):
        calls["tickers"] = tickers
        calls["kwargs"] = kwargs
        return prices

    monkeypatch.setattr(data.yf, "download", fake_download)

    result = data.download_prices(["AAA"])

    assert result is prices
    assert calls["tickers"] == ["AAA"]
    assert calls["kwargs"] == {"group_by": "ticker"}


def test_load_prices_reads_local_file(monkeypatch, prices):
    monkeypatch.setattr(data.pd, "read_parquet", lambda path: prices)

    def fail_download(tickers):
        raise AssertionError("should not download")

    monkeypatch.setattr(data, "download_prices", fail_download)

    assert data.load_prices(["AAA"]) is prices


def test_load_prices_downloads_and_saves_when_missing(monkeypatch, prices):
    def missing(path):
        raise FileNotFoundError

    saved = []
    monkeypatch.setattr(data.pd, "read_parquet", missing)
    monkeypatch.setattr(data, "download_prices", lambda tickers: prices)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: saved.append(path))

    result = data.load_prices(["AAA"])

    assert result is prices
    assert saved == ["data/raw/prices.parquet"]


def test_load_prices_refresh_skips_local_file(monkeypatch, prices):
    def fail_read(path):
        raise AssertionError("should not read when refreshing")

    saved = []
    monkeypatch.setattr(data.pd, "read_parquet", fail_read)
    monkeypatch.setattr(data, "download_prices", lambda tickers: prices)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: saved.append(path))

    result = data.load_prices(["AAA"], refresh=True)

    assert result is prices
    assert saved == ["data/raw/prices.parquet"]


def test_to_monthly_returns_values(prices):
    result = data.to_monthly_returns(prices)

    # Month-end prices: 110, 121, 133.1 -> first return dropped (NaN)
    assert list(result.index) == list(pd.to_datetime(["2024-02-29", "2024-03-31"]))
    assert result["AAA"].tolist() == pytest.approx([0.10, 0.10])


def test_to_monthly_returns_multiple_columns(prices):
    prices["BBB"] = [50.0, 50.0, 50.0, 55.0, 55.0]

    result = data.to_monthly_returns(prices)

    assert list(result.columns) == ["AAA", "BBB"]
    assert result["BBB"].tolist() == pytest.approx([0.10, 0.0])


def test_to_monthly_returns_single_month_is_empty():
    idx = pd.to_datetime(["2024-01-10", "2024-01-31"])
    df = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)

    assert data.to_monthly_returns(df).empty
