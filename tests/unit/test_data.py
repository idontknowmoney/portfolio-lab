import pandas as pd
import pytest

from portfolio_lab import data


@pytest.fixture(autouse=True)
def prices_path(monkeypatch, tmp_path):
    path = tmp_path / "raw" / "prices.parquet"
    monkeypatch.setattr(data, "PRICES_PATH", path)
    return path


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
    assert calls["kwargs"] == {"period": "max", "group_by": "ticker", "auto_adjust": True}


def test_load_prices_reads_local_file(monkeypatch, prices):
    monkeypatch.setattr(data.pd, "read_parquet", lambda path: prices)

    def fail_download(tickers):
        raise AssertionError("should not download")

    monkeypatch.setattr(data, "download_prices", fail_download)

    assert data.load_prices(["AAA"]) is prices


def test_load_prices_downloads_and_saves_when_missing(monkeypatch, prices, prices_path):
    def missing(path):
        raise FileNotFoundError

    saved = []
    monkeypatch.setattr(data.pd, "read_parquet", missing)
    monkeypatch.setattr(data, "download_prices", lambda tickers: prices)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: saved.append(path))

    result = data.load_prices(["AAA"])

    assert result is prices
    assert saved == [prices_path]
    assert prices_path.parent.is_dir()


def test_load_prices_refresh_skips_local_file(monkeypatch, prices, prices_path):
    def fail_read(path):
        raise AssertionError("should not read when refreshing")

    saved = []
    monkeypatch.setattr(data.pd, "read_parquet", fail_read)
    monkeypatch.setattr(data, "download_prices", lambda tickers: prices)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: saved.append(path))

    result = data.load_prices(["AAA"], refresh=True)

    assert result is prices
    assert saved == [prices_path]


def test_load_prices_redownloads_when_ticker_missing_from_cache(monkeypatch, prices, prices_path):
    both = prices.assign(BBB=1.0)
    saved = []
    monkeypatch.setattr(data.pd, "read_parquet", lambda path: prices)
    monkeypatch.setattr(data, "download_prices", lambda tickers: both)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path: saved.append(path))

    result = data.load_prices(["AAA", "BBB"])

    assert result is both
    assert saved == [prices_path]


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


def _month_ends(n, start="2020-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_to_eur_weaker_dollar_lowers_return():
    idx = _month_ends(2)
    usd = pd.Series([0.0, 0.0], index=idx)
    fx = pd.Series([0.10, 0.0], index=idx)
    assert data.to_eur(usd, fx).tolist() == pytest.approx([1 / 1.1 - 1, 0.0])


def test_to_eur_missing_fx_is_nan():
    idx = _month_ends(2)
    out = data.to_eur(pd.Series([0.01, 0.02], index=idx), pd.Series([0.0], index=idx[:1]))
    assert out.iloc[0] == pytest.approx(0.01)
    assert pd.isna(out.iloc[1])


def test_blend_applies_weights_and_drops_incomplete_months():
    idx = _month_ends(3)
    df = pd.DataFrame({"A": [0.10, 0.20, None], "B": [0.0, 0.10, 0.50]}, index=idx)
    out = data.blend(df, [0.7, 0.3])
    assert list(out.index) == list(idx[:2])
    assert out.tolist() == pytest.approx([0.07, 0.17])


def test_splice_has_no_overlap_and_keeps_primary():
    idx = _month_ends(6)
    primary = pd.Series([None, None, None, 0.1, 0.2, 0.3], index=idx)
    proxy = pd.Series([0.01, 0.02, 0.03, 0.9, 0.9, 0.9], index=idx)
    out = data.splice(primary, proxy)
    assert out.index.is_unique and list(out.index) == list(idx)
    assert out.tolist() == pytest.approx([0.01, 0.02, 0.03, 0.1, 0.2, 0.3])


def test_splice_empty_proxy_returns_primary():
    idx = _month_ends(3)
    primary = pd.Series([0.1, 0.2, 0.3], index=idx)
    out = data.splice(primary, pd.Series(dtype=float))
    assert out.tolist() == primary.tolist()


def test_series_monthly_returns_keeps_each_series_own_months():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    closes = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [None, 50.0, 55.0]}, index=idx)
    out = data.series_monthly_returns(closes)
    assert out["A"].dropna().tolist() == pytest.approx([0.10, 0.10])
    assert out["B"].dropna().tolist() == pytest.approx([0.10])
