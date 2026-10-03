import numpy as np
import pandas as pd
import pytest

from portfolio_lab import analysis
from portfolio_lab.config import Asset, Config


@pytest.fixture
def cfg():
    def asset(name, weight, initial, cost):
        return {
            "name": name,
            "isin": name,
            "ticker": name,
            "contribution_weight": weight,
            "initial_value": initial,
            "cost_basis": cost,
        }

    return Config.model_validate(
        {
            "portfolio": {
                "name": "t",
                "currency": "EUR",
                "assets": [asset("A", 1.0, 110.0, 100.0), asset("B", 0.0, 40.0, 50.0)],
            },
            "simulation": {
                "monthly_contribution": 10,
                "horizon_years": 1,
                "n_paths": 4,
                "seed": 0,
                "block_size": 3,
                "rebalance": "none",
            },
        }
    )


def test_contributions_by_asset_from_initial_value(cfg):
    out = analysis.contributions_by_asset(cfg)

    assert out.shape == (13, 2)
    np.testing.assert_allclose(out[:, 0], 110 + 10 * np.arange(13))
    np.testing.assert_allclose(out[:, 1], 40)  # zero-weight asset stays flat


def test_contributions_by_asset_from_cost_basis(cfg):
    out = analysis.contributions_by_asset(cfg, "cost_basis")

    np.testing.assert_allclose(out[0], [100, 50])
    np.testing.assert_allclose(out[-1], [220, 50])


def test_contributed_path_sums_assets(cfg):
    np.testing.assert_allclose(analysis.contributed_path(cfg)[[0, 12]], [150, 270])
    np.testing.assert_allclose(analysis.contributed_path(cfg, "cost_basis")[[0, 12]], [150, 270])


def test_final_value_summary_with_cost():
    total = np.array([[0.0, 90.0], [0.0, 110.0], [0.0, 130.0], [0.0, 150.0]])
    contributed = np.array([100.0, 100.0])
    cost = np.array([120.0, 120.0])

    summary = analysis.final_value_summary(total, contributed, cost)

    assert summary["P50"] == pytest.approx(120.0)
    assert summary["P(loss)"] == pytest.approx(0.25)
    assert summary["Cost basis + contrib."] == 120.0
    assert summary["P(loss vs cost)"] == pytest.approx(0.5)
    assert "Cost basis + contrib." not in analysis.final_value_summary(total, contributed)


def test_gain_summary():
    total = np.array([[0.0, 90.0], [0.0, 110.0], [0.0, 130.0], [0.0, 150.0]])
    cost = np.array([100.0, 100.0])

    out = analysis.gain_summary(total, cost)

    assert out.loc["P50", "Gain"] == pytest.approx(20.0)
    assert out.loc["P50", "Return"] == pytest.approx(0.2)


def test_gain_by_asset():
    # 3 paths, 2 months, 2 assets; second asset has nothing invested
    paths = np.zeros((3, 2, 2))
    paths[:, -1, 0] = [80.0, 100.0, 140.0]
    paths[:, -1, 1] = 5.0
    cost = np.array([[100.0, 0.0], [100.0, 0.0]])

    out = analysis.gain_by_asset(paths, cost, ["A", "B"])

    assert out.loc["A", "Invested"] == 100.0
    assert out.loc["A", "P50 gain"] == pytest.approx(0.0)
    assert out.loc["A", "P50 return"] == pytest.approx(0.0)
    assert out.loc["B", "P50 gain"] == pytest.approx(5.0)
    assert np.isnan(out.loc["B", "P50 return"])


def test_gain_by_asset_round_trip_with_run_simulation(cfg):
    import pandas as pd

    returns = pd.DataFrame(np.zeros((24, 2)), columns=["A", "B"])
    paths = analysis.run_simulation(cfg, returns)

    assert paths.shape == (4, 13, 2)
    out = analysis.gain_by_asset(
        paths, analysis.contributions_by_asset(cfg, "cost_basis"), ["A", "B"]
    )
    # zero returns: A ends at 110 + 120 = 230 vs invested 220; B stays 40 vs 50
    assert out.loc["A", "P50 gain"] == pytest.approx(10.0)
    assert out.loc["B", "P50 gain"] == pytest.approx(-10.0)


def test_sequence_risk_same_returns_different_final_values():
    rets = pd.Series([0.01] * 20 + [-0.3] * 4 + [0.01] * 20)
    sc = analysis.sequence_risk_scenarios(rets, 1000.0, 100.0, window=4)
    final = {k: v[-1] for k, v in sc.items()}
    assert final["Worst block last"] < final["Worst block first"]


def test_sequence_risk_no_contributions_is_order_independent():
    rets = pd.Series([0.02, -0.1, 0.03, 0.01, -0.05, 0.04, 0.0, 0.02])
    sc = analysis.sequence_risk_scenarios(rets, 1000.0, 0.0, window=2)
    finals = [v[-1] for v in sc.values()]
    assert finals == pytest.approx([finals[0]] * len(finals))


def test_worst_window_start():
    assert analysis.worst_window_start(np.array([0.1, -0.2, -0.2, 0.1, 0.1]), 2) == 1


def test_sequence_risk_summary_final_values():
    sc = {"a": np.array([100.0, 210.0, 330.0]), "b": np.array([100.0, 150.0, 300.0])}
    out = analysis.sequence_risk_summary(sc)
    assert out.to_dict() == {"a": 330.0, "b": 300.0}


def test_haircut_returns_subtracts_monthly_share():
    r = pd.DataFrame({"A": [0.01, 0.02]})
    out = analysis.haircut_returns(r, 0.12)
    np.testing.assert_allclose(out["A"], [0.0, 0.01])


def _prices(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Raw-prices frame with (ticker, "Close") columns, as yfinance returns it."""
    return pd.concat({t: pd.DataFrame({"Close": s}) for t, s in series.items()}, axis=1)


@pytest.fixture
def proxy_prices():
    """OLD launches 2020-07; its proxy (USD) exists all along; EURUSD flat."""
    idx = pd.date_range("2020-01-01", "2021-06-30", freq="D")
    n = np.arange(len(idx))
    old = pd.Series(100.0 * 1.001**n, index=idx)
    young = pd.Series(100.0 * 1.001**n, index=idx).where(idx >= "2020-07-01")
    proxy = pd.Series(50.0 * 1.001**n, index=idx)
    fx = pd.Series(1.1, index=idx)
    return _prices({"OLD": old, "YOUNG": young, "PX": proxy, analysis.FX_TICKER: fx})


@pytest.fixture
def proxy_assets():
    def asset(ticker, proxy):
        return Asset(
            name=ticker, isin=ticker, ticker=ticker, contribution_weight=0.5, initial_value=0,
            proxy=proxy,
        )  # fmt: skip

    px = [{"ticker": "PX", "currency": "USD", "weight": 1.0}]
    return [asset("OLD", []), asset("YOUNG", px)]


def test_required_tickers_adds_proxies_and_fx(proxy_assets):
    assert analysis.required_tickers(proxy_assets) == ["OLD", "YOUNG", "PX", analysis.FX_TICKER]


def test_build_returns_window_starts_at_proxy_and_flags_proxy_months(proxy_prices, proxy_assets):
    returns, is_proxy = analysis.build_returns(proxy_prices, proxy_assets, use_proxies=True)
    no_proxy, _ = analysis.build_returns(proxy_prices, proxy_assets, use_proxies=False)

    assert returns.index[0] < no_proxy.index[0]
    assert returns.index[0] == pd.Timestamp("2020-02-29")
    assert is_proxy["OLD"].eq(False).all()
    # YOUNG launched mid-2020, so June is the last month filled by the proxy
    assert list(is_proxy.index[is_proxy["YOUNG"]]) == list(
        pd.date_range("2020-02-29", "2020-07-31", freq="ME")
    )
    assert not returns.isna().any().any()


def test_build_returns_without_proxies_flags_nothing(proxy_prices, proxy_assets):
    returns, is_proxy = analysis.build_returns(proxy_prices, proxy_assets, use_proxies=False)
    assert not is_proxy.any().any()
    assert list(returns.columns) == ["OLD", "YOUNG"]


def test_history_window_counts_proxy_months(proxy_prices, proxy_assets):
    etf = analysis.etf_returns(proxy_prices, proxy_assets)
    spliced, is_proxy = analysis.spliced_returns(
        etf, analysis.proxy_returns(proxy_prices, proxy_assets)
    )
    out = analysis.history_window(spliced, is_proxy, block_size=3)

    assert out.loc["OLD", "Proxy months"] == 0
    assert out.loc["YOUNG", "Proxy months"] > 0
    common = out.loc["Common window"]
    assert common["ETF months"] + common["Proxy months"] == len(spliced.dropna())
    assert common["Distinct blocks"] == len(spliced.dropna()) - 2


def test_proxy_quality_perfect_tracker():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    r = pd.Series(np.linspace(-0.02, 0.03, 12), index=idx)
    out = analysis.proxy_quality(pd.DataFrame({"A": r}), pd.DataFrame({"A": r}))
    assert out.loc["A", "Overlap months"] == 12
    assert out.loc["A", "Correlation"] == pytest.approx(1.0)
    assert out.loc["A", "Tracking diff"] == pytest.approx(0.0)


def test_sensitivity_to_haircut_lowers_median(cfg):
    r = pd.DataFrame({"A": np.full(12, 0.01), "B": np.full(12, 0.01)})
    out = analysis.sensitivity_to_haircut(cfg, r, haircuts=(0.0, 0.1))
    assert list(out.index) == ["Baseline", "-10%/yr"]
    assert out.loc["-10%/yr", "P50"] < out.loc["Baseline", "P50"]
