import numpy as np
import pytest

from portfolio_lab import analysis
from portfolio_lab.config import Config


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
