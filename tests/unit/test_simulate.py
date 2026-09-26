import numpy as np
import pytest

from portfolio_lab.simulate import simulate


def run(**overrides):
    kwargs = dict(
        initial_values=np.array([1000.0, 500.0]),
        contribution_weights=np.array([0.6, 0.4]),
        monthly_contribution=100.0,
        monthly_returns=np.zeros((24, 2)),
        n_months=12,
        n_paths=5,
        block_size=3,
        seed=0,
    )
    kwargs.update(overrides)
    return simulate(**kwargs)


def test_zero_returns_only_accumulate_contributions():
    out = run()

    assert out.shape == (5, 13, 2)
    months = np.arange(13)[:, None]
    expected = np.array([1000.0, 500.0]) + months * np.array([60.0, 40.0])
    for path in out:
        np.testing.assert_allclose(path, expected)


def test_sampled_months_are_consecutive_within_blocks():
    n_hist, block_size, n_months = 20, 4, 12
    # Distinct per-month return so each simulated month identifies its source row.
    returns = (np.arange(n_hist) + 1)[:, None] / 100.0
    out = simulate(
        initial_values=np.array([1.0]),
        contribution_weights=np.array([1.0]),
        monthly_contribution=0.0,
        monthly_returns=returns,
        n_months=n_months,
        n_paths=50,
        block_size=block_size,
        seed=1,
    )

    growth = out[:, 1:, 0] / out[:, :-1, 0] - 1
    idx = np.rint(growth * 100).astype(int) - 1

    assert idx.min() >= 0 and idx.max() < n_hist
    blocks = idx.reshape(50, n_months // block_size, block_size)
    np.testing.assert_array_equal(np.diff(blocks, axis=2), 1)


def test_truncates_final_partial_block():
    out = run(n_months=10, block_size=4)

    assert out.shape == (5, 11, 2)


def test_bonus_contribution_applied_before_return():
    # Contribution at start of month earns that month's return.
    out = simulate(
        initial_values=np.array([0.0]),
        contribution_weights=np.array([1.0]),
        monthly_contribution=100.0,
        monthly_returns=np.full((6, 1), 0.10),
        n_months=1,
        n_paths=1,
        block_size=1,
        seed=0,
    )

    assert out[0, 1, 0] == pytest.approx(110.0)


def test_same_seed_is_reproducible():
    returns = np.random.default_rng(0).normal(0.01, 0.05, size=(24, 2))

    a = run(monthly_returns=returns, seed=42)
    b = run(monthly_returns=returns, seed=42)

    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("block_size", [0, 25])
def test_block_size_out_of_bounds_raises(block_size):
    with pytest.raises(ValueError):
        run(block_size=block_size)


def test_rebalance_not_implemented():
    with pytest.raises(NotImplementedError):
        run(rebalance="monthly")
