from typing import Literal

import numpy as np


def simulate(
    initial_values: np.ndarray,
    contribution_weights: np.ndarray,
    monthly_contribution: np.ndarray,
    monthly_returns: np.ndarray,
    n_months: int,
    n_paths: int,
    block_size: int = 12,
    rebalance: Literal["monthly", "none"] = "none",
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate the evolution of a portfolio over time.

    Parameters:
    initial_values (np.ndarray): Initial values of the portfolio.
    contribution_weights (np.ndarray): Weights for contributions to the portfolio.
    monthly_contribution (np.ndarray): Monthly contributions to the portfolio.
    monthly_returns (np.ndarray): Monthly returns for each asset in the portfolio.
    n_months (int): Number of months to simulate.
    n_paths (int): Number of simulation paths.
    block_size (int): Size of the blocks for block bootstrap sampling.
    rebalance (Literal["monthly", "none"]): Rebalancing strategy for the portfolio.
    seed (int | None): Random seed for reproducibility.

    Returns:
    np.ndarray: Simulated portfolio values over time.

    Raises:
    NotImplementedError: If rebalance is not "none".
    """
    if rebalance != "none":
        raise NotImplementedError("Rebalancing not implemented yet.")

    n_hist_months = monthly_returns.shape[0]

    if not 1 <= block_size <= n_hist_months:
        raise ValueError("block_size must be between 1 and the number of historical months.")

    rng = np.random.default_rng(seed)

    n_assets = initial_values.shape[0]
    portfolio_values = np.empty((n_paths, n_months + 1, n_assets))
    V = np.tile(initial_values, (n_paths, 1)).astype(float)
    portfolio_values[:, 0, :] = V

    # Block bootstrap sampling
    n_blocks = -(-n_months // block_size)

    starts = rng.integers(0, n_hist_months - block_size + 1, size=(n_paths, n_blocks))
    offsets = np.arange(block_size)
    month_idx = (starts[:, :, None] + offsets).reshape(n_paths, -1)[:, :n_months]

    contribution = monthly_contribution * contribution_weights

    for t in range(n_months):
        returns_t = monthly_returns[month_idx[:, t]]
        V += contribution  # contribution at the start of the month
        V *= 1 + returns_t
        portfolio_values[:, t + 1, :] = V

    return portfolio_values
