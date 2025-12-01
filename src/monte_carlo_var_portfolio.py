"""
Monte Carlo Value-at-Risk (VaR) for a simple multi-asset portfolio.

We assume asset returns are jointly normally distributed with
a given mean vector and covariance matrix, and we simulate many
scenarios to estimate the distribution of portfolio P&L.

From this distribution we compute:
- Value-at-Risk (VaR)
- Expected Shortfall (ES, also called CVaR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PortfolioConfig:
    """
    Configuration for the portfolio and simulation.

    Attributes
    ----------
    initial_value : float
        Current value of the portfolio in currency units.
    weights : Sequence[float]
        Portfolio weights for each asset (must sum to 1).
    mu : Sequence[float]
        Expected DAILY returns for each asset.
    vol : Sequence[float]
        DAILY volatilities (standard deviations) for each asset.
    corr : np.ndarray
        Correlation matrix between asset returns.
    horizon_days : int
        Risk horizon in trading days (e.g. 1, 10).
    var_level : float
        Confidence level for VaR (e.g. 0.99 for 99% VaR).
    n_paths : int
        Number of Monte Carlo scenarios.
    """

    initial_value: float
    weights: Sequence[float]
    mu: Sequence[float]
    vol: Sequence[float]
    corr: np.ndarray
    horizon_days: int = 1
    var_level: float = 0.99
    n_paths: int = 100_000


def _build_covariance(vol: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """
    Build covariance matrix from volatilities and correlation matrix.

    cov_ij = vol_i * vol_j * corr_ij
    """
    return np.outer(vol, vol) * corr


def monte_carlo_var(cfg: PortfolioConfig, seed: int | None = None) -> dict[str, float]:
    """
    Estimate portfolio VaR and Expected Shortfall via Monte Carlo.

    Parameters
    ----------
    cfg : PortfolioConfig
        Portfolio and simulation settings.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Dictionary with VaR, ES, and basic distribution stats.
    """
    if seed is not None:
        np.random.seed(seed)

    w = np.asarray(cfg.weights, dtype=float)
    mu = np.asarray(cfg.mu, dtype=float)
    vol = np.asarray(cfg.vol, dtype=float)
    corr = np.asarray(cfg.corr, dtype=float)

    n_assets = w.shape[0]

    if mu.shape[0] != n_assets or vol.shape[0] != n_assets:
        raise ValueError("weights, mu, and vol must have the same length.")

    if corr.shape != (n_assets, n_assets):
        raise ValueError("corr must be a square matrix with size = number of assets.")

    # Scale mean and covariance to the chosen horizon
    mu_h = mu * cfg.horizon_days
    cov_daily = _build_covariance(vol, corr)
    cov_h = cov_daily * cfg.horizon_days

    # Simulate horizon returns: shape (n_paths, n_assets)
    returns = np.random.multivariate_normal(
        mean=mu_h,
        cov=cov_h,
        size=cfg.n_paths,
    )

    # Portfolio return per scenario: (n_paths,)
    port_returns = returns @ w

    # Portfolio value and P&L
    v0 = cfg.initial_value
    values = v0 * (1.0 + port_returns)
    pnl = values - v0  # profit > 0, loss < 0

    # Loss is negative P&L (so losses are positive numbers)
    losses = -pnl

    # VaR at level alpha: threshold such that P(Loss <= VaR) = alpha
    alpha = cfg.var_level
    var = float(np.quantile(losses, alpha))

    # Expected Shortfall (ES): mean loss beyond VaR
    tail_losses = losses[losses >= var]
    es = float(tail_losses.mean()) if tail_losses.size > 0 else float("nan")

    # Some extra stats (useful for debugging / understanding)
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())

    return {
        "VaR": var,
        "ES": es,
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
    }


def _demo() -> None:
    """
    Run a small example: a 3-asset equity portfolio over 1 day.

    This is just a toy example with made-up parameters,
    but it demonstrates the workflow.
    """
    initial_value = 1_000_000.0  # 1 million

    weights = [0.5, 0.3, 0.2]  # 50% asset 1, 30% asset 2, 20% asset 3

    # Daily expected returns (e.g. 0.05 -> 5% annual roughly)
    mu = [0.0005, 0.0003, 0.0001]

    # Daily volatilities
    vol = [0.02, 0.015, 0.01]

    # Correlation matrix between the three assets
    corr = np.array(
        [
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ]
    )

    cfg = PortfolioConfig(
        initial_value=initial_value,
        weights=weights,
        mu=mu,
        vol=vol,
        corr=corr,
        horizon_days=1,
        var_level=0.99,
        n_paths=100_000,
    )

    results = monte_carlo_var(cfg, seed=42)

    print("=== Monte Carlo VaR example ===")
    print("Config:")
    print(f"  Initial portfolio value : {cfg.initial_value:,.2f}")
    print(f"  Horizon (days)          : {cfg.horizon_days}")
    print(f"  VaR confidence level    : {cfg.var_level:.2%}")
    print()
    print("Results (currency units):")
    print(f"  Mean P&L   : {results['mean_pnl']:,.2f}")
    print(f"  Std P&L    : {results['std_pnl']:,.2f}")
    print(f"  VaR        : {results['VaR']:,.2f}")
    print(f
