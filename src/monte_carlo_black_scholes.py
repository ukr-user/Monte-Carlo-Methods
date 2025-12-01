"""
Black–Scholes European option pricing:
- closed-form formula
- Monte Carlo estimator under geometric Brownian motion

Run this file directly to see a comparison between the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt, exp, erf
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function using math.erf,
    so we don't need scipy as a dependency.
    """
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass
class EuropeanOption:
    """Container for European option parameters."""
    spot: float          # S0: current underlying price
    strike: float        # K: strike price
    maturity: float      # T: time to maturity in years
    rate: float          # r: risk-free rate (continuously compounded)
    volatility: float    # sigma: annual volatility
    option_type: OptionType = "call"


def black_scholes_price(opt: EuropeanOption) -> float:
    """
    Closed-form Black–Scholes price for a European call or put.

    Parameters
    ----------
    opt : EuropeanOption
        Option parameters.

    Returns
    -------
    float
        Theoretical Black–Scholes price.
    """
    S, K, T, r, sigma = (
        opt.spot,
        opt.strike,
        opt.maturity,
        opt.rate,
        opt.volatility,
    )

    if T <= 0 or sigma <= 0:
        raise ValueError("Maturity T and volatility sigma must be positive.")

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if opt.option_type == "call":
        price = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    elif opt.option_type == "put":
        price = K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return price


def monte_carlo_price(
    opt: EuropeanOption,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> float:
    """
    Monte Carlo estimator for a European call or put under Black–Scholes.

    Simulates terminal prices using geometric Brownian motion.

    Parameters
    ----------
    opt : EuropeanOption
        Option parameters.
    n_paths : int
        Number of simulation paths.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated option price from simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    S0, K, T, r, sigma = (
        opt.spot,
        opt.strike,
        opt.maturity,
        opt.rate,
        opt.volatility,
    )

    # Simulate terminal stock price S_T under risk-neutral measure
    z = np.random.normal(size=n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * sqrt(T) * z)

    if opt.option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:  # put
        payoffs = np.maximum(K - ST, 0.0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    return float(discounted_payoffs.mean())


def _demo() -> None:
    """
    Compare closed-form vs Monte Carlo prices
    for a typical at-the-money call.
    """
    opt = EuropeanOption(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        option_type="call",
    )

    closed = black_scholes_price(opt)
    mc = monte_carlo_price(opt, n_paths=100_000, seed=42)

    print("Option parameters:", opt)
    print(f"Closed-form price : {closed:.4f}")
    print(f"Monte Carlo price : {mc:.4f}")
    print(f"Difference        : {abs(closed - mc):.4f}")


if __name__ == "__main__":
    _demo()
