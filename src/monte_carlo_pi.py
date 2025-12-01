"""
Estimate π using a simple Monte Carlo simulation.

This script generates random points inside a square
and counts the fraction that fall inside the unit circle.
"""

import numpy as np


def estimate_pi(n_samples: int = 100_000, seed: int | None = None) -> float:
    """
    Estimate the value of π using Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of random points to simulate.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated value of π.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)

    inside = (x**2 + y**2) <= 1
    return 4 * inside.mean()


def _demo() -> None:
    """Run a quick demonstration."""
    est = estimate_pi(200_000, seed=42)
    print(f"Estimated π = {est:.6f}")


if __name__ == "__main__":
    _demo()

