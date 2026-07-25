"""
Dixon-Coles low-score correlation adjustment (tau / rho).

Iteration 5 applies tau in training loss and match-level inference.
"""

import math

import numpy as np


def tau_factor(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Scalar Dixon-Coles tau for a single scoreline."""
    h = int(home_goals)
    a = int(away_goals)
    if h == 0 and a == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if h == 0 and a == 1:
        return 1.0 + lambda_home * rho
    if h == 1 and a == 0:
        return 1.0 + lambda_away * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def tau_factor_batch(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Vectorized tau for batched training tensors."""
    import torch

    tau = torch.ones_like(lambda_home)
    h = home_goals.view(-1)
    a = away_goals.view(-1)
    lh = lambda_home.view(-1)
    la = lambda_away.view(-1)

    mask00 = (h == 0) & (a == 0)
    mask01 = (h == 0) & (a == 1)
    mask10 = (h == 1) & (a == 0)
    mask11 = (h == 1) & (a == 1)

    tau[mask00] = 1.0 - lh[mask00] * la[mask00] * rho
    tau[mask01] = 1.0 + lh[mask01] * rho
    tau[mask10] = 1.0 + la[mask10] * rho
    tau[mask11] = 1.0 - rho

    return torch.clamp(tau, min=1e-6)


def _poisson_pmf(k, lam):
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def dixon_coles_outcome_probs(lambda_home, lambda_away, rho=-0.03, max_goals=10):
    """Return (p_home, p_draw, p_away) with Dixon-Coles scoreline adjustment."""
    probs = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tau = tau_factor(h, a, lambda_home, lambda_away, rho)
            probs[h, a] = tau * _poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)

    total = probs.sum()
    if total > 0:
        probs /= total

    p_home = float(probs[np.triu_indices(max_goals + 1, k=1)].sum())
    p_draw = float(np.trace(probs))
    p_away = float(probs[np.tril_indices(max_goals + 1, k=-1)].sum())
    return p_home, p_draw, p_away
