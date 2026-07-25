import numpy as np
import torch
import torch.optim as optim

from .config import ELO_ANCHOR_WEIGHT
from .dixon_coles import tau_factor_batch
from .models import DixonColesPyTorchNet


def _poisson_nll(lambdas, goals):
    eps = 1e-8
    return lambdas - goals * torch.log(lambdas + eps)


def _elo_anchor_loss(model, attack_prior, defense_prior):
    att = model.attack_embeddings.weight.view(-1)
    deff = model.defense_embeddings.weight.view(-1)
    att_prior = torch.tensor(attack_prior, dtype=torch.float32)
    def_prior = torch.tensor(defense_prior, dtype=torch.float32)
    return ((att - att_prior) ** 2 + (deff - def_prior) ** 2).mean()


def execute_model_calibration(
    df_train,
    num_teams,
    iteration=4,
    epochs=300,
    lr=0.005,
    attack_prior=None,
    defense_prior=None,
):
    """
    Calibrate Dixon-Coles Poisson net.

    Iteration 4+: Elo anchor regularization pulls embeddings toward Elo priors.
    Iteration 5: Dixon-Coles tau term in the match-level NLL (draw polishing).
    Iteration 6+: per-side opponent difficulty weights (home goals weighted by
    the away team's D_opp and vice versa); tau keeps the row-level weight.
    Iteration 7 adds competition confederation strength (C_conf) in features.
    Iteration 8 blends Elo priors with squad market-value priors.
    """
    model = DixonColesPyTorchNet(num_teams)
    if iteration >= 4 and attack_prior is not None and defense_prior is not None:
        with torch.no_grad():
            model.attack_embeddings.weight.copy_(
                torch.tensor(attack_prior, dtype=torch.float32).view(-1, 1)
            )
            model.defense_embeddings.weight.copy_(
                torch.tensor(defense_prior, dtype=torch.float32).view(-1, 1)
            )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_idx = torch.tensor(
        df_train[["home_id", "away_id"]].values.astype(np.int64), dtype=torch.long
    )
    y_goals = torch.tensor(
        df_train[["home_score", "away_score"]].values.astype(np.float32),
        dtype=torch.float32,
    )
    w_tensor = torch.tensor(
        df_train["final_weight"].values.astype(np.float32), dtype=torch.float32
    )

    use_elo_anchor = iteration >= 4 and attack_prior is not None and defense_prior is not None
    use_dc_tau = iteration >= 5
    use_side_weights = iteration >= 6 and "weight_home_side" in df_train.columns

    w_home_side = w_away_side = None
    if use_side_weights:
        w_home_side = torch.tensor(
            df_train["weight_home_side"].values.astype(np.float32), dtype=torch.float32
        )
        w_away_side = torch.tensor(
            df_train["weight_away_side"].values.astype(np.float32), dtype=torch.float32
        )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(x_idx[:, 0], x_idx[:, 1], iteration=iteration, neutral=False)
        lh, la = preds[:, 0], preds[:, 1]
        yh, ya = y_goals[:, 0], y_goals[:, 1]

        if use_side_weights:
            weighted_nll = w_home_side * _poisson_nll(lh, yh) + w_away_side * _poisson_nll(
                la, ya
            )
            if use_dc_tau:
                tau = tau_factor_batch(yh, ya, lh, la, model.rho)
                weighted_nll = weighted_nll - w_tensor * torch.log(tau)
            weighted_loss = weighted_nll.mean()
        else:
            nll = _poisson_nll(lh, yh) + _poisson_nll(la, ya)

            if use_dc_tau:
                tau = tau_factor_batch(yh, ya, lh, la, model.rho)
                nll = nll - torch.log(tau)

            weighted_loss = (nll * w_tensor).mean()

        if use_elo_anchor:
            weighted_loss = weighted_loss + ELO_ANCHOR_WEIGHT * _elo_anchor_loss(
                model, attack_prior, defense_prior
            )

        weighted_loss.backward()
        optimizer.step()

    print(f" -> Iteration {iteration} Optimized. Final Training Loss: {weighted_loss.item():.4f}")
    if use_dc_tau:
        print(f"    Dixon-Coles rho: {model.rho.item():.4f}")
    return model
