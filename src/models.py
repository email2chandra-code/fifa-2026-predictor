# Dixon-Coles-style Poisson net with iteration-specific forward passes.
import torch
import torch.nn as nn

from .config import DC_RHO_INIT


class DixonColesPyTorchNet(nn.Module):
    def __init__(self, num_teams):
        super().__init__()
        self.attack_embeddings = nn.Embedding(num_teams, 1)
        self.defense_embeddings = nn.Embedding(num_teams, 1)

        nn.init.uniform_(self.attack_embeddings.weight, -0.2, 0.2)
        nn.init.uniform_(self.defense_embeddings.weight, -0.2, 0.2)

        # Learned from qualifiers; zeroed at neutral World Cup venues
        self.home_advantage = nn.Parameter(torch.tensor([0.10]))
        self.rho = nn.Parameter(torch.tensor([DC_RHO_INIT]))

    def forward(self, home_ids, away_ids, iteration=4, neutral=False):
        h_att = self.attack_embeddings(home_ids)
        h_def = self.defense_embeddings(home_ids)
        a_att = self.attack_embeddings(away_ids)
        a_def = self.defense_embeddings(away_ids)

        ha = torch.zeros_like(self.home_advantage) if neutral else self.home_advantage

        if iteration == 1:
            lambda_home = torch.clamp(h_att * a_def + ha, min=1e-6)
            lambda_away = torch.clamp(a_att * h_def, min=1e-6)
        else:
            lambda_home = torch.exp(h_att - a_def + ha)
            lambda_away = torch.exp(a_att - h_def)

        return torch.cat((lambda_home, lambda_away), dim=1)
