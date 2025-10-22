from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )

class QNetwork(nn.Module):
    """
    Standard DQN head: MLP -> linear head to actions.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = _mlp(obs_dim, hidden)
        self.head = nn.Linear(hidden, act_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.backbone(x)
        q = self.head(z)
        return q


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN: separate value and advantage streams.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = _mlp(obs_dim, hidden)
        self.val = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, act_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.backbone(x)
        v = self.val(z)                 # [B,1]
        a = self.adv(z)                 # [B,A]
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
