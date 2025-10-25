# network_sonic.py
import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------- Utils ----------
def orthogonal_init(module: nn.Module, gain: float = 1.0):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module

# ---------- CNN Encoder ----------
class SonicEncoder(nn.Module):
    """
    Nature-CNN style encoder (works with 4 stacked frames).
    Supports 84x84 or 96x96 inputs (or any square size divisible by 2^3).
    """
    def __init__(self, in_channels: int = 4, c3: int = 64):
        super().__init__()
        # conv3 filters can be set (some repos use 48)
        self.enc = nn.Sequential(
            orthogonal_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Conv2d(64, c3, kernel_size=3, stride=1), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
        )
        self._n_flatten = None  # set after seeing a dummy input

    def output_dim(self, input_shape: Tuple[int, int, int]) -> int:
        """
        Compute flattened size dynamically. input_shape = (C,H,W)
        """
        if self._n_flatten is not None:
            return self._n_flatten
        with torch.no_grad():
            c, h, w = input_shape
            dummy = torch.zeros(1, c, h, w)
            y = self.enc(dummy)
            self._n_flatten = y.view(1, -1).size(1)
        return self._n_flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], pixel range [0,255] or [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.enc(x)


# ---------- Actor-Critic Head ----------
class ActorCriticCNN(nn.Module):
    """
    Actor-Critic network for PPO with discrete actions.
    - Encoder: CNN
    - Policy head: logits over actions
    - Value head: scalar V(s)
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],   # (C,H,W) after frame stacking and preprocessing
        num_actions: int,
        hidden_size: int = 512,
        conv3_channels: int = 64,
    ):
        super().__init__()
        c, h, w = obs_shape
        self.encoder = SonicEncoder(in_channels=c, c3=conv3_channels)
        n_flat = self.encoder.output_dim(obs_shape)

        self.pi = nn.Sequential(
            orthogonal_init(nn.Linear(n_flat, hidden_size), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(hidden_size, num_actions), gain=0.01),  # small init for policy
        )
        self.v = nn.Sequential(
            orthogonal_init(nn.Linear(n_flat, hidden_size), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    # ----- Core forward -----
    def forward(self, obs: torch.Tensor):
        """
        Returns (logits, value)
        obs: [B,C,H,W] (uint8 or float)
        """
        z = self.encoder(obs)
        z = z.flatten(1)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value

    # ----- Helpers for PPO -----
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample (or take greedy) action given obs.
        Returns (action, logprob, value).
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Used in PPO loss: returns (logprob, entropy, value) for given actions.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value

    # ----- Save/Load -----
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        self.load_state_dict(torch.load(path, map_location=map_location))


# ---------- Quick self-test ----------
if __name__ == "__main__":
    # Example for 4 stacked grayscale frames at 84x84
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = (4, 84, 84)   # or (4, 96, 96)
    num_actions = 10

    net = ActorCriticCNN(obs_shape, num_actions).to(device)
    x = torch.zeros(2, *obs_shape, dtype=torch.uint8).to(device)  # batch of 2
    with torch.no_grad():
        logits, v = net(x)
        a, logp, vv = net.act(x, deterministic=False)
    print("logits:", logits.shape)  # [2, num_actions]
    print("value :", v.shape)       # [2]
    print("sampled action:", a.shape, a)
    print("logprob:", logp.shape)
