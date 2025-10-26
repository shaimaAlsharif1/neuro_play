# network_sonic.py
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
    def __init__(self, in_channels: int = 4, c3: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            orthogonal_init(nn.Conv2d(in_channels, 32, 8, stride=4), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Conv2d(32, 64, 4, stride=2), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Conv2d(64, c3, 3, stride=1), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
        )
        self._n_flatten = None

    def output_dim(self, input_shape):
        if self._n_flatten is not None:
            return self._n_flatten
        with torch.no_grad():
            c, h, w = input_shape
            dummy = torch.zeros(1, c, h, w)
            y = self.enc(dummy)
            self._n_flatten = y.view(1, -1).size(1)
        return self._n_flatten

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.enc(x)

# ---------- Actor-Critic with extra features ----------
class ActorCriticCNNExtra(nn.Module):
    def __init__(self, obs_shape, num_actions, extra_state_dim=4, hidden_size=512, conv3_channels=64):
        super().__init__()
        c, h, w = obs_shape
        self.encoder = SonicEncoder(in_channels=c, c3=conv3_channels)
        n_flat = self.encoder.output_dim(obs_shape)

        # Small MLP for extra scalar features
        self.extra_fc = nn.Sequential(
            nn.Linear(extra_state_dim, 32),
            nn.ReLU(inplace=True)
        )

        combined_dim = n_flat + 32

        # Policy head
        self.pi = nn.Sequential(
            orthogonal_init(nn.Linear(combined_dim, hidden_size), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(hidden_size, num_actions), gain=0.01),
        )

        # Value head
        self.v = nn.Sequential(
            orthogonal_init(nn.Linear(combined_dim, hidden_size), gain=nn.init.calculate_gain('relu')),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    # Forward pass
    def forward(self, obs, extra_state):
        z = self.encoder(obs).flatten(1)
        extra = self.extra_fc(extra_state)
        z = torch.cat([z, extra], dim=1)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value

    # Sample action
    @torch.no_grad()
    def act(self, obs, extra_state, deterministic=False):
        logits, value = self.forward(obs, extra_state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    # PPO loss helpers
    def evaluate_actions(self, obs, extra_state, actions):
        logits, value = self.forward(obs, extra_state)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value

    # ----- Save/Load -----
    # Save / load
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))




# ---------- Quick self-test ----------
# if __name__ == "__main__":
#     # Example for 4 stacked grayscale frames at 84x84
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     obs_shape = (4, 84, 84)   # or (4, 96, 96)
#     num_actions = 10
#     extra_dim = 4

#     net = ActorCriticCNNExtra(obs_shape, num_actions,extra_state_dim=extra_dim).to(device)
#     x = torch.zeros(2, *obs_shape, dtype=torch.uint8).to(device)  # batch of 2
#     with torch.no_grad():
#         logits, v = net(x)
#         a, logp, vv = net.act(x, deterministic=False)
#     print("logits:", logits.shape)  # [2, num_actions]
#     print("value :", v.shape)       # [2]
#     print("sampled action:", a.shape, a)
#     print("logprob:", logp.shape)



if __name__ == "__main__":
    # Example for 4 stacked grayscale frames at 84x84
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = (4, 84, 84)   # or (4, 96, 96)
    num_actions = 10
    extra_dim = 4  # e.g., lives, screen_x, screen_y, screen_x_end

    # Instantiate the network
    net = ActorCriticCNNExtra(obs_shape, num_actions, extra_state_dim=extra_dim).to(device)

    # Dummy batch of 2 observations
    batch_size = 2
    x = torch.zeros(batch_size, *obs_shape, dtype=torch.uint8).to(device)

    # Dummy extra state (batch_size x extra_dim)
    extra_state = torch.zeros(batch_size, extra_dim, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Forward pass
        logits, v = net(x, extra_state)

        # Sample action
        a, logp, vv = net.act(x, extra_state, deterministic=False)

    print("logits:", logits.shape)  # [batch_size, num_actions]
    print("value :", v.shape)       # [batch_size]
    print("sampled action:", a.shape, a)  # [batch_size] tensor of sampled actions
    print("logprob:", logp.shape)   # [batch_size]
