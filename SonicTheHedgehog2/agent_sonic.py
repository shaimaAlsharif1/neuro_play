import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, network, optimizer, clip_range=0.2, gamma=0.99):
        self.net = network
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.gamma = gamma

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, obs, actions, old_logprobs, returns, advantages, epochs=4, batch_size=64):
        dataset_size = obs.size(0)
        for _ in range(epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                logits, values = self.net(obs[idx])
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - old_logprobs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, returns[idx])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
