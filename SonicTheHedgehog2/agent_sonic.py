import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, net, optimizer, clip_range=0.2, grad_clip=0.5, entropy_coef=0.02):
        self.net = net
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.grad_clip = grad_clip
        self.entropy_coef = entropy_coef  # can be replaced by a schedule fn if desired

    @torch.no_grad()
    def act(self, obs, extra):
        logits, value = self.net(obs, extra)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value.squeeze(-1)

    def update(
        self,
        obs, extra, actions, old_logprobs, returns, values_old, advantages,
        epochs=4, batch_size=64, entropy_coef=None, global_steps=None
    ):
        """
        Single, canonical PPO update with value clipping and entropy bonus.
        DO NOT duplicate updates elsewhere in the trainer.
        """
        self.net.train()
        N = obs.size(0)
        ent_coef = self.entropy_coef if entropy_coef is None else (
            entropy_coef(global_steps) if callable(entropy_coef) else float(entropy_coef)
        )

        for _ in range(epochs):
            indices = torch.randperm(N, device=obs.device)
            for start in range(0, N, batch_size):
                idx = indices[start:start + batch_size]

                logits, values = self.net(obs[idx], extra[idx])
                dist = Categorical(logits=logits)

                # Policy loss (clipped surrogate)
                new_logp = dist.log_prob(actions[idx])
                ratio = torch.exp(new_logp - old_logprobs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v = values.squeeze(-1)
                v_old = values_old[idx]
                v_clipped = v_old + (v - v_old).clamp(-self.clip_range, self.clip_range)
                v_loss_unclipped = (v - returns[idx])**2
                v_loss_clipped   = (v_clipped - returns[idx])**2
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy bonus
                entropy = dist.entropy().mean()

                loss = actor_loss + critic_loss - ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()
