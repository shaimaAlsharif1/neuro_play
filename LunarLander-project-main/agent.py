from __future__ import annotations

import math
import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import DQNConfig
from network import QNetwork  # swap to DuelingQNetwork if you like

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: str, cfg: DQNConfig):
        self.cfg = cfg
        self.device = device

        self.q = QNetwork(obs_dim, act_dim).to(device)
        self.target = QNetwork(obs_dim, act_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(cfg.buffer_size)
        self.act_dim = act_dim
        self.steps = 0

    def epsilon(self) -> float:
        # linear decay down to end_epsilon
        eps = self.cfg.end_epsilon + (self.cfg.start_epsilon - self.cfg.end_epsilon) * \
              max(0.0, (self.cfg.eps_decay_steps - self.steps) / self.cfg.eps_decay_steps)
        return max(self.cfg.end_epsilon, eps)

    def act(self, state: np.ndarray) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.act_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            q = self.q(s)
            return int(torch.argmax(q, dim=1).item())

    def push(self, *args):
        self.replay.push(*args)

    def train_step(self):
        if len(self.replay) < self.cfg.train_after or self.steps % self.cfg.train_every != 0:
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        a = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            max_next_q = self.target(ns).max(1, keepdim=True)[0]
            target = r + (1 - d) * self.cfg.gamma * max_next_q

        loss = self.criterion(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        if self.steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(self.q.state_dict(), path)
        print(f"Model saved at {path}")

    def load(self, path: str):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.q.state_dict())
        print(f"Loaded model from {path}")
