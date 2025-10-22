# main_lander.py — LunarLander (Gymnasium) random rollout + DQN training
# -------------------------------------------------------------
# Requirements (Ubuntu/WSL):
#   sudo apt-get update && sudo apt-get install -y swig python3-dev build-essential
#   pip install gymnasium[box2d]==0.29.1 pygame==2.5.2 numpy torch==2.3.1
#
# Usage examples:
#   # Random play for 5 episodes (no training)
#   python main_lander.py --random --episodes 5 --render
#
#   # Train DQN for 50k steps, then evaluate 5 episodes
#   python main_lander.py --train --train_steps 50000 --eval_episodes 5
#
from __future__ import annotations
import argparse
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass

# import and setup
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

# stores past experiences: state, action, reward, next_state, done
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


#Neural network Qnet, that approximate the Q function,
# input : state, output : Q values
class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# stores all the hyperparameters,
@dataclass
class DQNConfig:
    gamma: float = 0.99 #discount factor
    lr: float = 1e-3 #learning rate
    batch_size: int = 128
    buffer_size: int = 100_000
    start_epsilon: float = 1.0 # explorationi rate
    end_epsilon: float = 0.05
    eps_decay_steps: int = 30_000 # how queckly epsilon decreases
    target_update_every: int = 1_000
    train_after: int = 5_000  #start training after this steps
    train_every: int = 4
    max_grad_norm: float = 10.0

# the DQN agent
class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, device: str, cfg: DQNConfig):
        self.device = device
        self.cfg = cfg
        self.q = QNet(obs_dim, act_dim).to(device)
        self.target = QNet(obs_dim, act_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size)
        self.steps = 0
        self.act_dim = act_dim
    #epsilon policy
    def epsilon(self):
        # Linear decay
        eps = self.cfg.end_epsilon + (self.cfg.start_epsilon - self.cfg.end_epsilon) * \
              max(0.0, (self.cfg.eps_decay_steps - self.steps) / self.cfg.eps_decay_steps)
        return eps
    # chossing the action
    def act(self, state: np.ndarray) -> int:
        self.steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.act_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s)
            return int(torch.argmax(q, dim=1).item())

    def push(self, *args):
        self.replay.push(*args)
    # training steps, sample a batch , compute the Q using the network, compute target, compute loss,
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
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optim.step()
        if self.steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())



#the environment set up
def make_env(render: bool):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return env

# runs the game randomly, no training and learning
def random_rollout(episodes: int = 5, render: bool = False):
    env = make_env(render)
    try:
        for ep in range(1, episodes + 1):
            s, _ = env.reset()
            done, tr = False, False
            total = 0.0
            while not (done or tr):
                a = env.action_space.sample()
                ns, r, done, tr, _ = env.step(a)
                total += r
                s = ns
            print(f"[Random] Episode {ep}/{episodes} | Return: {total:.2f}")
    finally:
        env.close()

# training DQN,
# in each loop:
# select the action
# executes the environment,
# stores transitions in replay buffer
# perform training step
# resets the environment at the end of the ep

def train_dqn(train_steps: int = 50_000, eval_episodes: int = 5, render_eval: bool = False, seed: int = 1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = make_env(render=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DQNConfig()
    agent = DQNAgent(obs_dim, act_dim, device, cfg)

    s, _ = env.reset(seed=seed)
    episode_return = 0.0
    for t in range(1, train_steps + 1):
        a = agent.act(s)
        ns, r, done, tr, _ = env.step(a)
        agent.push(s, a, r, ns, float(done or tr))
        loss = agent.train_step()
        s = ns
        episode_return += r
        if done or tr:
            print(f"Step {t:6d} | eps={agent.epsilon():.3f} | return={episode_return:.1f} | replay={len(agent.replay)}" )
            episode_return = 0.0
            s, _ = env.reset()
    env.close()

    # Evaluate
    print("\nEvaluating...")
    eval_env = make_env(render_eval)
    try:
        returns = []
        for ep in range(1, eval_episodes + 1):
            s, _ = eval_env.reset()
            done, tr = False, False
            total = 0.0
            while not (done or tr):
                with torch.no_grad():
                    q = agent.q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                    a = int(torch.argmax(q, dim=1).item())
                s, r, done, tr, _ = eval_env.step(a)
                total += r
            returns.append(total)
            print(f"[Eval] Episode {ep}/{eval_episodes} | Return: {total:.2f}")
        print(f"Avg return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    finally:
        eval_env.close()

# for command line arguments
def parse_args():
    p = argparse.ArgumentParser(description="LunarLander-v2 random or DQN training")
    p.add_argument("--random", action="store_true", help="Run random agent for a few episodes")
    p.add_argument("--episodes", type=int, default=5, help="Episodes for random run")
    p.add_argument("--render", action="store_true", help="Render during random/eval runs")
    p.add_argument("--train", action="store_true", help="Train DQN")
    p.add_argument("--train_steps", type=int, default=50_000, help="Training steps")
    p.add_argument("--eval_episodes", type=int, default=5, help="Evaluation episodes after training")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    return p.parse_args()

# the main function
def main():
    args = parse_args()
    if args.random:
        random_rollout(episodes=args.episodes, render=args.render)
    elif args.train:
        train_dqn(train_steps=args.train_steps, eval_episodes=args.eval_episodes, render_eval=args.render, seed=args.seed)
    else:
        print("Select a mode: --random or --train (see --help)")


if __name__ == "__main__":
    main()
