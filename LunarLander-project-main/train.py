from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from agent import DQNAgent
from config import DQNConfig
from environment import make_env, make_env_human, make_env_rgb
import numpy as np


def evaluate(agent,
             eval_episodes: int = 3,
             render_window: bool = True,
             save_video: bool = True,
             seed: int = 1):
    """
    Evaluate the trained agent.
    - render_window=True -> show live window (requires local GUI)
    - save_video=True    -> write MP4(s) into ./videos using rgb_array mode
    """
    device = next(agent.q.parameters()).device

    # (A) live window
    if render_window:
        env = make_env_human()
        try:
            for ep in range(1, eval_episodes + 1):
                s, _ = env.reset(seed=seed + ep)
                done = tr = False
                total = 0.0
                while not (done or tr):
                    with torch.no_grad():
                        q = agent.q(torch.tensor(s, dtype=torch.float32, device=device))
                        a = int(q.argmax(dim=1).item())
                    s, r, done, tr, _ = env.step(a)
                    total += r
                print(f"[eval-human] episode {ep}/{eval_episodes} return={total:.2f}")
        finally:
            env.close()

    # (B) MP4 video
    if save_video:
                # MP4 video block inside evaluate()
        os.makedirs("LunarLander-project-main/videos", exist_ok=True)
        env_v = RecordVideo(
            make_env_rgb(),
            video_folder="videos/eval_video",
            name_prefix="lander_eval",
            episode_trigger=lambda ep_idx: True,  # record every eval run
            video_length=0                        # 0 = full episode
        )

        try:
            s, _ = env_v.reset(seed=seed + 999)
            done = tr = False
            while not (done or tr):
                with torch.no_grad():
                    q = agent.q(torch.tensor(s, dtype=torch.float32, device=device))
                    a = int(q.argmax(dim=1).item())
                s, _, done, tr, _ = env_v.step(a)
            print("Saved MP4(s) to ./videos")
        finally:
            env_v.close()


def train_dqn(train_steps: int = 50_000,
              eval_episodes: int = 5,
              render_eval: bool = False,
              save_video: bool = False,
              seed: int = 1,
              model_path: str = "LunarLander-project-main/dqn_lunarlander.pth"):
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Headless training env
    env = make_env(render=False)

    # Create env with rgb_array mode if saving video
    if save_video:
        os.makedirs("videos/train_video", exist_ok=True)
        env = make_env_rgb()  # rgb_array for RecordVideo
        env = RecordVideo(
            env,
            video_folder="videos/train_video",
            name_prefix="lander_train",
            episode_trigger=lambda ep_idx: True,  # record every episode
            video_length=0                        # full episode
        )
        print("\033[92m🎥 Training video recording enabled\033[0m")
    else:
        env = make_env(render=False)  # normal headless env

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DQNConfig()
    agent = DQNAgent(obs_dim, act_dim, device, cfg)

     # Load existing model if found
    if os.path.exists(model_path):
        print("\033[93m Loading existing model...\033[0m")
        agent.load(model_path)

    else:
        print("\033[93m Training from scratch... \033[0m")

    s, _ = env.reset(seed=seed)
    ep_ret = 0.0
    # training loop
    episode_rewards = []

    for t in range(1, train_steps + 1):
        a = agent.act(s)
        ns, r, done, tr, _ = env.step(a)
        agent.push(s, a, r, ns, float(done or tr))
        agent.train_step()

        s = ns
        ep_ret += r

        if done or tr:
            print(f"[train] step={t:6d} eps={agent.epsilon():.3f} return={ep_ret:7.2f} replay={len(agent.replay)}")
            episode_rewards.append(ep_ret)
            ep_ret = 0.0
            s, _ = env.reset()


    env.close()
    agent.save(model_path)

    # Evaluate at the end
    evaluate(agent,
             eval_episodes=eval_episodes,
             render_window=render_eval,
             save_video=save_video,
             seed=seed)

    total_episodes = len(episode_rewards)
    total_steps = train_steps
    max_reward = np.max(episode_rewards) if episode_rewards else 0
    min_reward = np.min(episode_rewards) if episode_rewards else 0


    print("\n" + "="*40)
    print("🎯 Training Summary")
    print("="*40)
    print(f"Total steps        : {total_steps}")
    print(f"Total episodes     : {total_episodes}")
    print(f"Max reward         : {max_reward:.2f}")
    print(f"Min reward         : {min_reward:.2f}")
    print(f"Final epsilon      : {agent.epsilon():.3f}")
    print(f"Replay buffer size : {len(agent.replay)}")
    print("="*40 + "\n")
