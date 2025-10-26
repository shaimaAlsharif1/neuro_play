import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
from gymnasium.wrappers import FrameStackObservation
import ale_py
import numpy as np
import os
from config import DQNConfig


# env = gym.make("BreakoutNoFrameskip-v4" ,render_mode="human")
# # Environment preprocessing
# env = AtariPreprocessing(env)
# # Stack four frames
# env = FrameStack(env, 4)
# env.seed(DQNConfig.seed)

# env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
# env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
# obs, _ = env.reset()
# # print(obs.shape)

def make_env(render: bool = False):
    """
    Headless for training unless render=True (human window).
    """
    render_mode = "human" if render else None
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)

    # Stack four frames
    # env = FrameStackObservation(env, 4)
    # env.seed(DQNConfig.seed)
    return env

def make_env_human():
    """
    On-screen window for evaluation (requires local GUI).
    """
    env =  gym.make("ALE/Breakout-v5", render_mode="human")
    env = AtariPreprocessing(env)
    # Stack four frames
    # env = FrameStackObservation(env, 4)
    # env.seed(DQNConfig.seed)
    return env

def make_env_rgb():
    """
    RGB array for RecordVideo (required for MP4 recording).
    """
    env =  gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = AtariPreprocessing(env)
    # Stack four frames
    # env = FrameStackObservation(env, 4)
    # env.seed(DQNConfig.seed)

    return env


# Optional helper for quick manual random rollout (debug)

def random_rollout(episodes: int = 3, render: bool = True, seed: int = 1, save_video: bool = False):
    """
    Runs random agent for a few episodes.
    If save_video=True, record MP4(s) into ./videos.
    """

    # if save_video:
    #     os.makedirs("LunarLander-project-main/videos/random_video", exist_ok=True)
    #     env = RecordVideo(
    #         make_env_rgb(),
    #         video_folder="videos/random_video",
    #         name_prefix="random_eval",
    #         episode_trigger=lambda ep_idx: True,  # record every run
    #         video_length=0                        # full episode
    #     )
    # else:
    #     env = make_env(render=render)

    env = make_env(render=render)
    env = AtariPreprocessing(env)
    # Stack four frames
    # env = FrameStackObservation(env, 4)
    # env.seed(DQNConfig.seed)
    try:
        for ep in range(1, episodes + 1):
            s, _ = env.reset(seed=seed + ep)
            done = tr = False
            total = 0.0
            while not (done or tr):
                a = env.action_space.sample()
                s, r, done, tr, _ = env.step(a)
                total += r
            print(f"[random] ep {ep}/{episodes} return={total:.2f}")
        # if save_video:
        #     print("Saved random MP4(s) to ./videos")
    finally:
        env.close()
