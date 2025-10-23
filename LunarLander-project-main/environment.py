import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

def make_env(render: bool = False):
    """
    Headless for training unless render=True (human window).
    """
    render_mode = "human" if render else None
    return gym.make("LunarLander-v3", render_mode=render_mode)

def make_env_human():
    """
    On-screen window for evaluation (requires local GUI).
    """
    return gym.make("LunarLander-v3", render_mode="human")

def make_env_rgb():
    """
    RGB array for RecordVideo (required for MP4 recording).
    """
    return gym.make("LunarLander-v3", render_mode="rgb_array")


# Optional helper for quick manual random rollout (debug)

def random_rollout(episodes: int = 3, render: bool = True, seed: int = 1, save_video: bool = False):
    """
    Runs random agent for a few episodes.
    If save_video=True, record MP4(s) into ./videos.
    """
    if save_video:
        os.makedirs("videos", exist_ok=True)
        env = RecordVideo(
            make_env_rgb(),
            video_folder="videos",
            name_prefix="random_eval",
            episode_trigger=lambda ep_idx: True,  # record every run
            video_length=0                        # full episode
        )
    else:
        env = make_env(render=render)

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
        if save_video:
            print("Saved random MP4(s) to ./videos")
    finally:
        env.close()
