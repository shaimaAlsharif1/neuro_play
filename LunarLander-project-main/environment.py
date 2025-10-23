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
                shaped_r = custom_reward(s, a, r, done, tr)
                total += shaped_r

            print(f"[random] ep {ep}/{episodes} return={total:.2f}")
        if save_video:
            print("Saved random MP4(s) to ./videos")
    finally:
        env.close()

def custom_reward(state, action, reward, done, truncated):
    """
    Modify the reward signal based on state/action.
    Encourages upright posture, smooth descent, and successful landing.
    """
    angle = state[4]
    angular_velocity = state[5]
    left_leg_contact = state[6]
    right_leg_contact = state[7]

    # Penalize excessive tilt
    if abs(angle) > 0.2:
        reward -= 2.0

    # Penalize spinning
    if abs(angular_velocity) > 0.5:
        reward -= 1.0

    # Bonus for both legs touching down
    if left_leg_contact and right_leg_contact:
        reward += 5.0

    # Bonus for successful landing (env gives +100)
    if done and not truncated and reward >= 100:
        reward += 10.0

    return reward