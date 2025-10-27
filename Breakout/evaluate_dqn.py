# evaluate_dqn.py
import numpy as np
from tensorflow import keras
from collections import deque
import os
import ale_py
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, AtariPreprocessing
import time

# ================================================================
# ✅ Environment creation (moved inside this file to avoid confusion)
# ================================================================
def make_breakout_env(video_folder="Breakout/videos_eval", name_prefix="breakout_eval"):

    os.makedirs(video_folder, exist_ok=True)

    # Disable original frameskip (set frameskip=1)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)

    # Record videos for all episodes
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda ep_idx: True,
        video_length=0
    )

    # Apply Atari preprocessing (this handles frame_skip correctly)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)

    return env


# ================================================================
# ✅ Evaluation logic
# ================================================================

import time  # for unique video folders
def evaluate(npz_path="Breakout/dqn_breakout_full_state_0300.npz", episodes=3, base_video_folder="Breakout/videos_eval"):
    print("\033[92m🎮 Loading .npz Q-table or prediction data...\033[0m")
    data = np.load(npz_path, allow_pickle=True)
    print("Loaded keys:", data.files)

    # Example: assume Q-table named 'q_table'
    q_table = data[data.files[0]]  # or replace with 'q_table' if key known

    timestamp = int(time.time())
    video_folder = base_video_folder
    env = make_breakout_env(video_folder)

    total_rewards = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            frame_stack = deque([obs] * 4, maxlen=4)
            state = np.stack(frame_stack, axis=-1)
            done = False
            total_reward = 0

            while not done:
                # Example placeholder: pick random or simple policy
                action = np.random.randint(0, env.action_space.n)
                obs, reward, done, _, _ = env.step(action)
                frame_stack.append(obs)
                total_reward += reward

            print(f"[Eval] Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}")
            total_rewards.append(total_reward)
    finally:
        env.close()

    avg_reward = np.mean(total_rewards)
    print(f"\n✅ Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"🎥 Videos saved in '{video_folder}/'")



# ================================================================
# ✅ Run as main
# ================================================================
if __name__ == "__main__":
    evaluate(episodes=5)
