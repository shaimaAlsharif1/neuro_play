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
def evaluate(model_path="dqn_breakout_model.keras",
             episodes=10,
             base_video_folder="Breakout/videos_eval"):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

    print("\033[92m🎮 Loading trained model...\033[0m")
    model = keras.models.load_model(model_path)
    total_rewards = []

    # Unique folder for this run
    timestamp = int(time.time())
    video_folder = f"{base_video_folder}/run_{timestamp}"
    env = make_breakout_env(video_folder=video_folder, name_prefix="breakout_eval")

    try:
        for ep in range(episodes):
            print("\033[92m now evaluating...\033[0m")
            obs, _ = env.reset()
            frame_stack = deque([obs] * 4, maxlen=4)
            state = np.stack(frame_stack, axis=-1)
            done = False
            total_reward = 0

            while not done:
                q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
                action = np.argmax(q_values[0])
                obs, reward, done, _, _ = env.step(action)
                frame_stack.append(obs)
                state = np.stack(frame_stack, axis=-1)
                total_reward += reward

            print(f"[Eval] Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}")
            total_rewards.append(total_reward)

    finally:
        env.close()  # closes once, saves all videos

    avg_reward = np.mean(total_rewards)
    print(f"\n✅ Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"🎥 Videos saved in '{video_folder}/'")


# ================================================================
# ✅ Run as main
# ================================================================
if __name__ == "__main__":
    evaluate(episodes=1)
