# utils.py
"""
Utility wrappers and helper functions for the Sonic RL environment.
Includes frame skipping, grayscale resizing, and normalization.
"""

import gymnasium as gym
import numpy as np
import cv2

# ===========================================================
# 🟩 1. SkipFrame Wrapper
# ===========================================================
class SkipFrame(gym.Wrapper):
    """
    Repeats the same action for N frames to speed up training.
    Rewards are accumulated across skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Always return 5 values: obs, reward, terminated, truncated, info"""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            step = self.env.step(action)

            # ✅ التعامل مع الحالتين (٤ قيم أو ٥ قيم)
            if len(step) == 5:
                obs, reward, term, trunc, info = step
                terminated = term or terminated
                truncated = trunc or truncated
            else:
                obs, reward, done, info = step
                terminated = done or terminated
                truncated = truncated or False

            total_reward += reward
            if terminated or truncated:
                break

        # ✅ إرجاع ٥ قيم دائمًا
        return obs, total_reward, terminated, truncated, info

# ===========================================================
# 🟨 2. GrayResizeWrapper
# ===========================================================
class GrayResizeWrapper(gym.ObservationWrapper):
    """
    Converts RGB frames to grayscale and resizes them (e.g., 84x84).
    """
    def __init__(self, env, width=84, height=84, keep_dim=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.keep_dim = keep_dim  # True → (H, W, 1), False → (H, W)

        shape = (self.height, self.width, 1) if keep_dim else (self.height, self.width)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.keep_dim:
            resized = np.expand_dims(resized, axis=-1)
        return resized.astype(np.uint8)


# ===========================================================
# 🟦 3. NormalizeObs
# ===========================================================
class NormalizeObs(gym.ObservationWrapper):
    """
    Normalizes pixel values from [0,255] → [0,1].
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0
