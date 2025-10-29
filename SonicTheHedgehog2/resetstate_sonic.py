# resetstate_sonic.py
import gymnasium as gym
import numpy as np
import os
import pandas as pd

class ResetStateWrapper(gym.Wrapper):
    """
    Custom reward shaping for Sonic the Hedgehog 2.
    Encourages moving forward
    Penalizes idling and useless jump spam
    """

    def __init__(self, env, max_steps=4500):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

    def reset(self, **kwargs):
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0
        obs, info = self.env.reset(**kwargs)
        self.prev_info = info
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # --- Reward shaping ---
        custom_reward = 0.0

        # Extract info fields (fallback defaults if missing)
        x = info.get("x", 0)
        score = info.get("score", 0)
        lives = info.get("lives", 3)
        screen_x_end = info.get("screen_x_end", 10000)

        if self.prev_info is None:
            self.prev_info = info

        prev_x = self.prev_info.get("x", 0)
        prev_lives = self.prev_info.get("lives", 3)

        # 1) Reward forward progress
        dx = x - prev_x
        if dx > 0:
            custom_reward += 0.1 * (dx / 100.0)
        elif dx == 0:
            custom_reward -= 0.05  # stronger penalty for staying still
        elif dx < 0:
            custom_reward -= 0.1  # penalty for moving backward

        # 2) Small dense reward for proximity to level end
        custom_reward += (x / screen_x_end) * 0.5

        # 3) Penalty for losing a life (and end episode)
        if lives < prev_lives:
            custom_reward -= 1.0
            done = True

        # 4) Bonus for finishing the level
        if x >= screen_x_end:
            custom_reward += 1.0
            done = True

        # --- Jump control (reduce useless jumping) ---
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = ['A', 'B', 'C']

        # Check if action involves jumping
        is_jump = False
        if hasattr(self.env, 'action_space'):
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # For discrete actions, check if it's a jump action
                if hasattr(self.env, '_decode_discrete_action'):
                    action_vec = self.env._decode_discrete_action[action]
                    is_jump = any(action_vec[buttons.index(b)] if b in buttons else False for b in jump_buttons)

        if is_jump:
            self.jump_counter += 1
        else:
            self.jump_counter = 0

        # Penalize jumping without forward movement
        if is_jump and dx <= 0:
            custom_reward -= 0.02

        # Allow more jumps before penalizing
        if self.jump_counter > 5:
            custom_reward -= 0.02 * (self.jump_counter - 5)

        # Penalize jump spam beyond 3 consecutive jumps
        if self.jump_counter > 3:
            custom_reward -= 0.1 * (self.jump_counter - 3)

        # --- Episode step cap ---
        self.steps += 1
        if self.steps > self.max_steps:
            done = True

        # --- Clip reward to a reasonable range ---
        custom_reward = np.clip(custom_reward, -1.0, 1.0)

        # Update previous info
        self.prev_info = info

        # Combine rewards
        total_reward = env_reward + custom_reward

        return obs, total_reward, done, False, info
