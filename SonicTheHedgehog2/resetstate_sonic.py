import gymnasium as gym
import numpy as np
import os
import pandas as pd

def _cap(x, lo, hi):
    return max(lo, min(hi, x))

class ResetStateWrapper(gym.Wrapper):
    """
    Sonic 2 – Reward Shaping (Exploration-Safe)
    -------------------------------------------
    - Rewards rightward progress and speed.
    - Keeps exploration alive with mild penalties and small base reward.
    - Encourages early jumps to learn locomotion.
    - Delays early resets to avoid training collapse.
    """

    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps

        # episode state
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

        # progress tracking
        self.x_best = 0
        self.no_progress_steps = 0
        self.right_streak = 0
        self.best_dx = 0
        self.dx_window = [0.0] * 10

        # logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_records = []
        self.episode_id = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = (out if isinstance(out, tuple) else (out, {}))

        self.steps = 0
        self.prev_info = info
        self.jump_counter = 0

        self.x_best = info.get("x", 0)
        self.no_progress_steps = 0
        self.right_streak = 0
        self.best_dx = 0
        self.dx_window = [0.0] * 10
        self.episode_records = []

        return obs, info

    def step(self, action):
        step = self.env.step(action)

        # normalize to 5-tuple
        if len(step) == 5:
            obs, reward_env, terminated, truncated, info = step
            done = terminated or truncated
        else:
            obs, reward_env, done, info = step
            terminated, truncated = done, False

        # ---------------- Reward Shaping ----------------
        R = 0.0

        x          = info.get("x", 0)
        lives      = info.get("lives", 3)
        screen_end = info.get("screen_x_end", 10_000)

        if self.prev_info is None:
            self.prev_info = {"x": 0, "lives": lives}
        prev_x    = self.prev_info.get("x", 0)
        prev_life = self.prev_info.get("lives", lives)
        dx = x - prev_x

        # update speed window
        self.dx_window.pop(0)
        self.dx_window.append(float(dx))
        dx_avg = sum(self.dx_window) / len(self.dx_window)

        # (A) Base survival reward — prevents collapse
        R += 0.05

        # (B) Reward for ANY rightward movement
        if dx > 0:
            R += 0.2 * _cap(dx, 0, 12)
            self.right_streak += 1
        elif dx == 0:
            R -= 0.1  # mild idle penalty
            self.right_streak = 0
        else:
            R -= 1.0  # lighter backtrack penalty
            self.right_streak = 0

        # (C) Reward new ground progress (beat x_best)
        new_progress = max(0, x - self.x_best)
        if new_progress > 0:
            R += 2.0 * (new_progress / 5.0)
            self.x_best = x
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        # (D) Momentum streak bonus
        if self.right_streak > 0 and (self.right_streak % 20) == 0:
            R += 8.0

        # (E) Burst bonus (speed burst)
        if dx > self.best_dx and dx > 4:
            R += 5.0 + 0.5 * _cap(dx - self.best_dx, 0, 8)
            self.best_dx = dx

        # (F) Light time pressure
        R -= 0.01

        # (G) Jump shaping
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = {'A', 'B', 'C'}

        # convert discrete action to button array if needed
        if hasattr(self.env, "action") and isinstance(action, (int, np.integer)):
            try:
                action_array = self.env.action(action)
            except Exception:
                action_array = np.zeros(len(buttons), dtype=np.int8)
        else:
            action_array = action

        pressed = [buttons[i] for i, val in enumerate(action_array) if val == 1]
        is_jump = any(b in jump_buttons for b in pressed)
        self.jump_counter = self.jump_counter + 1 if is_jump else 0

        if is_jump:
            if dx <= 0:
                R -= 0.5  # mild anti-jump penalty (was -2.0)
            else:
                R += 0.5  # productive jump reward

            # Early curiosity bonus for learning jump behavior
            if self.steps < 3000:
                R += 0.3

        # (H) Stuck penalty and delayed early reset
        if self.no_progress_steps and self.no_progress_steps % 120 == 0:
            R -= 2.0  # gentler penalty
        if self.no_progress_steps > 720:  # patience before reset
            R -= 10.0
            done = True

        # (I) Terminal events
        if lives < prev_life:
            R -= 40.0
            done = True

        if x >= screen_end:
            R += 400.0
            done = True

        # bookkeeping
        self.steps += 1
        if self.steps > self.max_steps:
            done = True

        self.episode_records.append({
            "step": self.steps,
            "action": int(action) if isinstance(action, (int, np.integer)) else -1,
            "is_jump": bool(is_jump),
            "x": int(x),
            "dx": int(dx),
            "dx_avg": float(dx_avg),
            "x_best": int(self.x_best),
            "streak": int(self.right_streak),
            "reward": float(round(R, 4)),
        })

        self.prev_info = info

        return obs, float(R), terminated, truncated, info
