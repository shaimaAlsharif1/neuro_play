import gymnasium as gym
import numpy as np
import os
import pandas as pd

class ResetStateWrapper(gym.Wrapper):
    """
    Reward shaping + logging for Sonic 2.
    - Pays only for *new* forward progress
    - Tiny bonus for per-step rightward velocity
    - Light penalties for idling/backtracking/useless jumps
    - Optional early reset when stuck
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

        # logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_records = []
        self.episode_id = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        self.steps = 0
        self.prev_info = info
        self.jump_counter = 0
        self.x_best = info.get("x", 0)
        self.no_progress_steps = 0
        self.episode_records = []
        return obs, info

    def step(self, action):
        """One environment step with shaped reward and logging."""
        step = self.env.step(action)

        # Support both 5-tuple and 4-tuple env.step outputs
        if len(step) == 5:
            obs, reward, terminated, truncated, info = step
            done = terminated or truncated
        else:
            obs, reward, done, info = step
            terminated, truncated = done, False

        # ------- Shaped reward -------
        custom = 0.0

        # read info
        x           = info.get("x", 0)
        lives       = info.get("lives", 3)
        screen_end  = info.get("screen_x_end", 10000)

        if self.prev_info is None:
            self.prev_info = {"x": 0, "lives": lives}
        prev_x    = self.prev_info.get("x", 0)
        prev_life = self.prev_info.get("lives", lives)

        # rightward delta this step
        dx = x - prev_x

        # (A) pay only for beating best-ever x this episode
        new_progress = max(0, x - self.x_best)
        if new_progress > 0:
            custom += 0.2 * (new_progress / 5.0)  # event-based progress reward
            self.x_best = x
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        # (B) tiny speed bonus for moving right this step
        if dx > 0:
            custom += 0.01 * dx
        elif dx == 0:
            custom -= 0.02
        else:
            custom -= 0.2  # going left is bad

        # (C) gentle time pressure
        custom -= 0.02

        # (D) light jump shaping (keep exploration alive)
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = {'A', 'B', 'C'}

        # Map discrete action -> button array if needed
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

        if is_jump and dx <= 0:
            custom -= 0.05  # jumping in place/backwards is slightly bad

        # (E) early stuck penalty + optional early reset
        if self.no_progress_steps and self.no_progress_steps % 120 == 0:
            custom -= 0.5   # every ~2s without new best x

        if self.no_progress_steps > 900:  # ~15s with skip=2 @60fps base
            done = True

        # (F) terminal events
        if lives < prev_life:
            custom -= 20.0
            done = True

        if x >= screen_end:
            custom += 200.0
            done = True

        # step bookkeeping / logging
        self.steps += 1
        if self.steps > self.max_steps:
            done = True

        self.episode_records.append({
            "step": self.steps,
            "action": int(action) if isinstance(action, (int, np.integer)) else -1,
            "is_jump": bool(is_jump),
            "x": int(x),
            "dx": int(dx),
            "x_best": int(self.x_best),
            "reward": float(round(custom, 4)),
        })

        # update previous info
        self.prev_info = info

        # return shaped reward (ignore env's raw reward)
        return obs, float(custom), terminated, truncated, info
