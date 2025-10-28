import gymnasium as gym
import numpy as np
import os
import pandas as pd

def _cap(x, lo, hi):
    return max(lo, min(hi, x))

class ResetStateWrapper(gym.Wrapper):
    """
    AGGRESSIVE reward shaping for Sonic 2.
    - Huge incentives for rightward progress and speed.
    - Momentum streak + burst bonuses.
    - Strong penalties for idling, backtracking, and useless jumps.
    - Early stuck resets to recycle quickly.
    """
    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps

        # episode state
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

        # progress/speed tracking
        self.x_best = 0
        self.no_progress_steps = 0
        self.right_streak = 0               # consecutive steps with dx > 0
        self.best_dx = 0                    # best single-step dx this episode
        self.dx_window = [0.0] * 10         # short moving window for avg speed

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
        self.right_streak = 0
        self.best_dx = 0
        self.dx_window = [0.0] * 10

        self.episode_records = []
        return obs, info

    def step(self, action):
        step = self.env.step(action)

        # Normalize to 5-tuple
        if len(step) == 5:
            obs, reward_env, terminated, truncated, info = step
            done = terminated or truncated
        else:
            obs, reward_env, done, info = step
            terminated, truncated = done, False

        # -------- Reward shaping (AGGRESSIVE) --------
        R = 0.0

        x          = info.get("x", 0)
        lives      = info.get("lives", 3)
        screen_end = info.get("screen_x_end", 10_000)

        if self.prev_info is None:
            self.prev_info = {"x": 0, "lives": lives}
        prev_x    = self.prev_info.get("x", 0)
        prev_life = self.prev_info.get("lives", lives)

        dx = x - prev_x

        # Update speed window
        self.dx_window.pop(0)
        self.dx_window.append(float(dx))
        dx_avg = sum(self.dx_window) / len(self.dx_window)

        # (A) NEW PROGRESS: big pay only when beating best x
        new_progress = max(0, x - self.x_best)
        if new_progress > 0:
            # Scale up: ~ +2 per 5 px of fresh ground (was tiny before)
            R += 2.0 * (new_progress / 5.0)
            self.x_best = x
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        # (B) PER-STEP SPEED (rightward)
        if dx > 0:
            # Cap dx per step to avoid rare spikes dominating
            dx_c = _cap(dx, 0, 12)  # tune if your skip/frame rate differs
            R += 0.4 * dx_c         # was 0.01*dx; now ~40x larger (capped)
            self.right_streak += 1
        elif dx == 0:
            R -= 0.6                 # strong anti-idle
            self.right_streak = 0
        else:
            R -= 3.0                 # strong anti-backtrack
            self.right_streak = 0

        # (C) MOMENTUM STREAK BONUS: every 20 steps of forward motion
        if self.right_streak > 0 and (self.right_streak % 20) == 0:
            R += 8.0

        # (D) BURST BONUS: break your best per-step dx
        if dx > self.best_dx and dx > 4:
            R += 5.0 + 0.5 * _cap(dx - self.best_dx, 0, 8)  # extra for bigger bursts
            self.best_dx = dx

        # (E) LIGHT TIME PRESSURE (still present, but small)
        R -= 0.02

        # (F) Jump shaping
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = {'A', 'B', 'C'}

        # Map discrete action to button array when needed
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
            # hammer jumps that don't push forward
            R -= 2.0
        elif is_jump and dx > 0:
            # small positive nudge for a productive jump (optional)
            R += 0.5

        # (G) Early stuck penalties + optional early reset
        if self.no_progress_steps and self.no_progress_steps % 120 == 0:
            R -= 4.0   # every ~2s with no new best x, whack it

        if self.no_progress_steps > 360:  # ~6s stuck -> reset faster
            done = True

        # (H) Terminal events (keep big signals)
        if lives < prev_life:
            R -= 40.0
            done = True

        if x >= screen_end:
            R += 400.0
            done = True

        # ------------- bookkeeping -------------
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

        # return shaped reward only
        return obs, float(R), terminated, truncated, info
