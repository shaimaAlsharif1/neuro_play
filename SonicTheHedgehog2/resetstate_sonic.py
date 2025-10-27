import gymnasium as gym
import numpy as np
import os
import pandas as pd


def _cap(x, lo, hi):
    return max(lo, min(hi, x))


class ResetStateWrapper(gym.Wrapper):
    """
    Aggressive reward shaping for Sonic 2 with Boss Mode.

    • Massive reward for going right & fast (progress, streaks, bursts)
    • Strong ring shaping: collect big +, lose big –
    • Boss Mode: camera-locked arenas pay for score/hits and survival;
      relax backtrack/idle penalties; disable 'stuck' resets
    • Exposes `self.in_boss` for action gating (e.g., enable LEFT only in boss)
    """

    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps

        # episode state
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

        # movement/progress trackers
        self.x_best = 0
        self.no_progress_steps = 0
        self.right_streak = 0
        self.best_dx = 0
        self.dx_window = [0.0] * 10

        # boss phase flag (exposed)
        self.in_boss = False

        # logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.episode_records = []
        self.episode_id = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})

        self.steps = 0
        self.prev_info = info
        self.jump_counter = 0

        self.x_best = info.get("x", 0)
        self.no_progress_steps = 0
        self.right_streak = 0
        self.best_dx = 0
        self.dx_window = [0.0] * 10

        self.in_boss = False
        self.episode_records = []
        return obs, info

    def step(self, action):
        step = self.env.step(action)

        # Normalize to 5-tuple from inner env
        if len(step) == 5:
            obs, reward_env, terminated, truncated, info = step
            done = bool(terminated or truncated)
        else:
            obs, reward_env, done, info = step
            terminated, truncated = done, False

        # ---------------- read state ----------------
        x          = info.get("x", 0)
        lives      = info.get("lives", 3)
        screen_end = info.get("screen_x_end", 10_000)
        rings      = info.get("rings", 0)
        score      = info.get("score", 0)

        if self.prev_info is None:
            self.prev_info = {
                "x": 0, "lives": lives, "rings": rings, "score": score
            }

        prev_x     = self.prev_info.get("x", 0)
        prev_life  = self.prev_info.get("lives", lives)
        prev_rings = self.prev_info.get("rings", rings)
        prev_score = self.prev_info.get("score", score)

        dx     = x - prev_x
        dr     = rings - prev_rings
        dscore = score - prev_score

        # rolling avg of dx (for logging/diagnostics)
        self.dx_window.pop(0)
        self.dx_window.append(float(dx))
        dx_avg = sum(self.dx_window) / len(self.dx_window)

        # ---------------- boss detection ----------------
        # Heuristic: near the end AND scrolling has stalled for a bit
        near_end = (max(self.x_best, x) >= 0.90 * float(screen_end))
        scroll_stalled = (self.no_progress_steps > 180)  # ~3s w/ skip=2

        if (near_end and scroll_stalled) or self.in_boss:
            self.in_boss = True
        else:
            self.in_boss = False

        # ---------------- reward shaping ----------------
        R = 0.0

        # (A) NEW PROGRESS: only pay when beating best x (disabled in boss)
        if not self.in_boss:
            new_progress = max(0, x - self.x_best)
            if new_progress > 0:
                R += 2.0 * (new_progress / 5.0)  # big progress pay
                self.x_best = x
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1
        else:
            # In boss: no progress signal (camera is locked)
            self.no_progress_steps += 1  # still track for diagnostics

        # (B) SPEED / MOTION
        if not self.in_boss:
            if dx > 0:
                dx_c = _cap(dx, 0, 12)
                R += 0.4 * dx_c
                self.right_streak += 1
            elif dx == 0:
                R -= 0.6
                self.right_streak = 0
            else:
                R -= 3.0
                self.right_streak = 0

            # streak every 20 forward steps
            if self.right_streak > 0 and self.right_streak % 20 == 0:
                R += 8.0

            # burst bonus: new record step speed
            if dx > self.best_dx and dx > 4:
                R += 5.0 + 0.5 * _cap(dx - self.best_dx, 0, 8)
                self.best_dx = dx
        else:
            # In boss: de-emphasize dx; allow free dodging
            if dx > 0:
                R += 0.1 * _cap(dx, 0, 8)  # small rightward nudge
            # No idle/backtrack penalties in boss

        # (C) LIGHT TIME PRESSURE (always on)
        R -= 0.02

        # (D) JUMP SHAPING
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = {"A", "B", "C"}

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

        if not self.in_boss:
            if is_jump and dx <= 0:
                R -= 2.0
            elif is_jump and dx > 0:
                R += 0.5
        else:
            # In boss: neutral on jumps unless they cause score gain (handled below)
            pass

                # (E) RINGS — collect good, lose bad (balanced)
        if dr > 0:
            R += 1.0 * dr          # +1 per ring gained
            if dr >= 10:           # bonus for big pickups
                R += 10.0
        elif dr < 0:
            R -= 2.0 * abs(dr)     # penalty for losing rings


        # (F) BOSS SCORE REWARD
        if self.in_boss and dscore > 0:
            # Boss hits usually add +1000; scale generously
            R += 0.1 * dscore          # +100 per +1000 score
            if dscore >= 1000:
                R += 150.0            # clear confirmation burst

        # (G) STUCK penalties / early reset (disabled in boss)
        if not self.in_boss:
            if self.no_progress_steps and self.no_progress_steps % 120 == 0:
                R -= 4.0
            if self.no_progress_steps > 360:
                done = True
                truncated = True

        # (H) Terminal events
        if lives < prev_life:
            R -= 40.0
            done = True
            terminated = True
        if x >= screen_end:
            R += 400.0
            done = True
            terminated = True

        # (I) Episode step cap
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
            truncated = True

        # ---------------- logging ----------------
        self.episode_records.append({
            "step": self.steps,
            "action": int(action) if isinstance(action, (int, np.integer)) else -1,
            "is_jump": bool(is_jump),
            "x": int(x),
            "dx": int(dx),
            "dx_avg": float(dx_avg),
            "x_best": int(self.x_best),
            "rings": int(rings),
            "dr": int(dr),
            "score": int(score),
            "dscore": int(dscore),
            "streak": int(self.right_streak),
            "in_boss": bool(self.in_boss),
            "reward": float(round(R, 4)),
        })

        # update previous info for next step
        self.prev_info = info

        # always return shaped reward (ignore env's raw reward)
        return obs, float(R), bool(terminated), bool(truncated), info
