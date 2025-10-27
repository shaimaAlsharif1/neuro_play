import gymnasium as gym
import numpy as np
import os


def _cap(x, lo, hi):
    return max(lo, min(hi, x))


class ResetStateWrapper(gym.Wrapper):
    """
    Reward shaping for Sonic 2 with Boss Mode and balanced ring rewards.

    • Pre-boss: strong incentives for going right/fast, new-best progress,
      streaks, and speed bursts. Mild time pressure; light penalties for idle/left.
    • Rings: +1 per ring gained (+10 burst if ≥10 at once); -2 per ring lost.
      (Balanced magnitudes so rings don't dominate progress.)
    • Boss Mode: activates only near end of level when scroll stalls; disables
      'stuck' penalties; pays on score increases (boss hits).
    • Exposes `in_boss` and `no_progress_steps` for the discretizer.
    """

    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
        self.max_steps = max_steps

        # episode state
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

        # movement trackers
        self.x_best = 0
        self.no_progress_steps = 0
        self.right_streak = 0
        self.best_dx = 0
        self.dx_window = [0.0] * 10

        # boss flag
        self.in_boss = False

        # logging (optional dir only)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

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
        return obs, info

    def step(self, action):
        step = self.env.step(action)
        if len(step) == 5:
            obs, reward_env, terminated, truncated, info = step
            done = bool(terminated or truncated)
        else:
            obs, reward_env, done, info = step
            terminated, truncated = done, False

        # -------- read state --------
        x          = info.get("x", 0)
        lives      = info.get("lives", 3)
        rings      = info.get("rings", 0)
        score      = info.get("score", 0)
        screen_end = info.get("screen_x_end", 10_000)

        if self.prev_info is None:
            self.prev_info = {"x": 0, "lives": lives, "rings": rings, "score": score}

        prev_x     = self.prev_info.get("x", 0)
        prev_life  = self.prev_info.get("lives", lives)
        prev_rings = self.prev_info.get("rings", rings)
        prev_score = self.prev_info.get("score", score)

        dx     = x - prev_x
        dr     = rings - prev_rings
        dscore = score - prev_score

        # smooth dx for logs
        self.dx_window.pop(0)
        self.dx_window.append(float(dx))
        dx_avg = sum(self.dx_window) / len(self.dx_window)

        # -------- boss detection (stricter) --------
        # Enter boss only when near end AND scroll stalled for a while.
        near_end = (screen_end > 0) and (max(self.x_best, x) >= 0.97 * float(screen_end))
        scroll_stalled = (self.no_progress_steps > 240) and (abs(dx) <= 1)  # ~4s
        if not self.in_boss and near_end and scroll_stalled:
            self.in_boss = True
        # once in boss, stay until episode ends (no auto-exit)

        # -------- reward shaping --------
        R = 0.0

        # (A) progress/new-best (disabled in boss)
        if not self.in_boss:
            new_progress = max(0, x - self.x_best)
            if new_progress > 0:
                R += 2.0 * (new_progress / 5.0)
                self.x_best = x
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1
        else:
            self.no_progress_steps += 1  # still track for info

        # (B) movement/speed
        if not self.in_boss:
            if dx > 0:
                R += 0.4 * _cap(dx, 0, 12)
                self.right_streak += 1
            elif dx == 0:
                R -= 0.6
                self.right_streak = 0
            else:
                R -= 3.0
                self.right_streak = 0

            if self.right_streak > 0 and self.right_streak % 20 == 0:
                R += 8.0

            if dx > self.best_dx and dx > 4:
                R += 5.0 + 0.5 * _cap(dx - self.best_dx, 0, 8)
                self.best_dx = dx
        else:
            if dx > 0:
                R += 0.1 * _cap(dx, 0, 8)  # small nudge
            # no idle/backtrack penalty in boss

        # (C) light time pressure (always)
        R -= 0.02

        # (D) jump shaping (mild)
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = {"A", "B", "C"}

        # map discrete index to button array if needed
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
                R -= 1.5
            elif is_jump and dx > 0:
                R += 0.4

        # (E) rings — balanced
        if dr > 0:
            R += 1.0 * dr
            if dr >= 10:
                R += 10.0
        elif dr < 0:
            R -= 2.0 * abs(dr)

        # (F) boss score reward
        if self.in_boss and dscore > 0:
            R += 0.05 * dscore      # +50 per +1000 score
            if dscore >= 1000:
                R += 80.0           # clear hit bonus

        # (G) stuck penalties (disabled in boss)
        if not self.in_boss:
            if self.no_progress_steps and self.no_progress_steps % 120 == 0:
                R -= 3.0
            if self.no_progress_steps > 360:
                done = True
                truncated = True

        # (H) terminal events
        if lives < prev_life:
            R -= 40.0
            done = True
            terminated = True
        if x >= screen_end:
            R += 400.0
            done = True
            terminated = True

        # step cap
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
            truncated = True

        # update prev
        self.prev_info = info

        # return shaped reward
        return obs, float(R), bool(terminated), bool(truncated), info
