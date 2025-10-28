# resetstate_sonic.py
import gymnasium as gym
import numpy as np
import os


def _cap(x, lo, hi):
    return max(lo, min(hi, x))


class ResetStateWrapper(gym.Wrapper):
    """
    Aggressive reward shaping for Sonic 2 (crazy mode, but PPO-safe).

    Big ideas:
      • Progress + speed = HUGE rewards (with caps)
      • Rings matter: collect ++, lose ---
      • Smart 'stuck' logic: encourage a jump/backoff when blocked; bonus on breakout
      • Boss mode: pay on score hits + survival; relax idle penalties
      • Safety: per-step reward is clipped, then globally scaled

    Tunables:
      SCALE: global multiplier on shaped reward
      STEP_CLIP: per-step clip to keep PPO stable
    """

    SCALE = 5.0          # global gain (turn up/down overall intensity)
    STEP_CLIP = 150.0    # per-step clamp (keeps PPO from exploding)

    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
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

        # boss phase flag (exposed to discretizer)
        self.in_boss = False

        # optional logging folder
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    # ---------------- core gym API ----------------
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
            obs, _, terminated, truncated, info = step
            done = bool(terminated or truncated)
        else:
            obs, _, done, info = step
            terminated, truncated = done, False

        # ------------- read state -------------
        x          = info.get("x", 0)
        lives      = info.get("lives", 3)
        screen_end = info.get("screen_x_end", 10_000)
        rings      = info.get("rings", 0)
        score      = info.get("score", 0)

        if self.prev_info is None:
            self.prev_info = {"x": 0, "lives": lives, "rings": rings, "score": score}

        prev_x     = self.prev_info.get("x", 0)
        prev_life  = self.prev_info.get("lives", lives)
        prev_rings = self.prev_info.get("rings", rings)
        prev_score = self.prev_info.get("score", score)

        dx     = x - prev_x
        dr     = rings - prev_rings
        dscore = score - prev_score

        # maintain dx rolling avg (for debugging/optional)
        self.dx_window.pop(0)
        self.dx_window.append(float(dx))
        dx_avg = sum(self.dx_window) / len(self.dx_window)

        # ------------- boss detection (strict) -------------
        # Near the end AND scrolling stalled for ~4s (with frame-skip 2)
        near_end = (screen_end > 0) and (max(self.x_best, x) >= 0.97 * float(screen_end))
        scroll_stalled = (self.no_progress_steps > 240) and (abs(dx) <= 1)
        if not self.in_boss and near_end and scroll_stalled:
            self.in_boss = True
        # once in boss, stay in boss until episode ends

        # ------------- reward shaping -------------
        R = 0.0

        # (A) Progress (only pre-boss)
        if not self.in_boss:
            new_progress = max(0, x - self.x_best)
            if new_progress > 0:
                # big: +2 per 5px (capped via clip at end)
                R += 2.0 * (new_progress / 5.0)
                self.x_best = x
                # breaking out of stuck?
                if self.no_progress_steps >= 60:
                    R += 15.0  # breakout bonus
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1
        else:
            self.no_progress_steps += 1

        # (B) Speed / motion
        if not self.in_boss:
            if dx > 0:
                R += 0.8 * _cap(dx, 0, 14)   # big forward speed
                self.right_streak += 1
            elif dx == 0:
                R -= 1.2                     # idle hurts
                self.right_streak = 0
            else:
                R -= 5.0                     # backtrack hurts a lot
                self.right_streak = 0

            # forward streaks: every ~20 forward frames => bonus
            if self.right_streak > 0 and self.right_streak % 20 == 0:
                R += 12.0

            # burst: new personal best step speed
            if dx > self.best_dx and dx > 5:
                R += 10.0 + 0.6 * _cap(dx - self.best_dx, 0, 10)
                self.best_dx = dx
        else:
            # tiny nudge to keep pressing right, but allow dodging freely
            if dx > 0:
                R += 0.2 * _cap(dx, 0, 10)

        # (C) Time pressure (always)
        R -= 0.05

        # (D) Action-level jump shaping & stuck escape
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = {"A", "B", "C"}

        # map discrete index -> button bitfield if needed
        if hasattr(self.env, "action") and isinstance(action, (int, np.integer)):
            try:
                action_array = self.env.action(action)
            except Exception:
                action_array = np.zeros(len(buttons), dtype=np.int8)
        else:
            action_array = action

        pressed = [buttons[i] for i, v in enumerate(action_array) if v == 1]
        is_jump = any(b in jump_buttons for b in pressed)
        self.jump_counter = self.jump_counter + 1 if is_jump else 0

        if not self.in_boss:
            if self.no_progress_steps >= 60 and is_jump:
                R += 6.0                      # try a jump when stuck
            if is_jump and dx <= 0:
                R -= 1.0                      # random jump in place
            elif is_jump and dx > 0:
                R += 0.6                      # helpful hop while moving

        # (E) Rings (amped but not insane)
        if dr > 0:
            R += 3.0 * dr                     # +3 per ring
            if dr >= 10:
                R += 30.0                     # big pickup burst
        elif dr < 0:
            loss = min(abs(dr), 20)
            R -= 6.0 * loss                   # strong penalty per ring lost

        # (F) Boss: pay on score (hits) + survival
        if self.in_boss:
            if dscore > 0:
                R += 0.15 * dscore            # +150 per +1000 score
                if dscore >= 1000:
                    R += 200.0                # clear hit confirmation
            # small survival drip per second in boss
            if (self.steps % 60) == 0:
                R += 5.0

        # (G) Stuck management (pre-boss)
        if not self.in_boss:
            if self.no_progress_steps and self.no_progress_steps % 120 == 0:
                R -= 6.0
            if self.no_progress_steps >= 90 and dx < -1:
                # allow small back-off to build momentum
                R += 2.0

        # (H) Terminals
        if lives < prev_life:
            R -= 80.0
            done = True
            terminated = True
        if x >= screen_end:
            R += 600.0
            done = True
            terminated = True

        # step cap
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
            truncated = True

        # update prev
        self.prev_info = info

        # safety: clip per-step, then apply global scale
        R = _cap(R, -self.STEP_CLIP, self.STEP_CLIP) * self.SCALE

        return obs, float(R), bool(terminated), bool(truncated), info
