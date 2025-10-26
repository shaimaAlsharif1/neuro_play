import gymnasium as gym
import numpy as np
import os
import pandas as pd


class ResetStateWrapper(gym.Wrapper):
    """
    Custom reward shaping + per-step logging for Sonic the Hedgehog 2.
    - Encourages moving forward
    - Penalizes idling and useless jump spam
    - Records every step (action, reward, progress) and writes a CSV per episode
    """

    def __init__(self, env, max_steps=4500, log_dir="logs"):
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps
        self.steps = 0
        self.prev_info = None
        self.jump_counter = 0

        # Logging setup
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
        self.episode_records = []
        return obs, info

    def step(self, action):
        """One environment step with shaped reward and logging."""
        step = self.env.step(action)

        # Support both (obs, reward, terminated, truncated, info) and (obs, reward, done, info)
        if len(step) == 5:
            obs, reward, terminated, truncated, info = step
            done = terminated or truncated
        else:
            obs, reward, done, info = step
            terminated, truncated = done, False

        # -------- Reward shaping --------
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
            custom_reward += 3 * (dx / 100.0)
        elif dx == 0:
            custom_reward -= 1  # stronger penalty for staying still
        elif dx < 0:
            custom_reward -= 50  # penalty for moving backward

        # 2) Small dense reward for proximity to level end
        custom_reward += (x / screen_x_end) * 20

        # 3) Penalty for losing a life (and end episode)
        if lives < prev_lives:
            custom_reward -= 20
            done = True

        # 4) Bonus for finishing the level
        if x >= screen_x_end:
            custom_reward += 200
            done = True

        # -------- Jump control (reduce useless jumping) --------
        buttons = getattr(self.env.unwrapped, "buttons", [])
        jump_buttons = ['A', 'B', 'C']

        # If the action is Discrete, convert it to the button array via the discretizer
        if hasattr(self.env, "action") and isinstance(action, (int, np.integer)):
            try:
                action_array = self.env.action(action)
            except Exception:
                action_array = np.zeros(len(buttons), dtype=np.int8)
        else:
            action_array = action

        # Which buttons are pressed this step?
        pressed_buttons = [buttons[i] for i, val in enumerate(action_array) if val == 1]
        is_jump = any(b in pressed_buttons for b in jump_buttons)

        # Track consecutive jumps
        if is_jump:
            self.jump_counter += 1
        else:
            self.jump_counter = 0

        # Penalize jumping without forward movement
        if is_jump and dx <= 0:
            custom_reward -= 5

        # Penalize jump spam beyond 3 consecutive jumps
        if self.jump_counter > 2:
            custom_reward -= 40 * (self.jump_counter )

        # -------- Episode step cap --------
        self.steps += 1
        if self.steps > self.max_steps:
            done = True

        # -------- Clip reward to a reasonable range --------
        custom_reward = np.clip(custom_reward, -1.0, 1.0)

        # -------- Logging (per step) --------
        action_id = int(action) if isinstance(action, (int, np.integer)) else -1
        record = {
            "step": self.steps,
            "action": action_id,
            "is_jump": bool(is_jump),
            "x": x,
            "reward": round(float(custom_reward), 4),
        }
        self.episode_records.append(record)

        # Force-save logs if max steps reached (even if the env didn't signal done)
        if self.steps >= self.max_steps - 1 and self.episode_records:
            df = pd.DataFrame(self.episode_records)
            self.episode_id += 1
            path = os.path.join(self.log_dir, f"episode_{self.episode_id:03d}.csv")
            df.to_csv(path, index=False)
            print(f"📄 Episode log (forced save) → {path}")
            self.episode_records = []

        # Update previous info snapshot
        self.prev_info = info

        # Return in Gymnasium's 5-tuple format
        terminated = done
        truncated = False
        return obs, custom_reward, terminated, truncated, info
