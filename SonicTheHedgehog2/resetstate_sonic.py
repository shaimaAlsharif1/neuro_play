import gymnasium as gym
import numpy as np
from collections import deque

class ResetStateWrapper(gym.Wrapper):
    """
    Reward shaping wrapper for Sonic 2.

    This wrapper implements the specialized reward system to:
    1. Strongly reward forward progress (increase in screen_x).
    2. Penalize and terminate the episode if the agent gets stuck (no screen_x progress).
    3. Lightly penalize jump actions to reduce excessive jumping.
    4. Terminate the episode if lives are lost or max steps are reached.
    """

    # --- Reward & Penalty Constants (Key Tuning Parameters) ---
    PROGRESS_REWARD_SCALE = 10.0    # Reward per screen_x unit of progress (Increased to 10.0 as requested)
    JUMP_PENALTY = -0.05            # Penalty for taking a jump action (to curb excessive jumping)
    STUCK_TIMEOUT = 500             # Max frames without screen_x progress before termination (to address getting stuck)
    STUCK_TERMINATION_PENALTY = -3.0 # Significant penalty for getting stuck
    LIFE_LOSS_PENALTY = -5.0        # Significant penalty for losing a life
    STEP_PENALTY = -0.00005         # Minor penalty per step (encourages speed/efficiency)

    def __init__(self, env, max_steps):
        super().__init__(env)
        self._max_steps = max_steps

        # The wrapped environment must be the Discretizer to get action button mapping
        if hasattr(self.env, '_decode_discrete_action'):
            # This map holds the actual button combinations for each discrete action index.
            self._action_map = self.env._decode_discrete_action
            self._buttons = self.env.unwrapped.buttons
            # Find indices for the jump buttons, typically 'A', 'B', or 'C'
            self.jump_button_indices = [
                self._buttons.index(btn) for btn in ['A', 'B', 'C']
                if btn in self._buttons
            ]
        else:
            print("[WARN] ResetStateWrapper expects a Discretizer wrapper below it for detailed action analysis.")
            self.jump_button_indices = []
            self._action_map = None

        self.reset_internal_state()

    def reset_internal_state(self):
        """Resets all internal episode tracking variables."""
        self._steps = 0
        self._max_x = -np.inf         # Max horizontal position reached
        self._stuck_steps = 0         # Steps since _max_x last increased
        self._prev_lives = None       # Tracks lives for life loss penalty

    def reset(self, **kwargs):
        self.reset_internal_state()
        # Reset the environment and get the initial observation/info
        obs, info = self.env.reset(**kwargs)
        # Initialize max_x with the starting position
        self._max_x = info.get("screen_x", info.get("x", 0))
        self._prev_lives = info.get("lives", 3)
        return obs, info

    def step(self, action):
        # action is the discrete index (0-N) from the SonicDiscretizer
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        # --- 1. Basic Status & Position ---
        current_x = info.get("screen_x", info.get("x", 0))
        current_lives = info.get("lives", self._prev_lives)

        new_shaped_reward = 0.0

        # --- 2. Reward Shaping Logic ---

        # A. Forward Progress Reward (Core Reward)
        if current_x > self._max_x:
            progress_reward = (current_x - self._max_x) * self.PROGRESS_REWARD_SCALE
            new_shaped_reward += progress_reward
            self._max_x = current_x
            self._stuck_steps = 0
        else:
            self._stuck_steps += 1

        # B. Jumping Penalty (Addressing "jumping a lot")
        if self._action_map is not None:
            # Get the boolean array of buttons pressed for this discrete action
            button_array = self._action_map[action]
            # If the action contains an 'A' or 'B' (Jump/Spin-Dash), apply a penalty
            is_jump_action = any(button_array[idx] for idx in self.jump_button_indices)

            if is_jump_action:
                new_shaped_reward += self.JUMP_PENALTY

        # C. Step Penalty (Encourages faster solutions)
        new_shaped_reward += self.STEP_PENALTY

        # D. Game Over / Lives Lost Penalty
        if current_lives < self._prev_lives:
            # Apply a large penalty and terminate the episode immediately
            terminated = True
            new_shaped_reward += self.LIFE_LOSS_PENALTY
        self._prev_lives = current_lives

        # --- 3. Termination Check ---

        # E. Stuck Penalty (Addressing "cannot pass the wall")
        if self._stuck_steps >= self.STUCK_TIMEOUT:
            # Apply a large penalty for failure to progress and terminate
            terminated = True
            new_shaped_reward += self.STUCK_TERMINATION_PENALTY
            info['stuck_timeout'] = True

        # F. Max Steps Timeout
        if self._steps >= self._max_steps:
            truncated = True
            info['max_steps_timeout'] = True

        # Combine the original environment reward (e.g., rings, score) with the shaped reward
        final_reward = reward + new_shaped_reward

        return obs, final_reward, terminated, truncated, info
