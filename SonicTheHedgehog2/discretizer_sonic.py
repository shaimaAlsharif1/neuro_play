import gymnasium as gym
import numpy as np
from config_sonic import DISCRETE_ACTIONS


class Discretizer(gym.ActionWrapper):
    """
    Converts multi-binary Sonic controls (Retro) into a discrete action space.
    Reads combos from config_sonic.DISCRETE_ACTIONS.

    Adds smart gating for LEFT:
      • During normal play: LEFT combos can be blocked to keep forward bias
      • During boss fights (wrapper exposes env.unwrapped.in_boss): LEFT allowed
    """

    def __init__(self, env, block_left_when_not_boss=True):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.buttons = env.unwrapped.buttons  # e.g., ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self.button_index = {b: i for i, b in enumerate(self.buttons)}

        # store original action combos (from config)
        self.actions = [tuple(combo) for combo in DISCRETE_ACTIONS]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # behavior flags
        self.block_left_when_not_boss = block_left_when_not_boss

        # precompute which actions are LEFT-involving for quick gating
        self._is_left_action = []
        for combo in self.actions:
            has_left = ('LEFT' in combo)
            self._is_left_action.append(has_left)

    def action(self, a: int):
        """
        Map discrete index -> MultiBinary buttons array, with optional gating:
        If we're not in boss mode and left-blocking is enabled, LEFT actions are turned into no-ops.
        """
        # safety on index
        if a < 0 or a >= len(self.actions):
            a = 0

        combo = list(self.actions[a])

        # Boss awareness from wrapper (may not exist in some envs)
        try:
            in_boss = bool(self.env.unwrapped.in_boss)
        except Exception:
            in_boss = False

        # If not in boss and we want to bias forward, block LEFT actions
        if self.block_left_when_not_boss and not in_boss and self._is_left_action[a]:
            combo = []  # no-op instead of moving left

        # Build button array
        arr = np.zeros(len(self.buttons), dtype=np.int8)
        for b in combo:
            idx = self.button_index.get(b, None)
            if idx is not None:
                arr[idx] = 1

        return arr.copy()

    def reverse_action(self, buttons):
        # Optional: map MultiBinary back to discrete index (not needed by PPO)
        return 0


class SonicDiscretizer(Discretizer):
    """
    Alias wrapper for convenience; uses DISCRETE_ACTIONS from config.
    block_left_when_not_boss=True keeps LEFT disabled during normal play, but it
    becomes available automatically in boss rooms (env.unwrapped.in_boss).
    """
    def __init__(self, env, block_left_when_not_boss=True):
        super().__init__(env, block_left_when_not_boss=block_left_when_not_boss)
