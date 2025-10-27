import gymnasium as gym
import numpy as np
from config_sonic import DISCRETE_ACTIONS


class Discretizer(gym.ActionWrapper):
    """
    Converts multi-binary Sega buttons into a single Discrete action.

    Key fix:
      • When not in boss mode, any LEFT-involving combo is REMAPPED to a RIGHT
        equivalent (instead of becoming a no-op). This prevents the policy from
        sampling many 'dead' actions that keep Sonic idle.

    During boss mode (env.unwrapped.in_boss == True), original LEFT combos are
    kept so the agent can dodge.
    """

    def __init__(self, env, block_left_when_not_boss=True):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.buttons = env.unwrapped.buttons  # e.g. [..., 'LEFT','RIGHT', ...]
        self.button_index = {b: i for i, b in enumerate(self.buttons)}

        # store the discrete -> combo mapping from config
        self.actions = [tuple(a) for a in DISCRETE_ACTIONS]
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.block_left_when_not_boss = block_left_when_not_boss

        # cache: which discrete actions include 'LEFT'?
        self._is_left_action = [("LEFT" in combo) for combo in self.actions]

    def _remap_left_to_right(self, combo):
        """Mirror LEFT->RIGHT while preserving other buttons."""
        combo = list(combo)
        has_right = ("RIGHT" in combo)
        remapped = []
        for b in combo:
            if b == "LEFT":
                if not has_right:
                    remapped.append("RIGHT")
            else:
                remapped.append(b)
        return tuple(remapped)

    def action(self, a: int):
        if a < 0 or a >= len(self.actions):
            a = 0
        combo = self.actions[a]

        # read boss flag if available
        try:
            in_boss = bool(self.env.unwrapped.in_boss)
        except Exception:
            in_boss = False

        # if not boss and blocking-left is enabled, REMAP left→right
        if self.block_left_when_not_boss and not in_boss and self._is_left_action[a]:
            combo = self._remap_left_to_right(combo)

        # build MultiBinary button array
        arr = np.zeros(len(self.buttons), dtype=np.int8)
        for b in combo:
            idx = self.button_index.get(b, None)
            if idx is not None:
                arr[idx] = 1
        return arr

    def reverse_action(self, buttons):
        # not used in PPO
        return 0


class SonicDiscretizer(Discretizer):
    """Convenience alias."""
    def __init__(self, env, block_left_when_not_boss=True):
        super().__init__(env, block_left_when_not_boss=block_left_when_not_boss)
