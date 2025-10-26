# discretizer_sonic.py
import gymnasium as gym
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Converts multi-binary Sonic controls into a discrete action space.
    Example: RIGHT, RIGHT+A, LEFT, JUMP, etc.
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []

        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SonicDiscretizer(Discretizer):
    """
    Predefined Sonic-specific discrete actions.
    """
    def __init__(self, env):
        combos = [
            [],                 # 0: No-op
            ['RIGHT'],          # 1
            ['RIGHT','A'],      # 2
            ['RIGHT','B'],      # 3
            ['LEFT'],           # 4
            ['LEFT','A'],       # 5
            ['DOWN'],           # 6
            ['UP'],             # 7
            ['A'],              # 8
            ['B'],              # 9
        ]
        super().__init__(env, combos)
