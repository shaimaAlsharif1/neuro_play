# discretizer_sonic.py
import gymnasium as gym
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Converts multi-binary Sonic controls into a discrete action space.
    Example: RIGHT, RIGHT+B (jump), spindash, etc.
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

    def action(self, act: int):
        # IMPORTANT: do NOT override the chosen action
        # (This used to randomly force RIGHT+jump and caused jump-spam.)
        return self._decode_discrete_action[act].copy()


class SonicDiscretizer(Discretizer):
    """
    Small, go-right-first action set.
    You can expand it later (e.g., add LEFT) once forward motion is learned.
    """
    def __init__(self, env):
        combos = [
            [],                    # 0: no-op (rare)
            ['RIGHT'],             # 1: run right
            ['RIGHT', 'B'],        # 2: run + jump
            ['B'],                 # 3: neutral jump (occasional)
            ['DOWN', 'B'],         # 4: spindash charge
            ['RIGHT', 'DOWN', 'B'] # 5: spindash burst while holding RIGHT
        ]
        super().__init__(env, combos)
