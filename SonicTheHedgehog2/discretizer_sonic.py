import gymnasium as gym
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Converts MultiBinary controller into a compact Discrete action space.
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []

        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for b in combo:
                arr[buttons.index(b)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act: int):
        # Return exactly the combo specified for this discrete index.
        return self._decode_discrete_action[act].copy()

class SonicDiscretizer(Discretizer):
    """
    Compact 4-action Sonic controller with working spindash.
      0: ['RIGHT']          - run / burst
      1: ['RIGHT','B']      - run + jump
      2: ['DOWN','B']       - spindash charge
      3: ['DOWN']           - crouch / stay charging
    """
    def __init__(self, env):
        combos = [
            ['RIGHT'],
            ['RIGHT', 'B'],
            ['DOWN', 'B'],
            ['DOWN'],
        ]
        super().__init__(env, combos)
