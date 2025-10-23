import retro
import numpy as np
import cv2

class SonicEnv:
    """
    Sonic The Hedgehog environment wrapper for Retro
    Includes preprocessing and step/reward shaping.
    """
    def __init__(self, game="SonicTheHedgehog2-Genesis", state="EmeraldHillZone.Act1", render=False):
        self.env = retro.make(game=game, state=state)
        self.render_mode = render
        self.action_space = self.env.action_space
        self.obs_shape = (84, 84, 1)
        self.done = False
        self.prev_x = 0
        self.prev_rings = 0

    def preprocess(self, frame):
        """Convert RGB frame to grayscale and resize to 84x84."""
        """
        Convert a raw RGB frame from the game into a standardized observation for the agent.

        Steps:
        1. Convert the RGB frame to grayscale → reduces complexity and channels from 3 to 1.
        2. Resize the frame to 84x84 → standard input size for most RL agents.
        3. Add an extra channel dimension → makes the output shape (84, 84, 1),
        which is compatible with convolutional neural networks.

        Args:
            frame (np.ndarray): Original RGB frame from the Retro environment.

        Returns:
            np.ndarray: Preprocessed frame (grayscale, resized, with single channel).
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        return frame[:, :, None]

    def reset(self):
        """Reset environment and return initial observation."""
        obs = self.env.reset()
        self.prev_x = 0
        self.prev_rings = 0
        self.done = False
        return self.preprocess(obs)

    def step(self, action):
        """Take a step in the environment with reward shaping."""
        obs, _, done, info = self.env.step(action)
        self.done = done

        x = info.get("x", self.prev_x)
        rings = info.get("rings", self.prev_rings)
        reward = (x - self.prev_x) + (rings - self.prev_rings)
        self.prev_x = x
        self.prev_rings = rings

        return self.preprocess(obs), reward, done, info

    def close(self):
        self.env.close()

    def rollout(self, episodes=1, random_policy=True, sleep=0.02):
        """Run a quick rollout for testing or observation."""
        import time
        for ep in range(episodes):
            obs = self.reset()
            total_reward = 0
            while not self.done:
                if random_policy:
                    action = self.action_space.sample()
                else:
                    action = self.manual_action(obs)
                obs, reward, done, info = self.step(action)
                total_reward += reward
                if self.render_mode:
                    time.sleep(sleep)
            print(f"[Rollout] Episode {ep+1}/{episodes} | Total reward: {total_reward}")

    def manual_action(self, obs):
        """Optional simple helper: move right + jump every 15 frames."""
        btn = np.zeros(self.action_space.shape, dtype=np.uint8)
        try:
            right_idx = self.env.buttons.index("RIGHT")
            btn[right_idx] = 1
        except Exception:
            pass
        try:
            jump_idx = next(self.env.buttons.index(b) for b in ("B","A","C") if b in self.env.buttons)
            btn[jump_idx] = 1
        except Exception:
            pass
        return btn
