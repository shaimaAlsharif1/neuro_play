
import retro
import numpy as np
from PIL import Image
import os
import cv2

class SonicEnv:
    """
    Wrapper for the Sonic The Hedgehog environment using Gym-Retro.
    Handles setup, reset, stepping, and optional preprocessing.
    """

    def __init__(self,
                 game_id="SonicTheHedgehog2-Genesis",
                 state_id="EmeraldHillZone.Act1",
                 mask_dir="mask",
                 render=False):
        self.game_id = game_id
        self.state_id = state_id
        self.render = render
        self.mask_dir = mask_dir

        os.makedirs(mask_dir, exist_ok=True)

        # Create environment
        self.env = retro.make(
            game=self.game_id,
            state=self.state_id,
            use_restricted_actions=retro.Actions.ALL,
            render_mode="human" if render else None
        )

        # Try to get the START button index
        try:
            self.buttons = self.env.unwrapped.buttons
        except AttributeError:
            self.buttons = getattr(getattr(self.env, "data", None), "buttons",
                                   ['B','C','A','START','UP','DOWN','LEFT','RIGHT','Z','Y','X','MODE'])
        self.start_idx = self.buttons.index('START')

    def press_start(self, frames=12):
        """Press the START button for a few frames to skip title/menu."""
        action = np.zeros(self.env.action_space.shape, dtype=np.int8)
        for _ in range(frames):
            action[self.start_idx] = 1
            step = self.env.step(action)
            if len(step) == 5:
                _, _, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                _, _, done, _ = step
            if done:
                self.env.reset()

    def reset(self):
        """Reset the environment and press START."""
        obs = self.env.reset()
        self.press_start()
        return obs

    def step(self, action=None):
        """Step through the environment (default = no-op)."""
        if action is None:
            action = np.zeros(self.env.action_space.shape, dtype=np.int8)

        step = self.env.step(action)
        if len(step) == 5:
            obs, reward, terminated, truncated, info = step
            done = terminated or truncated
        else:
            obs, reward, done, info = step
        return obs, reward, done, info

    def make_mask(self, frame):
        """Simple color-based mask example for Sonic and rings."""
        rgb = frame[:, :, :3]
        ring_mask = ((rgb[:,:,0] > 200) & (rgb[:,:,1] > 180) & (rgb[:,:,2] < 100))
        player_mask = ((rgb[:,:,2] > 100) & (rgb[:,:,0] < 120))
        return ((ring_mask | player_mask).astype(np.uint8) * 255)

        #goal is training a model (like DQN, PPO, etc.), you often want grayscale and smaller frames.
        # You can easily add a helper method inside theclass
    def preprocess(self, frame):
        """Convert RGB frame to grayscale 84x84."""

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized[:, :, None]  # shape (84, 84, 1)

    def run(self, steps=1000):
        """Example loop to run the environment and save masks."""
        obs = self.reset()
        for t in range(steps):
            obs, reward, done, info = self.step()
            mask = self.make_mask(obs)
            Image.fromarray(mask).save(f"{self.mask_dir}/mask_{t:06d}.png")
            if done:
                obs = self.reset()
        self.env.close()

    def close(self):
        """Close the environment."""
        self.env.close()
