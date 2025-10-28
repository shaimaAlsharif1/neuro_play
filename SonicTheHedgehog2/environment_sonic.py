"""
Complete environment builder for Sonic 2 (Genesis)
Includes preprocessing, action discretization, reward shaping, wrappers, and optional video recording.
"""

import os
import retro
from gymnasium.wrappers import RecordVideo
from config_sonic import (
    GAME_ID,
    STATE,
    FRAME_SKIP,
    IMG_SIZE,
    MAX_EPISODE_STEPS,
)
from discretizer_sonic import SonicDiscretizer
from resetstate_sonic import ResetStateWrapper
from utils import GrayResizeWrapper, NormalizeObs, SkipFrame

class SonicEnv:
    """
    Base Sonic environment wrapper.
    Handles reset, start-button press, and raw retro env creation.
    """

    def __init__(self, render=False):
        self.render = render
        self.env = retro.make(
            game=GAME_ID,
            state=STATE,
            use_restricted_actions=retro.Actions.ALL,
            render_mode="rgb_array"
        )

        try:
            self.buttons = self.env.unwrapped.buttons
        except AttributeError:
            self.buttons = getattr(getattr(self.env, "data", None), "buttons", [
                "B","C","A","START","UP","DOWN","LEFT","RIGHT","Z","Y","X","MODE"
            ])

        # START button index
        self.start_idx = self.buttons.index("START")

    def press_start(self, frames=20):
        """Press START button to skip title screen."""
        import numpy as np
        action = np.zeros(self.env.action_space.shape, dtype=np.int8)
        for _ in range(frames):
            action[:] = 0
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
        """Reset environment and skip intro/title."""
        out = self.env.reset()
        self.press_start()
        return out

    def close(self):
        """Close retro environment."""
        self.env.close()


def make_env(render=True, record_video=False, video_dir="videos", episode_trigger=lambda e: True):
    """
    Builds the full Sonic pipeline and ensures we leave the title screen once.
    """
    # --- Create and clear title screen ---
    se = SonicEnv(render=render)
    base_env = se.env
    try:
        base_env.reset()
        se.press_start(frames=20)
    except Exception as e:
        print("[warn] auto-press START failed:", e)

    # ---- Discretize actions (reduced combos) ----
    env = SonicDiscretizer(base_env)

    # ---- Reward shaping & step limit ----
    env = ResetStateWrapper(env, max_steps=MAX_EPISODE_STEPS)

    # ---- Frame skipping ----
    env = SkipFrame(env, skip=FRAME_SKIP)

    # ---- Resize + grayscale ----
    env = GrayResizeWrapper(env, width=IMG_SIZE, height=IMG_SIZE, keep_dim=False)

    # ---- Normalize to [0,1] ----
    env = NormalizeObs(env)

    # ---- Optional video recording ----
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=episode_trigger,
            name_prefix="sonic_ep"
        )

    return env
