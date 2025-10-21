#!/usr/bin/env python3
"""
main.py — robust runner for stable-retro / gym-retro

What’s new vs previous version?
- Always shows a real window via pygame (no dependence on retro's "human" renderer).
- Always records MP4 via imageio if --record-dir is provided (no gym wrappers).
- Works across gym/gymnasium/retro versions (we only need obs frames).
- Helpful logs about where the video is saved.

Usage examples:
  python main.py --game SonicTheHedgehog-Genesis --state GreenHillZone.Act1
  python main.py --game SonicTheHedgehog-Genesis --sonic-helper --record-dir ./videos
  python main.py --game Airstriker-Genesis --episodes 3 --max-steps 3000 --scale 3

Requirements:
  pip install stable-retro pygame imageio imageio-ffmpeg
  (and import ROMs once:  python -m retro.import <path_to_roms>)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- Import retro ---
try:
    import retro  # stable-retro or gym-retro
except Exception:
    print("ERROR: Could not import 'retro'. Install with: pip install stable-retro", file=sys.stderr)
    raise

# --- Optional: gym/gymnasium just for types (not required for our viewer/recorder) ---
try:
    import gymnasium as gym  # type: ignore
except Exception:
    try:
        import gym  # type: ignore
    except Exception:
        gym = None  # Not needed

# --- Pygame (window) ---
try:
    import pygame
except Exception:
    print("ERROR: Could not import 'pygame'. Install with: pip install pygame", file=sys.stderr)
    raise

# --- ImageIO (recording) ---
try:
    import imageio.v3 as iio
except Exception:
    iio = None  # We'll warn if user asks for recording


def make_env(game: str, state: str | None):
    """
    Create a retro environment that can produce RGB frames.
    We DO NOT rely on retro's built-in window; we'll display frames ourselves.
    """
    # Prefer rgb_array so env.render('rgb_array') is also available if needed
    try:
        env = retro.make(game=game, state=state, render_mode="human")
    except TypeError:
        # Older gym-retro: 'render_mode' not supported, still fine: obs is the frame.
        env = retro.make(game=game, state=state)
    return env


def simple_sonic_policy(action_space, buttons):
    """
    A very naive helper policy for Sonic-like games:
    - Always press RIGHT
    - Tap B every ~15 frames
    Works only for MultiBinary action spaces (standard in retro).
    """
    if hasattr(action_space, "n"):
        # Discrete fallback
        def policy(_obs, t):
            return action_space.sample()
        return policy

    btn_array = np.zeros(action_space.shape, dtype=np.uint8)
    try:
        right_idx = buttons.index("RIGHT")
    except ValueError:
        right_idx = None
    # Try Genesis jump buttons in common order
    for name in ("B", "A", "C"):
        try:
            b_idx = buttons.index(name)
            break
        except ValueError:
            b_idx = None

    def policy(_obs, t):
        btn_array[:] = 0
        if right_idx is not None:
            btn_array[right_idx] = 1
        if b_idx is not None and (t % 15 == 0):
            btn_array[b_idx] = 1
        return btn_array.copy()

    return policy


def random_policy(action_space):
    def policy(_obs, _t):
        return action_space.sample()
    return policy


def get_buttons(env):
    try:
        return list(getattr(env, "buttons", []))
    except Exception:
        return []


class PygameViewer:
    def __init__(self, width: int, height: int, scale: int = 2, title: str = "retro"):
        pygame.init()
        self.scale = max(1, int(scale))
        self.size = (width * self.scale, height * self.scale)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self._surf = None

    def draw(self, frame_rgb: np.ndarray, fps_limit: int | None):
        """
        frame_rgb: H x W x 3, dtype uint8
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        h, w, _ = frame_rgb.shape
        # Create / update surface
        self._surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))  # (W,H,3)
        if self.scale != 1:
            self._surf = pygame.transform.scale(self._surf, self.size)

        self.screen.blit(self._surf, (0, 0))
        pygame.display.flip()

        if fps_limit and fps_limit > 0:
            self.clock.tick(fps_limit)

    def close(self):
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass


class VideoWriter:
    """
    Simple MP4 recorder using imageio (requires imageio-ffmpeg).
    """
    def __init__(self, out_dir: str | Path, basename: str = "episode", fps: int = 60):
        if iio is None:
            raise RuntimeError("imageio not available. Install with: pip install imageio imageio-ffmpeg")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Autoincrement filename
        idx = 1
        while True:
            candidate = out_dir / f"{basename}_{idx:03d}.mp4"
            if not candidate.exists():
                self.path = candidate
                break
            idx += 1
        self.fps = fps
        self._writer = iio.imopen(self.path, "w", plugin="pyav", fps=fps, codec="h264", pix_fmt="yuv420p")

    def write(self, frame_rgb: np.ndarray):
        # Ensure uint8 HxWx3
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        self._writer.write(frame_rgb)

    def close(self):
        try:
            self._writer.close()
        except Exception:
            pass


def run_episode(env, policy_fn, max_steps: int, viewer: PygameViewer | None,
                fps_limit: int | None, recorder: VideoWriter | None):
    """
    Runs a single episode:
    - Uses obs frames for both display and recording.
    - Handles gym/gymnasium step signatures.
    """
    # Reset
    try:
        obs, info = env.reset()
    except TypeError:
        obs = env.reset()
        info = {}

    steps = 0
    done = False
    truncated = False

    while not done and not truncated and steps < max_steps:
        action = policy_fn(obs, steps)

        # Step
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated)
        except ValueError:
            obs, reward, done, info = env.step(action)
            truncated = False

        # Get a frame for display/record. In retro, obs is usually the RGB frame already.
        frame = obs
        # Fallback: ask env to render a frame if obs isn't an image
        if not (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] in (1, 3)):
            try:
                frame = env.render(mode="rgb_array")  # type: ignore
            except Exception:
                frame = None

        if frame is not None:
            # Ensure RGB (HxWx3)
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            if viewer is not None:
                viewer.draw(frame, fps_limit=fps_limit)
            if recorder is not None:
                recorder.write(frame)

        steps += 1

    return steps


def main():
    parser = argparse.ArgumentParser(description="Run a game on stable-retro / gym-retro with a real window and recorder.")
    parser.add_argument("--game", required=True, help="Game name, e.g., SonicTheHedgehog-Genesis")
    parser.add_argument("--state", default=None, help="Starting state/level, e.g., GreenHillZone.Act1")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to run")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=60, help="Window FPS cap (0 = uncapped)")
    parser.add_argument("--scale", type=int, default=2, help="Window scale factor (1=original size)")
    parser.add_argument("--record-dir", default=None, help="Folder to save MP4 videos via imageio")
    parser.add_argument("--record-fps", type=int, default=60, help="FPS for recorded video")
    parser.add_argument("--sonic-helper", action="store_true",
                        help="Use a simple 'go-right + jump sometimes' policy (good for Sonic-like games).")
    parser.add_argument("--no-window", action="store_true", help="Disable window (headless), only record if enabled.")
    args = parser.parse_args()

    # Create env
    env = make_env(args.game, args.state)

    # Determine buttons (for helper policy)
    buttons = get_buttons(env)

    # Policy
    if args.sonic_helper:
        policy_fn = simple_sonic_policy(env.action_space, buttons)
        print(f"[info] Using sonic-helper policy. Buttons: {buttons}")
    else:
        policy_fn = random_policy(env.action_space)
        print(f"[info] Using random policy. Buttons: {buttons}")

    # Prepare viewer and recorder
    viewer = None
    recorder = None

    try:
        # Peek at one frame to size the window/recorder correctly
        try:
            obs, _ = env.reset()
        except TypeError:
            obs = env.reset()
        probe = obs
        if not (isinstance(probe, np.ndarray) and probe.ndim == 3 and probe.shape[-1] in (1, 3)):
            try:
                probe = env.render(mode="rgb_array")  # type: ignore
            except Exception:
                probe = None

        if probe is None:
            raise RuntimeError("Could not obtain an RGB frame from the environment. Check your ROM import and retro version.")

        # Ensure RGB
        if probe.ndim == 3 and probe.shape[-1] == 1:
            probe = np.repeat(probe, 3, axis=-1)

        h, w = probe.shape[:2]

        if not args.no_window:
            try:
                viewer = PygameViewer(width=w, height=h, scale=args.scale, title=f"{args.game} {args.state or ''}".strip())
            except Exception as e:
                print(f"[warn] Could not initialize window: {e}. Continuing headless.")

        if args.record_dir:
            if iio is None:
                print("[warn] Recording requested but imageio not available. Install with: pip install imageio imageio-ffmpeg")
            else:
                recorder = VideoWriter(args.record_dir, basename=f"{args.game}", fps=args.record_fps)
                print(f"[info] Recording to: {recorder.path}")

        # Run episodes
        total_steps = 0
        for ep in range(args.episodes):
            print(f"\n=== Episode {ep+1}/{args.episodes} ===")
            # Make sure env is reset at each episode start
            try:
                env.reset()
            except TypeError:
                env.reset()

            steps = run_episode(env, policy_fn, args.max_steps, viewer, args.fps if args.fps > 0 else None, recorder)
            total_steps += steps
            print(f"[done] Episode {ep+1} finished after {steps} step(s).")

        if recorder is not None:
            print(f"[save] Video saved to: {recorder.path}")

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
    finally:
        try:
            if recorder is not None:
                recorder.close()
        except Exception:
            pass
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
