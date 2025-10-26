#!/usr/bin/env python3
"""
main.py — demo runner for stable-retro / gym-retro (no external agent needed)

- Shows a real window via pygame
- Plays with either a random policy (default) or a simple 'go-right + jump sometimes' helper
- Records MP4 via imageio-ffmpeg if --record-dir is provided
- Compatible across gym/gymnasium/retro API differences
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from environment_sonic import SonicEnv

# --- retro (stable-retro / gym-retro) ---
try:
    import retro
except Exception:
    print("ERROR: Could not import 'retro'. Install with: pip install stable-retro", file=sys.stderr)
    raise

# --- pygame (window) ---
try:
    import pygame
except Exception:
    print("ERROR: Could not import 'pygame'. Install with: pip install pygame", file=sys.stderr)
    raise

# --- imageio (FFmpeg for MP4) ---
try:
    import imageio  # v2 API with imageio-ffmpeg
except Exception:
    print("ERROR: Could not import 'imageio'. Install with: pip install imageio imageio-ffmpeg", file=sys.stderr)
    raise


# ---------- helpers ----------
def make_env(game_id="SonicTheHedgehog2-Genesis",
             state_id="EmeraldHillZone.Act1",
             mask_dir="mask",
             render=False):
    """
    Factory function to create and return a SonicEnv instance.
    """
    env = SonicEnv(
        game_id=game_id,
        state_id=state_id,
        mask_dir=mask_dir,
        render=render
    )
    return env

def get_buttons(env):
    try:
        return list(getattr(env, "buttons", []))
    except Exception:
        return []


def simple_sonic_policy(action_space, buttons):
    """
    Very naive Sonic helper:
    - Hold RIGHT
    - Tap B (or A/C) every ~15 frames
    """
    if hasattr(action_space, "n"):
        # Discrete fallback
        def policy(_obs, t):
            return action_space.sample()
        return policy

    btn = np.zeros(action_space.shape, dtype=np.uint8)

    try:
        right_idx = buttons.index("RIGHT")
    except ValueError:
        right_idx = None

    jump_idx = None
    for name in ("B", "A", "C"):
        try:
            jump_idx = buttons.index(name)
            break
        except ValueError:
            pass

    def policy(_obs, t):
        btn[:] = 0
        if right_idx is not None:
            btn[right_idx] = 1
        if jump_idx is not None and (t % 15 == 0):
            btn[jump_idx] = 1
        return btn.copy()

    return policy


def random_policy(action_space):
    def policy(_obs, _t):
        return action_space.sample()
    return policy


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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        # frame: HxWx3
        self._surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
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
    """MP4 recorder using imageio-ffmpeg (no PyAV)."""
    def __init__(self, out_dir: str | Path, basename: str = "episode", fps: int = 60):
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
        self._writer = imageio.get_writer(
            self.path.as_posix(),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
        )

    def write(self, frame_rgb: np.ndarray):
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        self._writer.append_data(frame_rgb)

    def close(self):
        try:
            self._writer.close()
        except Exception:
            pass


def ensure_rgb_frame(env, obs):
    """Return HxWx3 uint8 frame either from obs or env.render()."""
    frame = obs
    if not (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] in (1, 3)):
        try:
            frame = env.render(mode="rgb_array")  # older gym signature
        except Exception:
            frame = None
    if frame is not None and frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def run_episode(env, policy_fn, max_steps: int, viewer: PygameViewer | None,
                fps_limit: int | None, recorder: VideoWriter | None):
    """Play one episode with the chosen built-in policy and optionally record."""
    try:
        obs, _ = env.reset()
    except TypeError:
        obs = env.reset()

    steps = 0
    done = False
    truncated = False

    while not done and not truncated and steps < max_steps:

        rand_ = np.random.random(10)
        if rand_< 2:
            action = random_policy(env.action_space)
        action = policy_fn(obs, steps)

        # Step (gymnasium vs gym)
        try:
            obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated)
        except ValueError:
            obs, reward, done, _info = env.step(action)
            truncated = False

        frame = ensure_rgb_frame(env, obs)
        if frame is not None:
            if viewer is not None:
                viewer.draw(frame, fps_limit=fps_limit)
            if recorder is not None:
                recorder.write(frame)

        steps += 1

    return steps






def main():
    parser = argparse.ArgumentParser(description="Demo: run a retro game with a built-in policy (no external agent).")
    parser.add_argument("--game", required=True, help="e.g., SonicTheHedgehog-Genesis or SonicTheHedgehog2-Genesis")
    parser.add_argument("--state", default=None, help="e.g., GreenHillZone.Act1 or EmeraldHillZone.Act1")
    parser.add_argument("--episodes", type=int, default=1, help="How many episodes to play")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--fps", type=int, default=60, help="Window FPS cap (0 = uncapped)")
    parser.add_argument("--scale", type=int, default=3, help="Window scale factor (1=original size)")
    parser.add_argument("--record-dir", default=None, help="Folder to save MP4 videos")
    parser.add_argument("--record-fps", type=int, default=60, help="FPS for recording")
    parser.add_argument("--sonic-helper", action="store_true",
                        help="Use simple Sonic helper policy (hold RIGHT, jump sometimes). Otherwise random.")
    parser.add_argument("--no-window", action="store_true", help="Headless mode (only record if --record-dir given).")
    args = parser.parse_args()

    # Create env
    env = make_env(args.game, args.state)

    # Choose policy (no external agent needed)
    buttons = get_buttons(env)
    policy_fn = simple_sonic_policy(env.action_space, buttons) if args.sonic_helper else random_policy(env.action_space)
    print(f"[info] Policy: {'sonic-helper' if args.sonic_helper else 'random'} | Buttons: {buttons}")

    viewer = None
    recorder = None

    try:
        # Probe a frame for sizing
        try:
            obs, _ = env.reset()
        except TypeError:
            obs = env.reset()
        probe = ensure_rgb_frame(env, obs)
        if probe is None:
            raise RuntimeError("Could not obtain an RGB frame from the environment. Check your ROM import and retro version.")

        h, w = probe.shape[:2]

        if not args.no_window:
            try:
                viewer = PygameViewer(width=w, height=h, scale=args.scale,
                                      title=f"{args.game} {args.state or ''}".strip())
            except Exception as e:
                print(f"[warn] Could not init window: {e}. Running headless.")

        if args.record_dir:
            try:
                recorder = VideoWriter(args.record_dir, basename=f"{args.game}", fps=args.record_fps)
                print(f"[info] Recording to: {recorder.path}")
            except Exception as e:
                print(f"[warn] Could not start recorder: {e}")

        # Play episodes
        for ep in range(args.episodes):
            print(f"\n=== Episode {ep+1}/{args.episodes} ===")
            try:
                env.reset()
            except TypeError:
                env.reset()
            steps = run_episode(env, policy_fn, args.max_steps, viewer, args.fps if args.fps > 0 else None, recorder)
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
