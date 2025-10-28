# resetstate_sonic.py
import gymnasium as gym
import numpy as np
from collections import deque

class ResetStateWrapper(gym.Wrapper):
    """
    Sonic-2 reward shaping.

    Highlights:
      - Uses info['screen_x'] / info['screen_x_end'] for robust progress.
      - Contest-style progress (primary), modest dx/explore/flow (secondary).
      - No global jump penalty; start cooldown + jump-when-stuck nudge.
      - Launch bonus via |dy|.
      - When stuck at a wall while holding RIGHT, prefer SPINDASH:
          charge (DOWN+B) → burst (RIGHT+DOWN+B),
          with charge-hold reward, burst bonus, and post-burst speed multiplier.
      - Early termination on stagnation.
    """

    # ====== Primary ======
    K_CONTEST = 9000.0  # Δ(screen_x/end_x)

    # ====== Secondary ======
    K_DX        = 3.0   # per-pixel forward
    K_EXPLORE   = 5.0   # new-farthest pixels
    FLOW_WINDOW = 30
    FLOW_SCALE  = 1.0

    # ====== Finish / lives / rings / score ======
    FINISH_BONUS      = 500.0
    TIME_FINISH_BONUS = 1000.0
    LIFE_GAIN         = 1000.0
    LIFE_LOSS         = -100.0
    RING_GAIN         = 1000.0
    RING_LOSS         = -10.0
    RING_DEFICIT      = -5.0
    SCORE_DELTA       = 10.0

    # ====== Anti-stall / heuristics ======
    IDLE_PENALTY            = -0.2     # light so exploration isn't drowned
    BACKWARD_PENALTY_PER_PX = -2.0
    JUMP_TOL_COUNT          = 2
    JUMP_TOL_PERIOD         = 10
    START_JUMP_COOLDOWN     = 90       # allow earlier experiments
    STUCK_JUMP_NUDGE_AT     = 90

    # ====== Stagnation cutoff ======
    DEFAULT_STAGNATION_CUTOFF = 240    # ~4s w/out new best_x
    DEFAULT_MAX_STEPS         = 4500

    # ====== Spindash & wall-behavior ======
    SPINDASH_WINDOW      = 30          # frames to go from charge to burst
    SPINDASH_BONUS       = 20.0        # big positive for correct sequence
    CHARGE_HOLD_REWARD   = 0.6         # per-frame while holding DOWN+B (stuck)
    WALL_JUMP_PENALTY    = -3.0        # discourage pogo at a wall
    BURST_TIMER_FRAMES   = 10          # reward speed for a short window
    BURST_SPEED_SCALE    = 2.0         # multiplier for dx during burst window

    def __init__(self, env, max_steps=None, stagnation_cutoff=None):
        super().__init__(env)
        self.MAX_STEPS = int(max_steps) if max_steps is not None else self.DEFAULT_MAX_STEPS
        self.STAGNATION_CUTOFF = (
            int(stagnation_cutoff) if stagnation_cutoff is not None else self.DEFAULT_STAGNATION_CUTOFF
        )

        # Episode state
        self.steps = 0
        self.frame_counter = 0

        # Progress tracking
        self.prev_info = None
        self.prev_sx = 0
        self.best_x = 0
        self.prev_progress = 0.0

        # Info counters
        self.prev_rings = 0
        self.prev_score = 0
        self.prev_lives = 3
        self._prev_sy = 0

        # Jump control
        self.jump_history = deque()

        # Stagnation & flow
        self.stuck_steps = 0
        self.flow_dx = deque(maxlen=self.FLOW_WINDOW)

        # Spindash helpers
        self.spindash_arming = 0
        self.charge_hold = 0
        self.burst_timer = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self.steps = 0
        self.frame_counter = 0

        self.prev_info = info
        self.prev_sx = int(info.get("screen_x", info.get("x", 0)))
        self.best_x = self.prev_sx
        self.prev_progress = 0.0

        self.prev_rings = int(info.get("rings", 0))
        self.prev_score = int(info.get("score", 0))
        self.prev_lives = int(info.get("lives", 3))
        self._prev_sy = int(info.get("y", 0))

        self.jump_history.clear()
        self.stuck_steps = 0
        self.flow_dx.clear()

        self.spindash_arming = 0
        self.charge_hold = 0
        self.burst_timer = 0

        return obs, info

    def step(self, action):
        # Env step (supports 5-tuple or 4-tuple)
        result = self.env.step(action)
        if len(result) == 5:
            obs, base_rew, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, base_rew, done, info = result
            terminated, truncated = done, False

        self.frame_counter += 1

        # Extract info safely
        sx = int(info.get("screen_x", info.get("x", 0)))
        end_x = max(int(info.get("screen_x_end", 0)), 10_000)
        rings = int(info.get("rings", self.prev_rings))
        score = int(info.get("score", self.prev_score))
        lives = int(info.get("lives", self.prev_lives))
        sy = int(info.get("y", self._prev_sy))

        dx = sx - self.prev_sx
        dy = sy - self._prev_sy
        custom = 0.0

        # Primary: contest-style potential
        progress = sx / float(end_x)
        custom += self.K_CONTEST * (progress - self.prev_progress)
        self.prev_progress = progress

        # Secondary: dx + explore + flow
        custom += self.K_DX * dx
        if sx > self.best_x:
            custom += self.K_EXPLORE * (sx - self.best_x)
            self.best_x = sx
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1

        self.flow_dx.append(max(dx, 0))
        avg_dx = sum(self.flow_dx) / max(len(self.flow_dx), 1)
        custom += self.FLOW_SCALE * avg_dx

        # Finish / life
        if sx >= end_x:
            t = np.clip(self.frame_counter / 18000.0, 0.0, 1.0)
            custom += self.FINISH_BONUS + (1.0 - t) * self.TIME_FINISH_BONUS
            done = True
            terminated = True

        if lives > self.prev_lives:
            custom += self.LIFE_GAIN * (lives - self.prev_lives)
        elif lives < self.prev_lives:
            custom += self.LIFE_LOSS * (self.prev_lives - lives)
            done = True
            terminated = True

        # Rings / score
        ring_diff = rings - self.prev_rings
        if ring_diff > 0:
            custom += self.RING_GAIN * ring_diff
        elif ring_diff < 0:
            custom += self.RING_LOSS
        if rings == 0:
            custom += self.RING_DEFICIT
        custom += self.SCORE_DELTA * (score - self.prev_score)

        # Idle / backward shaping
        if dx == 0:
            custom += self.IDLE_PENALTY
        elif dx < 0:
            custom += self.BACKWARD_PENALTY_PER_PX * (-dx)

        # Decode current pressed buttons (robust to action indexing)
        buttons = getattr(self.env.unwrapped, "buttons", [])
        if hasattr(self.env, "action") and isinstance(action, (int, np.integer)):
            try:
                act_arr = self.env.action(action)  # Discrete -> MultiBinary
            except Exception:
                act_arr = np.zeros(len(buttons), dtype=np.int8)
        else:
            act_arr = action

        pressed = {buttons[i] for i, v in enumerate(act_arr) if v == 1} if buttons else set()
        jumped = bool({'A', 'B', 'C'} & pressed)
        right = 'RIGHT' in pressed
        down_b = ('DOWN' in pressed) and ('B' in pressed)
        burst = right and down_b  # RIGHT+DOWN+B

        # Early RIGHT bias (~4s) so he actually hits obstacles/springs
        if right and self.frame_counter < 240:
            custom += 2.0

        # Start-of-episode jump cooldown & jump-when-stuck nudge (no global penalty)
        if self.frame_counter < self.START_JUMP_COOLDOWN and jumped:
            custom += -5.0
        if self.stuck_steps > self.STUCK_JUMP_NUDGE_AT and jumped:
            custom += 2.0

        # Record jumps within tolerance window (no penalty now)
        if jumped:
            self.jump_history.append(self.frame_counter)
            while self.jump_history and self.jump_history[0] + self.JUMP_TOL_PERIOD <= self.frame_counter:
                self.jump_history.popleft()

        # Launch/spring bonus: significant vertical change while not going backward
        if abs(dy) >= 6 and dx >= 0:
            custom += 10.0

        # ----- Wall-aware behavior & spindash sequence -----
        stuck_at_wall = (self.stuck_steps > 90 and dx <= 0 and right)

        # discourage pogo at wall while pressing RIGHT
        if stuck_at_wall and jumped:
            custom += self.WALL_JUMP_PENALTY

        if self.stuck_steps > 120:
            # 1) CHARGE (DOWN+B) without RIGHT → arm and reward holding charge
            if down_b and not right:
                if self.spindash_arming == 0:
                    self.spindash_arming = self.SPINDASH_WINDOW
                    self.charge_hold = 0
                self.charge_hold += 1
                custom += self.CHARGE_HOLD_REWARD
            # 2) BURST (RIGHT+DOWN+B) within window → big bonus + start burst timer
            elif burst and self.spindash_arming > 0:
                custom += self.SPINDASH_BONUS
                self.spindash_arming = 0
                self.burst_timer = self.BURST_TIMER_FRAMES
        else:
            self.spindash_arming = 0
            self.charge_hold = 0

        # decay arming window
        if self.spindash_arming > 0:
            self.spindash_arming -= 1

        # reward post-burst forward speed briefly
        if self.burst_timer > 0:
            if dx > 0:
                custom += self.BURST_SPEED_SCALE * dx
            self.burst_timer -= 1

        # Bookkeeping
        self.steps += 1
        self.prev_sx = sx
        self._prev_sy = sy
        self.prev_rings = rings
        self.prev_score = score
        self.prev_lives = lives

        # Early termination on stagnation
        if self.stuck_steps >= self.STAGNATION_CUTOFF:
            done = True
            terminated = True

        if self.steps >= self.MAX_STEPS:
            done = True
            truncated = True

        reward = base_rew + custom
        return obs, reward, terminated, truncated, info
