# resetstate_sonic.py
"""
Reward shaping and state reset wrapper for Sonic 2.
Uses impactful rewards (29-30-50 scale) without ring dependencies.
"""

import gymnasium as gym
import numpy as np

class ResetStateWrapper(gym.Wrapper):
    """
    Custom reward shaping with high-impact rewards (29-30-50 scale).
    Focused purely on progress, strategic actions, and penalties.
    No ring-based rewards.
    """

    def __init__(self, env, max_steps=4500):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_steps = 0

        # Jump management
        self.jump_count = 0
        self.consecutive_jumps = 0
        self.last_action = 0
        self.jump_cooldown = 0

        # Progress tracking
        self.last_x = 0
        self.max_x_reached = 0
        self.stuck_counter = 0
        self.last_reward_x = 0
        self.zone_start_x = 0

        # State tracking
        self.last_lives = 3
        self.last_score = 0

    def reset(self, **kwargs):
        """Reset environment and tracking variables"""
        obs, info = self.env.reset(**kwargs)

        # Reset tracking variables
        self.current_steps = 0
        self.jump_count = 0
        self.consecutive_jumps = 0
        self.last_action = 0
        self.jump_cooldown = 0

        # Get initial position
        self.last_x = self._get_screen_x(info)
        self.max_x_reached = self.last_x
        self.zone_start_x = self.last_x
        self.stuck_counter = 0
        self.last_reward_x = self.last_x

        self.last_lives = info.get('lives', 3)
        self.last_score = info.get('score', 0)

        return obs, info

    def _get_screen_x(self, info):
        """Extract screen x position from info dict"""
        return info.get('screen_x', info.get('x', 0))

    def _calculate_reward_shaping(self, info, action):
        """
        Calculate shaped rewards using impactful values (29-30-50 scale)
        No ring-based rewards or penalties
        """
        reward = 0.0
        current_x = self._get_screen_x(info)
        current_lives = info.get('lives', 3)
        current_score = info.get('score', 0)

        # === MAJOR PROGRESS REWARDS (50 scale) ===
        # Major forward progress reward
        x_progress = current_x - self.last_reward_x
        if x_progress > 20:  # Significant movement
            progress_reward = min(50, x_progress * 2)  # Cap at 50
            reward += progress_reward
            self.last_reward_x = current_x
            self.stuck_counter = 0

        # Major milestone reward - reaching new maximum
        if current_x > self.max_x_reached + 100:  # Major progress
            milestone_bonus = 50
            reward += milestone_bonus
            self.max_x_reached = current_x
            self.stuck_counter = 0

        # Zone completion bonus (estimated)
        zone_length = 6000  # Approximate Emerald Hill Zone length
        zone_progress = (current_x - self.zone_start_x) / zone_length
        if zone_progress > 0.8:  # 80% through zone
            reward += 50

        # === STRATEGIC ACTION REWARDS (30 scale) ===
        # Smart spindash usage (action 2) - helps with walls
        if action == 2 and self.stuck_counter > 10:  # Using spindash when stuck
            reward += 30
            self.stuck_counter = 0  # Reset stuck counter

        # Smart crouching when stuck (action 3)
        if action == 3 and self.stuck_counter > 15:
            reward += 15

        # Effective jumping (action 1) - only when it makes sense
        if action == 1 and self.stuck_counter < 5 and self.consecutive_jumps == 0:
            reward += 10  # Reward strategic jumping

        # === PENALTIES (29 scale) ===
        # Major penalty for excessive jumping
        if action == 1:  # Jump action
            self.jump_count += 1
            self.consecutive_jumps += 1 if self.last_action == 1 else 1

            # Heavy penalty for spam jumping
            if self.consecutive_jumps > 3:
                reward -= 29  # Major penalty for jump spam
            elif self.consecutive_jumps > 2:
                reward -= 15

        # Penalty for being stuck
        if self.stuck_counter > 30:
            reward -= 20
        if self.stuck_counter > 60:
            reward -= 29

        # Major penalty for losing life
        if current_lives < self.last_lives:
            reward -= 50  # Very heavy penalty

        # === CONSISTENT PROGRESS REWARDS ===
        # Small consistent forward movement
        if x_progress > 0 and x_progress <= 20:
            reward += x_progress * 0.5  # 1-10 points for small progress

        # Score increase reward (optional - can remove if you want)
        score_gain = current_score - self.last_score
        if score_gain > 0:
            reward += min(20, score_gain * 0.1)

        # === STUCK DETECTION ===
        # Detect if agent is stuck (not making progress)
        if abs(current_x - self.last_x) < 5:  # Minimal movement
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        # Small survival reward
        reward += 1

        # Update trackers
        self.last_x = current_x
        self.last_action = action
        self.last_lives = current_lives
        self.last_score = current_score

        return reward

    def step(self, action):
        """Step with high-impact reward shaping (no rings)"""
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # Calculate shaped reward
        shaped_reward = self._calculate_reward_shaping(info, action)

        # Combine environment reward with shaped reward
        total_reward = env_reward + shaped_reward

        # Step counter for episode termination
        self.current_steps += 1
        if self.current_steps >= self.max_steps:
            truncated = True

            # Major bonus for surviving full episode
            survival_bonus = 30
            total_reward += survival_bonus

            # Progress bonus based on how far they got
            progress_bonus = (self.max_x_reached - self.zone_start_x) * 0.01
            total_reward += min(50, progress_bonus)

        # Debug info
        if shaped_reward != 0:
            print(f"Step {self.current_steps}: "
                  f"X={self._get_screen_x(info)}, "
                  f"Action={action}, "
                  f"ShapedReward={shaped_reward:.1f}, "
                  f"TotalReward={total_reward:.1f}")

        return obs, total_reward, terminated, truncated, info

class SonicRewardWrapper(gym.Wrapper):
    """
    Alternative high-impact reward wrapper (no rings)
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_x = 0
        self.jump_history = []
        self.max_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x = self._get_x(info)
        self.max_x = self.last_x
        self.jump_history = []
        return obs, info

    def _get_x(self, info):
        return info.get('screen_x', info.get('x', 0))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_x = self._get_x(info)

        # High-impact progress rewards
        if current_x > self.max_x:
            progress = current_x - self.max_x
            if progress > 50:
                reward += 50  # Major progress
            elif progress > 20:
                reward += 20  # Good progress
            else:
                reward += progress * 0.5  # Small progress
            self.max_x = current_x

        # Jump management with high penalties
        if action == 1:  # Jump action
            self.jump_history.append(1)
            if len(self.jump_history) > 8:
                self.jump_history.pop(0)

            # Heavy penalty for jump spam
            if sum(self.jump_history) >= 6:  # 75% jumping
                reward -= 29
        else:
            self.jump_history.append(0)
            if len(self.jump_history) > 8:
                self.jump_history.pop(0)

        # Reward strategic actions
        if action == 2:  # Spindash
            reward += 15

        self.last_x = current_x
        return obs, reward, terminated, truncated, info
