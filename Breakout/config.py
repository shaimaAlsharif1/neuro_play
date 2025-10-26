# config.py

import os
import argparse

# --- Hyperparameters and Constants (DQN) ---

class DQNConfig:
    """Configuration class for DQN training parameters."""

    # Environment and Model
    env_id = "BreakoutNoFrameskip-v4"
    seed = 42 # Default seed

    # Training Loop
    max_steps_per_episode = 10000
    max_episodes = 0 # 0 means run until solved or steps limit reached

    # Optimization
    gamma = 0.99  # Discount factor for past rewards
    learning_rate = 0.00025 # Adam learning rate (as per Keras code)
    clipnorm = 1.0 # Gradient clipping max norm (handled in PyTorch train step)

    # Exploration settings
    epsilon_max = 1.0
    epsilon_min = 0.1
    epsilon_random_frames = 50000  # Frames for pure random action
    epsilon_greedy_frames = 1000000.0 # Frames for epsilon decay

    # Replay Buffer settings
    max_memory_length = 100000
    batch_size = 32

    # Update frequencies
    update_after_actions = 4    # Train the model after 4 environment steps
    update_target_network = 10000 # How often to update the target network

    # Target running reward for solving
    solved_running_reward = 40.0 # Condition to consider the task solved
