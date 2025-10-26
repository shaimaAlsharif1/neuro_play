import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import sys

# Import model, environment setup, and configurations
from network import create_q_model, NUM_ACTIONS
from environment import make_env
from config import DQNConfig

# --- Helper Functions for Checkpointing and State Management ---

# def save_checkpoint(model, optimizer, frame_count, episode_count, running_reward, path):
#     """Saves the current state of the model and optimizer."""
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save({
#         'frame_count': frame_count,
#         'episode_count': episode_count,
#         'running_reward': running_reward,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, path)
#     print(f"\n--- Checkpoint saved successfully at frame {frame_count} to {path} ---")

# def load_checkpoint(model, optimizer, path, device):
#     """Loads a saved checkpoint and returns the training stats."""
#     if not os.path.exists(path):
#         print(f"Error: Checkpoint file not found at {path}")
#         return 0, 0, 0.0 # Return default starting values

#     print(f"Loading checkpoint from {path}...")
#     checkpoint = torch.load(path, map_location=device)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#     frame_count = checkpoint['frame_count']
#     episode_count = checkpoint['episode_count']
#     running_reward = checkpoint['running_reward']

#     print(f"Resuming training from Frame: {frame_count}, Episode: {episode_count}, Reward: {running_reward:.2f}")
#     return frame_count, episode_count, running_reward


def process_state(state, device):
    """
    Converts a FrameStack output (which is (C, H, W)) to a PyTorch tensor
    with batch dimension, and moves it to the device.
    """
    # state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
    # return state_tensor
    state = np.array(state, dtype=np.float32)
    if len(state.shape) == 2:  # single frame
        state = np.expand_dims(state, axis=0)  # add channel dimension
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # add batch dim
    return state_tensor

def train_step(model, model_target, optimizer, loss_function, device,
               action_history, state_history, state_next_history,
               rewards_history, done_history):
    """Performs one step of the DQN training process."""

    # 1. Sample Replay Buffer Data
    indices = np.random.choice(len(rewards_history), size=DQNConfig.batch_size, replace=False)

    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = np.array([rewards_history[i] for i in indices])
    action_sample = np.array([action_history[i] for i in indices])
    done_sample = np.array([done_history[i] for i in indices], dtype=float)

    # Convert to PyTorch tensors
    state_tensor = torch.tensor(state_sample, dtype=torch.float32).to(device)
    state_next_tensor = torch.tensor(state_next_sample, dtype=torch.float32).to(device)
    rewards_tensor = torch.tensor(rewards_sample, dtype=torch.float32).to(device)
    action_tensor = torch.tensor(action_sample, dtype=torch.long).to(device)
    done_tensor = torch.tensor(done_sample, dtype=torch.float32).to(device)

    # 2. Compute Target Q-values (Y)
    model_target.eval()
    with torch.no_grad():
        future_rewards = model_target(state_next_tensor)
        max_future_q = torch.max(future_rewards, dim=1).values

        # Y = R + gamma * max_a' Q_target(s', a') * (1 - done)
        updated_q_values = rewards_tensor + DQNConfig.gamma * max_future_q * (1 - done_tensor)

    # 3. Compute Current Q-values (Q_model)
    model.train()
    current_q_values = model(state_tensor)
    # Gather Q-values for the actions that were actually taken
    q_action = torch.gather(current_q_values, dim=1, index=action_tensor.unsqueeze(1)).squeeze()

    # 4. Calculate Loss
    loss = loss_function(q_action, updated_q_values)

    # 5. Optimization
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=DQNConfig.clipnorm)
    optimizer.step()

    return loss.item()

# --- Main Training Orchestrator ---

def run_dqn_training(train_steps, render, seed, checkpoint_path=None):
    """Initializes and runs the main DQN training loop."""

    # Set initial frame and episode counters
    frame_count = 0
    episode_count = 0
    running_reward = 0.0

    # PyTorch Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    model = create_q_model().to(device)
    model_target = create_q_model().to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=DQNConfig.learning_rate)
    loss_function = nn.HuberLoss()

    # Checkpoint Loading (if provided)
    # if checkpoint_path:
    #     frame_count, episode_count, running_reward = load_checkpoint(
    #         model, optimizer, checkpoint_path, device
    #     )

    # Copy weights from model to model_target (or update after loading)
    # model_target.load_state_dict(model.state_dict())

    # Exploration and Decay
    epsilon_interval = DQNConfig.epsilon_max - DQNConfig.epsilon_min
    epsilon = DQNConfig.epsilon_max
    # Adjust epsilon if resuming from a high frame count
    if frame_count > DQNConfig.epsilon_random_frames:
        decayed_steps = frame_count - DQNConfig.epsilon_random_frames
        epsilon = max(DQNConfig.epsilon_min, DQNConfig.epsilon_max - (epsilon_interval / DQNConfig.epsilon_greedy_frames) * decayed_steps)

    # Replay Buffer
    action_history = deque(maxlen=DQNConfig.max_memory_length)
    state_history = deque(maxlen=DQNConfig.max_memory_length)
    state_next_history = deque(maxlen=DQNConfig.max_memory_length)
    rewards_history = deque(maxlen=DQNConfig.max_memory_length)
    done_history = deque(maxlen=DQNConfig.max_memory_length)
    episode_reward_history = deque(maxlen=100)

    # Environment Setup
    # Use render flag from main.py for the training environment
    env = make_env(render=render)
    max_steps_per_episode = DQNConfig.max_steps_per_episode
    max_episodes = DQNConfig.max_episodes

    print(f"Starting DQN training loop...")
    print(f"Target steps: {train_steps} | Device: {device} | Initial Epsilon: {epsilon:.4f}")

    # Epsilon decay happens once per step
    epsilon_decay_step = epsilon_interval / DQNConfig.epsilon_greedy_frames

    try:
        while frame_count < train_steps:
            # Reset environment and get initial state
            # Use seed + episode_count to ensure diverse starting states across episodes
            observation, _ = env.reset(seed=seed + episode_count)
            state = np.array(observation)
            episode_reward = 0

            for timestep in range(1, max_steps_per_episode + 1):

                # --- Break if global frame limit reached ---
                if frame_count >= train_steps:
                    break

                frame_count += 1

                # --- Epsilon-Greedy Action Selection ---
                if frame_count < DQNConfig.epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    action = np.random.choice(NUM_ACTIONS)
                else:
                    state_tensor = process_state(state, device)
                    model.eval()
                    with torch.no_grad():
                        action_probs = model(state_tensor)
                    action = torch.argmax(action_probs[0]).item()
                    model.train()

                # Decay probability of taking random action
                epsilon -= epsilon_decay_step
                epsilon = max(epsilon, DQNConfig.epsilon_min)

                # --- Apply action in environment ---
                state_next, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state_next = np.array(state_next)
                episode_reward += reward

                # --- Save to Replay Buffer ---
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # --- Training Step ---
                if (frame_count % DQNConfig.update_after_actions == 0 and
                    len(rewards_history) > DQNConfig.batch_size):

                    train_step(
                        model, model_target, optimizer, loss_function, device,
                        action_history, state_history, state_next_history,
                        rewards_history, done_history
                    )

                # --- Update Target Network ---
                if frame_count % DQNConfig.update_target_network == 0:
                    model_target.load_state_dict(model.state_dict())
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                    # Optional: Save checkpoint periodically
                    # save_checkpoint(model, optimizer, frame_count, episode_count, running_reward, f"checkpoints/dqn_frame_{frame_count}.pth")

                if done:
                    break

            # --- Episode End Logic ---
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                episode_reward_history.popleft()

            running_reward = np.mean(episode_reward_history)
            episode_count += 1

            # Check for solving condition
            if running_reward > DQNConfig.solved_running_reward:
                print("Solved at episode {}! Running reward: {:.2f}".format(episode_count, running_reward))
                break

            # Check for max episodes limit
            if max_episodes > 0 and episode_count >= max_episodes:
                print("Stopped at episode {}!".format(episode_count))
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")

    finally:
        print(f"Training finished after {frame_count} frames.")

        # Save the final model
        final_path = f"dqn_final_{DQNConfig.env_id}_{frame_count}.pth"
        # save_checkpoint(
        #     model,
        #     optimizer,
        #     frame_count,
        #     episode_count,
        #     running_reward,
        #     path=final_path
        # )
        env.close()
