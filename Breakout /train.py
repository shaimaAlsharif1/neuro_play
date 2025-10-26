import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from collections import deque

# Import model, environment setup, and configurations
from network import create_q_model, NUM_ACTIONS
from environment import make_env
from config import DQNConfig
from main import parse_args, main as main_parser # Import main's parser

# --- Initialization ---

# Check if main.py provided arguments for running (e.g., --train)
args = parse_args()
if not args.train:
    # If not running training via the command line, exit gracefully
    print("Run 'python main.py --train' to start training. Exiting train.py setup.")
    # Note: We exit here because `train.py` should only execute the loop if training is requested.
    sys.exit(0) 

# Environment setup
env = make_env(render=False) # Use the headless environment for training
max_steps_per_episode = DQNConfig.max_steps_per_episode # Max steps per episode
max_episodes = DQNConfig.max_episodes

# PyTorch Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Models
model = create_q_model().to(device)
model_target = create_q_model().to(device)
model_target.load_state_dict(model.state_dict())

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=DQNConfig.learning_rate)
loss_function = nn.HuberLoss() 

# Exploration and Decay
epsilon_interval = DQNConfig.epsilon_max - DQNConfig.epsilon_min 
epsilon = DQNConfig.epsilon_max
epsilon_decay_step = epsilon_interval / DQNConfig.epsilon_greedy_frames

# Frame and episode counters
frame_count = 0
episode_count = 0
running_reward = 0

# Experience Replay Buffer 
action_history = deque(maxlen=DQNConfig.max_memory_length)
state_history = deque(maxlen=DQNConfig.max_memory_length)
state_next_history = deque(maxlen=DQNConfig.max_memory_length)
rewards_history = deque(maxlen=DQNConfig.max_memory_length)
done_history = deque(maxlen=DQNConfig.max_memory_length)
episode_reward_history = deque(maxlen=100)


def process_state(state):
    """
    Converts a FrameStack output (which is already (C, H, W) via AtariPreprocessing) 
    to a PyTorch tensor with batch dimension, and moves it to the device.
    """
    # Note: FrameStack output is typically a LazyArray which converts to np.array on access.
    # It has shape (4, 84, 84), which is (C, H, W).
    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
    return state_tensor

def train_step(model, model_target, optimizer, loss_function, indices):
    """Performs one step of the DQN training process."""

    # 1. Sample Replay Buffer Data
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


# --- Main Training Loop ---

print("Starting DQN training loop...")
print(f"Target steps: {args.train_steps} | Env: {args.env} | Seed: {args.seed}")

while frame_count < args.train_steps:
    # Reset environment and get initial state
    observation, _ = env.reset(seed=DQNConfig.seed + episode_count)
    state = np.array(observation)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode + 1):
        frame_count += 1
        
        # --- Epsilon-Greedy Action Selection ---
        if frame_count < DQNConfig.epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(NUM_ACTIONS)
        else:
            state_tensor = process_state(state)
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
            
            # Get random indices for the batch
            indices = np.random.choice(len(rewards_history), 
                                      size=DQNConfig.batch_size, 
                                      replace=False)
            
            # Run the training update
            loss = train_step(model, model_target, optimizer, loss_function, indices)
            
        # --- Update Target Network ---
        if frame_count % DQNConfig.update_target_network == 0:
            model_target.load_state_dict(model.state_dict())
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

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

print(f"Training finished after {frame_count} frames.")
env.close()

# # --- train.py (Add to the end of the file) ---

# def save_checkpoint(model, optimizer, frame_count, running_reward, path="dqn_checkpoint.pth"):
#     """Saves the current state of the model and optimizer."""
#     torch.save({
#         'frame_count': frame_count,
#         'running_reward': running_reward,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }, path)
#     print(f"\n--- Model saved successfully at frame {frame_count} to {path} ---")


# # --- Add this call after the main training loop breaks ---

# # ... (Previous training loop code) ...

# print(f"Training finished after {frame_count} frames.")
# env.close()

# # Save the final model
# save_checkpoint(
#     model, 
#     optimizer, 
#     frame_count, 
#     running_reward, 
#     path=f"dqn_final_{DQNConfig.env_id}_{frame_count}.pth"
# )

# # Example of how to load a saved model later

# # 1. Initialize models and optimizer first (using create_q_model)
# loaded_model = create_q_model().to(device)
# loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=DQNConfig.learning_rate)

# # 2. Load the checkpoint
# checkpoint = torch.load("dqn_final_BreakoutNoFrameskip-v4_X.pth")

# # 3. Apply the saved states
# loaded_model.load_state_dict(checkpoint['model_state_dict'])
# loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # 4. Resume training counters (optional)
# # start_frame = checkpoint['frame_count']
# # last_reward = checkpoint['running_reward']

# # Set the model to evaluation mode (if running inference/testing)
# # loaded_model.eval()
