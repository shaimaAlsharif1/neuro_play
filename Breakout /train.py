import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment import make_env
from network import create_q_model

# Model and training parameters
gamma = 0.99  # Discount factor for past rewards
epsilon_max = 1.0  # Epsilon start value (initial exploration rate)
epsilon_min = 0.1  # Epsilon minimum value (minimum exploration rate)
# Note: epsilon_interval is not explicitly defined in the Keras code block, 
# so we'll infer the decay step from the formula: epsilon_interval = epsilon_max - epsilon_min
epsilon_interval = epsilon_max - epsilon_min 

# Frame and episode counters
frame_count = 0
episode_count = 0
running_reward = 0

# Exploration settings
epsilon_random_frames = 50000  # Number of frames to take random action
epsilon_greedy_frames = 1000000.0 # Number of frames for exploration decay
epsilon = epsilon_max

# Replay Buffer settings
max_memory_length = 100000  # Maximum replay length
update_after_actions = 4    # Train the model after 4 actions
update_target_network = 10000 # How often to update the target network
batch_size = 32             # Batch size for sampling from replay buffer

# Environment parameters (placeholders, must be defined for actual run)
# NOTE: Replace these with actual values/imports for a runnable script
max_steps_per_episode = 10000 
max_episodes = 0 # Set to 0 for infinite loop unless 'solved'
env = make_env(render='human') 
# -------------------------------------------------------------------

# --- PyTorch Setup ---

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
model = create_q_model().to(device)
model_target = create_q_model().to(device)
# Copy weights from model to model_target initially
model_target.load_state_dict(model.state_dict())

# Optimizer 
learning_rate = 0.00025
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss function 
loss_function = nn.HuberLoss() 

# --- Experience Replay Buffer (Using deque for efficiency) ---
action_history = deque(maxlen=max_memory_length)
state_history = deque(maxlen=max_memory_length)
state_next_history = deque(maxlen=max_memory_length)
rewards_history = deque(maxlen=max_memory_length)
done_history = deque(maxlen=max_memory_length)
episode_reward_history = deque(maxlen=100) # For tracking running reward


def process_state(state):
    """
    Converts a numpy state array (H, W, C) from environment (if applicable) 
    or (C, H, W) to a PyTorch tensor with batch dimension, 
    and moves it to the appropriate device.
    """
    # Assuming state is already (C, H, W)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    return state_tensor

def train_step(model, model_target, optimizer, loss_function, indices):
    """Performs one step of the DQN training process."""

    # 1. Sample Replay Buffer Data
    # Convert sampled lists to numpy arrays
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
    
    # Get Q-values for the next states from the TARGET model (stability)
    # model_target is in eval mode for inference
    with torch.no_grad():
        future_rewards = model_target(state_next_tensor)
        # Max Q-value of the next state: max_a' Q_target(s', a')
        max_future_q = torch.max(future_rewards, dim=1).values
        
        # Calculate Target Q-value: Y = R + gamma * max_a' Q_target(s', a')
        # Mask out Q-values for terminal states (done_tensor is 1.0 for terminal)
        # The logic in the Keras code: Y = rewards_sample + gamma * max_future_q
        # and then Y = Y * (1 - done_sample) - done_sample
        # which is equivalent to: Y = R if terminal, or R + gamma * max_future_q if not terminal
        
        # Standard DQN target: R + gamma * max_a' Q_target(s', a') * (1 - done)
        updated_q_values = rewards_tensor + gamma * max_future_q * (1 - done_tensor)

        # The specific Keras logic for terminal states was:
        # updated_q_values = updated_q_values * (1 - done_sample) - done_sample
        # If done=1, this simplifies to 0 - 1 = -1. This is a common but not mandatory 
        # way to handle terminal state rewards when the reward itself isn't used as the Q-target.
        # We will use the standard DQN target: R if terminal, R + gamma * max_future_q if not.
        # This is achieved by: rewards_tensor + (1 - done_tensor) * gamma * max_future_q

    # 3. Compute Current Q-values (Q_model)
    
    model.train() # Set the main model to training mode
    # Get Q-values for the current states from the MAIN model
    current_q_values = model(state_tensor) 

    # Gather Q-values for the actions that were actually taken
    # action_tensor.unsqueeze(1) for indexing
    q_action = torch.gather(current_q_values, dim=1, index=action_tensor.unsqueeze(1)).squeeze()

    # 4. Calculate Loss
    # Calculate loss between the computed target Q-values and the current Q-values for the taken actions
    loss = loss_function(q_action, updated_q_values)

    # 5. Optimization (Backpropagation)
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (equivalent to Keras's clipnorm=1.0)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()

# --- Main Training Loop (Placeholder) ---

print("Starting training loop...")

# NOTE: You MUST initialize and import 'env' (your Gym/Atari environment)
# and ensure 'env' has the expected 'reset()' and 'step()' methods for 
# this loop to run correctly.

# while True:
#     # NOTE: Mocking the environment for demonstration. 
#     # Replace this with your actual environment setup.
#     try:
#         observation, _ = env.reset()
#         state = np.array(observation)
#     except NameError:
#         print("\n*** ERROR: 'env' is not defined. Please set up your environment (e.g., Gym) and uncomment the loop. ***")
#         break

#     episode_reward = 0
    
#     # Epsilon decay happens once per step
#     epsilon_decay_step = epsilon_interval / epsilon_greedy_frames

#     for timestep in range(1, max_steps_per_episode):
#         frame_count += 1
        
#         # --- Epsilon-Greedy Action Selection ---
#         if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
#             # Take random action
#             action = np.random.choice(NUM_ACTIONS)
#         else:
#             # Predict action Q-values
#             state_tensor = process_state(state)
            
#             # Set model to evaluation mode for inference
#             model.eval() 
#             with torch.no_grad():
#                 action_probs = model(state_tensor)
            
#             # Take best action (argmax)
#             action = torch.argmax(action_probs[0]).item()
            
#             # Set model back to training mode (if not training, this has no effect)
#             model.train()

#         # Decay probability of taking random action
#         epsilon -= epsilon_decay_step
#         epsilon = max(epsilon, epsilon_min)

#         # --- Apply action in environment ---
#         # state_next, reward, done, _, _ = env.step(action) 
#         # state_next = np.array(state_next)
        
#         # NOTE: Mocking environment step for demonstration
#         state_next = np.zeros_like(state) 
#         reward = random.uniform(-1, 1)
#         done = timestep == (max_steps_per_episode - 1) 
        
#         episode_reward += reward

#         # --- Save to Replay Buffer ---
#         action_history.append(action)
#         state_history.append(state)
#         state_next_history.append(state_next)
#         done_history.append(done)
#         rewards_history.append(reward)
#         state = state_next

#         # --- Training Step (Update Main Model) ---
#         if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
#             # Get random indices for the batch
#             indices = np.random.choice(len(done_history), size=batch_size, replace=False)
            
#             # Run the training update
#             loss = train_step(model, model_target, optimizer, loss_function, indices)
            
#         # --- Update Target Network ---
#         if frame_count % update_target_network == 0:
#             # update the target network with new weights
#             model_target.load_state_dict(model.state_dict())
            
#             # Log details
#             template = "running reward: {:.2f} at episode {}, frame count {}"
#             print(template.format(running_reward, episode_count, frame_count))

#         if done:
#             break

#     # --- Episode End Logic ---
#     episode_reward_history.append(episode_reward)
#     running_reward = np.mean(episode_reward_history)

#     episode_count += 1

#     if running_reward > 40:  # Condition to consider the task solved
#         print("Solved at episode {}!".format(episode_count))
#         break

#     if max_episodes > 0 and episode_count >= max_episodes: 
#         print("Stopped at episode {}!".format(episode_count))
#         break