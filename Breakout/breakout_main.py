import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import os

# model_path = "Breakout/dqn_breakout_model.h5"
# Define the paths for the model and the training state checkpoint file
MODEL_DIR = "Breakout"
CHECKPOINT_BASE_NAME = "dqn_breakout_full_state_{:04d}.npz"
# os.makedirs(MODEL_DIR, exist_ok=True) # Ensure the directory exists
# Reward Shaping Constants
# We amplify the standard environment reward and heavily penalize life loss.
SCORE_MULTIPLIER = 10.0        # Amplifies the reward received for breaking bricks 
LIFE_LOSS_PENALTY = -7.0      # Increased penalty for losing a life
TIME_STEP_PENALTY = -0.001    # Small penalty per step to encourage faster play
CHECKPOINT_FREQUENCY = 100 # Save a checkpoint every 50 episodes
# --- Hyperparameters --- 
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay_frames = 250000.0
num_actions = 4
max_steps_per_episode = 10000
max_episodes = 1000

# Optimizer and loss
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()

# Replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []

# Counters
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 50000
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

# ----------------- Checkpointing Functions -----------------

def save_checkpoint(model, model_target, optimizer, episode_count, frame_count, epsilon, running_reward, episode_reward_history):
    """
    Saves the entire training state (weights and variables) into a single .npz file.
    """
    # ... (filename setup code) ...
    checkpoint_name = CHECKPOINT_BASE_NAME.format(episode_count)
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_name)

    # 2. Extract Weights as NumPy arrays
    model_weights = model.get_weights()
    model_target_weights = model_target.get_weights()

    # Convert the list of arrays to a NumPy array with dtype=object 
    # to force it to save the list structure intact.
    model_weights_np = np.asarray(model_weights, dtype=object)
    model_target_weights_np = np.asarray(model_target_weights, dtype=object)
    
    # 3. Save all data to a single NPZ file
    try:
        np.savez(
            checkpoint_path,
            # Scalar State
            episode_count=episode_count,
            frame_count=frame_count,
            epsilon=epsilon,
            running_reward=running_reward,
            episode_reward_history=np.array(episode_reward_history),
            # Model Weights (now saved correctly as object arrays)
            model_weights=model_weights_np,
            model_target_weights=model_target_weights_np,
        )
        print(f"Single-file checkpoint saved successfully to {checkpoint_path}")
    except Exception as e:
        print(f"Error saving single-file checkpoint: {e}")

def load_checkpoint():
    """Loads the models and the training state from the latest available single-file checkpoint."""
    global episode_count, frame_count, epsilon, running_reward, episode_reward_history

    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('dqn_breakout_full_state_') and f.endswith('.npz')]

    if not checkpoint_files:
        print("No existing checkpoint found, starting from scratch...")
        return None, None

    # Extract episode numbers and find the largest one
    latest_episode = 0
    for filename in checkpoint_files:
        try:
            # Assumes the episode number is a 4-digit number before '.npz'
            episode_num = int(filename[-7:-4]) 
            if episode_num > latest_episode:
                latest_episode = episode_num
        except ValueError:
            continue

    if latest_episode == 0:
        print("No valid numbered checkpoints found.")
        return None, None

    # Construct the path for the latest checkpoint
    latest_checkpoint_name = CHECKPOINT_BASE_NAME.format(latest_episode)
    checkpoint_path = os.path.join(MODEL_DIR, latest_checkpoint_name)

    print(f"Loading latest single-file checkpoint from Episode {latest_episode}...")
    
    try:
        # 1. Load data from NPZ
        # Need allow_pickle=True because weights are stored as objects (list of NumPy arrays)
        state_data = np.load(checkpoint_path, allow_pickle=True)

        # 2. Restore State (Note: .item() is used to extract scalar values from 0-dim arrays)
        episode_count = state_data['episode_count'].item()
        frame_count = state_data['frame_count'].item()
        epsilon = state_data['epsilon'].item()
        running_reward = state_data['running_reward'].item()
        episode_reward_history = state_data['episode_reward_history'].tolist()
        
        # 3. Restore Models
        model = build_model(num_actions)
        model_target = build_model(num_actions)
        
        # --- FIX APPLIED HERE ---
        # Use .item() to extract the original Python list of weights from the NumPy object array
        model_weights = state_data['model_weights'].item() 
        model_target_weights = state_data['model_target_weights'].item()
        
        model.set_weights(model_weights)
        model_target.set_weights(model_target_weights)
        
        model.compile(optimizer=optimizer, loss=loss_function) 
        
        print(f"State and models loaded: Resuming from Episode {episode_count}")
        return model, model_target

    except Exception as e:
        print(f"Error loading single-file checkpoint at {checkpoint_path}. Error: {e}")
        return None, None
    
# --- Create Q-network ---
def build_model(num_actions):
    """Creates the convolutional neural network model for DQN."""
    return keras.Sequential([
        keras.layers.Input(shape=(84, 84, 4)),  # 4 stacked frames
        keras.layers.Conv2D(32, 8, strides=4, activation='relu'),
        keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
        keras.layers.Conv2D(64, 3, strides=1, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')
    ])

# Initialize or load models
model, model_target = load_checkpoint()

if model is None:
    # If load_checkpoint() returned None (meaning no checkpoint was found),
    # create new models and set the initial weights/state.
    model = build_model(num_actions)
    model_target = build_model(num_actions)
    model_target.set_weights(model.get_weights())

# Initialize environment
# # FIX: Changed render_mode to "human" to display the game window.
# env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
# FIX: Changed render_mode to "rgb_array" to train on gcloud
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)

# --- Training loop ---
while episode_count < max_episodes:
    observation, _ = env.reset()
    frame_stack = deque(maxlen=4)

    # Fill the stack with initial frame repeated 4 times
    for _ in range(4):
        frame_stack.append(observation)

    state = np.stack(frame_stack, axis=-1)  # (84, 84, 4)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode + 1):
        frame_count += 1

        # --- Epsilon-greedy action selection ---
        if frame_count < epsilon_random_frames or epsilon > np.random.rand():
            action = np.random.choice(num_actions)
        else:
            state_input = np.expand_dims(state, axis=0)  # (1, 84, 84, 4)
            q_values = model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])

        # Decay epsilon
        epsilon -= (1.0 - epsilon_min) / epsilon_decay_frames
        epsilon = max(epsilon, epsilon_min)

        # If the agent performs a No-Op (0) or unnecessary Fire (2), apply a penalty
        # This encourages the agent to prioritize Move Right (1) or Move Left (3)
        if action == 0 or action == 1:
            reward_adjustment = -0.005 # small penalty for inactivity
        else:
            reward_adjustment = 0.0

        # Store the number of lives BEFORE the step
        # This requires accessing the ALE object via unwrapped
        lives_before = env.unwrapped.ale.lives() 

        # --- Apply action ---
        state_next, reward, done, _, info = env.step(action) # reward is the standard score change
        frame_stack.append(state_next)
        state_next_stacked = np.stack(frame_stack, axis=-1)  # (84, 84, 4)

        # *** REWARD SHAPING LOGIC ***
        lives_after = env.unwrapped.ale.lives() 

        # 1. Amplify Standard Score Reward
        reward *= SCORE_MULTIPLIER

        # 2. Add Time Step Penalty (Encourages faster action)
        reward += TIME_STEP_PENALTY
        
        # --- FIX: ADD THE ACTION-BASED PENALTY HERE ---
        reward += reward_adjustment  # Apply the penalty for No-op or Fire
        
        # 3. Life Loss Penalty: Explicitly punish losing a life
        if lives_after < lives_before:
            # Modify the reward with the large negative value
            reward += LIFE_LOSS_PENALTY 
            print(f"Lives lost! Applying penalty: {LIFE_LOSS_PENALTY}")

        # Accumulate the SHAPED reward for episode tracking
        episode_reward += reward 

        # --- Store in replay buffer ---
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next_stacked)
        # Store the SHAPED reward
        rewards_history.append(reward) 
        done_history.append(done)
        state = state_next_stacked

        # --- Training step (DQN) ---
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(len(done_history), batch_size, replace=False)

            state_sample = np.array([state_history[i] for i in indices], dtype=np.float32)
            state_next_sample = np.array([state_next_history[i] for i in indices], dtype=np.float32)
            rewards_sample = np.array([rewards_history[i] for i in indices], dtype=np.float32)
            action_sample = np.array([action_history[i] for i in indices])
            done_sample = np.array([float(done_history[i]) for i in indices], dtype=np.float32)

            # Compute target Q-values (DQN update rule)
            future_rewards = model_target.predict(state_next_sample, verbose=0)
            max_future_q = np.max(future_rewards, axis=1)
            updated_q_values = rewards_sample + gamma * max_future_q * (1 - done_sample)

            # Train on batch
            masks = tf.one_hot(action_sample, num_actions)
            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(q_values * masks, axis=1)
                loss = loss_function(updated_q_values, q_action)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --- Update target network ---
        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())

        # --- Limit replay buffer size ---
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    # --- End of episode ---
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[0]
    running_reward = np.mean(episode_reward_history)
    episode_count += 1

    # --- CHECKPOINTING ---
    if episode_count % CHECKPOINT_FREQUENCY == 0:
        save_checkpoint(model, model_target, optimizer, episode_count, frame_count, epsilon, running_reward, episode_reward_history)

    print(f"Episode {episode_count}, Reward: {episode_reward:.2f}, Running Reward: {running_reward:.2f}, Epsilon: {epsilon:.3f}")

    if running_reward > 40: # solved condition
        print(f"Solved at episode {episode_count}!")
        break

# Final save after loop completion (either by max_episodes or solved condition)
save_checkpoint(model, model_target, optimizer, episode_count, frame_count, epsilon, running_reward, episode_reward_history)
print("Training finished. Final model and state saved.")