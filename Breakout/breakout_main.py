import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import os

model_path = "dqn_breakout_model.h5"

# --- Hyperparameters ---
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay_frames = 1000000.0
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

# --- Create Q-network ---
def build_model(num_actions):
    return keras.Sequential([
        keras.layers.Input(shape=(84, 84, 4)),  # 4 stacked frames
        keras.layers.Conv2D(32, 8, strides=4, activation='relu'),
        keras.layers.Conv2D(64, 4, strides=2, activation='relu'),
        keras.layers.Conv2D(64, 3, strides=1, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')
    ])

if os.path.exists(model_path):
    print("Loading existing model...")
    model = keras.models.load_model(model_path)
    model_target = keras.models.load_model(model_path)  # Load target network too
else:
    print("No existing model found, creating a new one...")
    model = build_model(num_actions)
    model_target = build_model(num_actions)
    model_target.set_weights(model.get_weights())

env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
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

        # --- Apply action ---
        state_next, reward, done, _, _ = env.step(action)
        frame_stack.append(state_next)
        state_next_stacked = np.stack(frame_stack, axis=-1)  # (84, 84, 4)
        episode_reward += reward

        # --- Store in replay buffer ---
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next_stacked)
        rewards_history.append(reward)
        done_history.append(done)
        state = state_next_stacked

        # --- Training step ---
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(len(done_history), batch_size, replace=False)

            state_sample = np.array([state_history[i] for i in indices], dtype=np.float32)
            state_next_sample = np.array([state_next_history[i] for i in indices], dtype=np.float32)
            rewards_sample = np.array([rewards_history[i] for i in indices], dtype=np.float32)
            action_sample = np.array([action_history[i] for i in indices])
            done_sample = np.array([float(done_history[i]) for i in indices], dtype=np.float32)

            # Compute target Q-values
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
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)
    episode_count += 1
    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    print(f"Episode {episode_count}, Reward: {episode_reward:.2f}, Running Reward: {running_reward:.2f}, Epsilon: {epsilon:.3f}")

    if running_reward > 40:  # solved condition
        print(f"Solved at episode {episode_count}!")
        break
