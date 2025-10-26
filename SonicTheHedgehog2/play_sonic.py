# play_sonic.py
"""
Run a trained Sonic PPO agent and watch it play.

This script:
1. Loads the trained PPO model (Actor-Critic CNN)
2. Initializes the Sonic environment
3. Lets the agent play using a greedy policy (highest probability action)
"""

import torch
import numpy as np
from torch.distributions import Categorical

from environment_sonic import make_env
from network_sonic import ActorCriticCNN
from config_sonic import IMG_SIZE, DEVICE

# ===========================
# 1️⃣ Environment setup
# ===========================
env = make_env(render=True)  # Render the game window to watch the agent play

out = env.reset()
if isinstance(out, tuple):
    obs, info = out
else:
    obs, info = out, {}

# ===========================
# 2️⃣ Load trained model
# ===========================
checkpoint_path = "checkpoints/sonic_ppo_51k.pt"  # Path to saved checkpoint
ckpt = torch.load(checkpoint_path, map_location=DEVICE)

obs_shape = (1, IMG_SIZE, IMG_SIZE)
num_actions = env.action_space.n

# Build and load the model
net = ActorCriticCNN(obs_shape=obs_shape, num_actions=num_actions).to(DEVICE)
net.load_state_dict(ckpt["model"])
net.eval()

print(f"✅ Loaded model from {checkpoint_path}")
print(f"🕹️ Starting play session...")

# ===========================
# 3️⃣ Preprocessing function
# ===========================
def preprocess(obs):
    """
    Convert observation to (C, H, W) format for the neural network.
    - Handles both grayscale (84x84x1) and RGB inputs.
    """
    if obs.ndim == 2:
        return obs[None, :, :].astype(np.float32)
    elif obs.ndim == 3 and obs.shape[-1] == 1:
        return np.transpose(obs, (2, 0, 1)).astype(np.float32)
    else:
        # Convert RGB to grayscale
        gray = np.mean(obs, axis=-1, keepdims=True)
        return np.transpose(gray, (2, 0, 1)).astype(np.float32)

# ===========================
# 4️⃣ Play loop
# ===========================
episode_reward = 0
for step in range(3000):  # Number of frames to run the agent
    x = torch.from_numpy(preprocess(obs))[None].to(DEVICE)  # (1, C, H, W)
    with torch.no_grad():
        logits, value = net(x)
        probs = Categorical(logits=logits)
        # Choose the most likely action (greedy policy)
        action = probs.probs.argmax(dim=-1).item()

    # Step the environment with the chosen action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward

    # If episode finished, print reward and reset environment
    if done:
        print(f"🏁 Episode finished! Total reward = {episode_reward:.2f}")
        obs, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0

env.close()
print("🎮 Finished play session.")
