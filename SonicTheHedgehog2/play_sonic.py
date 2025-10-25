# play_sonic.py
"""
Run a trained Sonic PPO agent and watch it play.
"""

import torch
import numpy as np
from torch.distributions import Categorical

from environment_sonic import make_env
from network_sonic import ActorCriticCNN
from config_sonic import IMG_SIZE, DEVICE

# ===========================
# 1️⃣ إعداد البيئة
# ===========================
env = make_env(render=True)

out = env.reset()
if isinstance(out, tuple):
    obs, info = out
else:
    obs, info = out, {}

# ===========================
# 2️⃣ تحميل النموذج
# ===========================
checkpoint_path = "checkpoints/sonic_ppo_51k.pt"  # ← غيّري الاسم إذا عندك ملف آخر
ckpt = torch.load(checkpoint_path, map_location=DEVICE)

obs_shape = (1, IMG_SIZE, IMG_SIZE)
num_actions = env.action_space.n

net = ActorCriticCNN(obs_shape=obs_shape, num_actions=num_actions).to(DEVICE)
net.load_state_dict(ckpt["model"])
net.eval()

print(f"✅ Loaded model from {checkpoint_path}")
print(f"🕹️ Starting play session...")

# ===========================
# 3️⃣ دالة تجهيز الصورة
# ===========================
def preprocess(obs):
    if obs.ndim == 2:
        return obs[None, :, :].astype(np.float32)
    elif obs.ndim == 3 and obs.shape[-1] == 1:
        return np.transpose(obs, (2, 0, 1)).astype(np.float32)
    else:
        # fallback: لو كانت RGB
        gray = np.mean(obs, axis=-1, keepdims=True)
        return np.transpose(gray, (2, 0, 1)).astype(np.float32)

# ===========================
# 4️⃣ حلقة اللعب
# ===========================
episode_reward = 0
for step in range(3000):  # عدد الخطوات للعرض
    x = torch.from_numpy(preprocess(obs))[None].to(DEVICE)  # (1,C,H,W)
    with torch.no_grad():
        logits, value = net(x)
        probs = Categorical(logits=logits)
        action = probs.probs.argmax(dim=-1).item()  # استخدم أكثر أكشن احتمالاً

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward

    if done:
        print(f"🏁 Episode finished! total reward = {episode_reward:.2f}")
        obs, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        episode_reward = 0

env.close()
print("🎮 Finished play session.")
