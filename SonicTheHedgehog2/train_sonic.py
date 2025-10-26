# main_sonic_train.py

"""
PPO Training Pipeline for Sonic 2

Key hyperparameters:
- Rollout steps: 2048
- Learning rate: 2.5e-4
- Clip range: 0.2
- GAE lambda: 0.95

Training loop:
1. Collect 2048 steps of experience
2. Compute advantages using GAE
3. Update policy for 4 epochs
4. Save checkpoint every 50k steps
"""
"""

Simple PPO training loop for Sonic 2 using your existing env + network.
- Adapts to (H,W) or (H,W,1) observations
- Handles Gymnasium 5-tuple steps
- Logs rollout stats and saves checkpoints
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from environment_sonic import make_env
from network_sonic import ActorCriticCNN
from config_sonic import (
    IMG_SIZE, FRAME_STACK, LEARNING_RATE, GAMMA, CLIP_RANGE,
    TOTAL_TIMESTEPS, SAVE_FREQ, DEVICE
)

# ==============================
# Helpers
# ==============================
def to_chw(obs_np):
    """Convert env obs to CHW float32 tensor in [0,1]."""
    # obs can be (H,W) or (H,W,1)
    if obs_np.ndim == 2:
        chw = obs_np[None, :, :]  # (1,H,W)
    elif obs_np.ndim == 3 and obs_np.shape[-1] == 1:
        chw = np.transpose(obs_np, (2, 0, 1))  # (1,H,W)
    else:
        # If someone changed wrappers to RGB, convert to gray mean
        chw = np.mean(obs_np, axis=-1, keepdims=True).transpose(2, 0, 1)
    return chw.astype(np.float32)

@torch.no_grad()
def act_net(net, obs_batch_t):
    logits, value = net(obs_batch_t)
    dist = Categorical(logits=logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return action, logprob, value.squeeze(-1)

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * float(next_value if t == T-1 else values[t+1]) * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns

def minibatches(*arrays, batch_size=64, shuffle=True):
    n = arrays[0].shape[0]
    idx = np.arange(n)
    if shuffle: np.random.shuffle(idx)
    for s in range(0, n, batch_size):
        j = idx[s:s+batch_size]
        yield [a[j] for a in arrays]

# ==============================
# Train
# ==============================
def main():
    env = make_env(render=False)

    # Infer shapes
    obs0, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    chw0 = to_chw(obs0)
    obs_channels = chw0.shape[0]  # likely 1
    num_actions = env.action_space.n

    # Build net & opt
    net = ActorCriticCNN(obs_shape=(obs_channels, IMG_SIZE, IMG_SIZE), num_actions=num_actions).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Hyper-params
    rollout_steps = 2048
    epochs = 4
    batch_size = 64
    lam = 0.95

    global_steps = 0
    episode = 0
    obs_chw = chw0
    ep_return = 0.0

    print(f"✅ Training starts | obs_shape={(obs_channels, IMG_SIZE, IMG_SIZE)} | actions={num_actions}")

    while global_steps < TOTAL_TIMESTEPS:
        # --------- Collect rollout ---------
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        for _ in range(rollout_steps):
            obs_t = torch.from_numpy(obs_chw)[None].to(DEVICE)  # (1,C,H,W)
            with torch.no_grad():
                logits, value = net(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            a = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)

            # store
            obs_buf.append(obs_chw)
            act_buf.append(a)
            logp_buf.append(float(logprob.item()))
            rew_buf.append(float(reward))
            val_buf.append(float(value.squeeze().item()))
            done_buf.append(done)

            # prep next
            obs_chw = to_chw(next_obs)
            ep_return += reward
            global_steps += 1

            if done:
                episode += 1
                # print episodic return occasionally
                if episode % 1 == 0:
                    print(f"[ep {episode:04d}] return={ep_return:.2f} | steps={global_steps}")
                # reset
                out = env.reset()
                obs_chw, _info = out if isinstance(out, tuple) else (out, {})
                obs_chw = to_chw(obs_chw)
                ep_return = 0.0

            if global_steps >= TOTAL_TIMESTEPS:
                break

        # bootstrap with last value
        with torch.no_grad():
            last_v = net(torch.from_numpy(obs_chw)[None].to(DEVICE))[1].squeeze().item()

        # to numpy
        obs_arr = np.array(obs_buf, dtype=np.float32)
        acts = np.array(act_buf, dtype=np.int64)
        old_logp = np.array(logp_buf, dtype=np.float32)
        rews = np.array(rew_buf, dtype=np.float32)
        vals = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.bool_)

        adv, ret = compute_gae(rews, vals, dones, next_value=last_v, gamma=GAMMA, lam=lam)
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --------- PPO update ---------
        obs_t = torch.from_numpy(obs_arr).to(DEVICE)            # (T,C,H,W)
        acts_t = torch.from_numpy(acts).to(DEVICE)              # (T,)
        old_logp_t = torch.from_numpy(old_logp).to(DEVICE)      # (T,)
        adv_t = torch.from_numpy(adv).to(DEVICE)                # (T,)
        ret_t = torch.from_numpy(ret).to(DEVICE)                # (T,)

        Tn = obs_t.shape[0]
        for _ in range(epochs):
            for b_obs, b_act, b_oldlogp, b_adv, b_ret in minibatches(
                obs_t, acts_t, old_logp_t, adv_t, ret_t, batch_size=batch_size, shuffle=True
            ):
                logits, value = net(b_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(b_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - b_oldlogp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value.squeeze(-1), b_ret)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        # --------- checkpoint ---------
        if global_steps // SAVE_FREQ != (global_steps - rollout_steps) // SAVE_FREQ:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = f"checkpoints/sonic_ppo_{global_steps//1000}k.pt"
            torch.save({"model": net.state_dict(),
                        "steps": global_steps,
                        "cfg": dict(A=IMG_SIZE, K=obs_channels, actions=num_actions)}, ckpt)
            print(f"💾 saved {ckpt}")

    env.close()
    print("✅ Training finished.")

if __name__ == "__main__":
    main()
