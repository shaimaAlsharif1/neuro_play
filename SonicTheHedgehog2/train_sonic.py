# main_sonic_train.py
# main_sonic_train.py

"""
PPO Training Pipeline for Sonic 2
Includes extra scalar features: lives, screen_x, screen_y, screen_x_end
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from environment_sonic import make_env
from network_sonic import ActorCriticCNNExtra
from config_sonic import (
    IMG_SIZE, LEARNING_RATE, GAMMA, CLIP_RANGE,
    TOTAL_TIMESTEPS, SAVE_FREQ, DEVICE
)

# ---------------------------
# Helpers
# ---------------------------
def to_chw(obs_np):
    """Convert env obs to CHW float32 tensor in [0,1]."""
    if obs_np.ndim == 2:
        chw = obs_np[None, :, :]
    elif obs_np.ndim == 3 and obs_np.shape[-1] == 1:
        chw = np.transpose(obs_np, (2, 0, 1))
    else:
        chw = np.mean(obs_np, axis=-1, keepdims=True).transpose(2, 0, 1)
    return chw.astype(np.float32)

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * (next_value if t == T-1 else values[t+1]) * non_terminal - values[t]
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

# ---------------------------
# Training
# ---------------------------
def main():
    env = make_env(render=False)

    # Infer obs shape
    obs0, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    chw0 = to_chw(obs0)
    obs_channels = chw0.shape[0]
    num_actions = env.action_space.n

    # Network & optimizer
    net = ActorCriticCNNExtra(
        obs_shape=(obs_channels, IMG_SIZE, IMG_SIZE),
        num_actions=num_actions,
        extra_state_dim=4  # lives, screen_x, screen_y, screen_x_end
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Hyperparams
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
        obs_buf, extra_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], []

        for _ in range(rollout_steps):
            obs_t = torch.from_numpy(obs_chw)[None].to(DEVICE)  # (1,C,H,W)

            # Prepare extra features
            extra_features = np.array([
                info.get('lives', 3),
                info.get('screen_x', 0),
                info.get('screen_y', 0),
                info.get('screen_x_end', 10000)
            ], dtype=np.float32)
            extra_t = torch.from_numpy(extra_features[None]).to(DEVICE)  # (1,4)

            with torch.no_grad():
                logits, value = net(obs_t, extra_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            a = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)

            # store
            obs_buf.append(obs_chw)
            extra_buf.append(extra_features)
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
                print(f"[ep {episode:04d}] return={ep_return:.2f} | steps={global_steps}")
                out = env.reset()
                obs_chw, info = out if isinstance(out, tuple) else (out, {})
                obs_chw = to_chw(obs_chw)
                ep_return = 0.0

            if global_steps >= TOTAL_TIMESTEPS:
                break

        # bootstrap last value
        last_extra = torch.from_numpy(extra_features[None]).to(DEVICE)
        with torch.no_grad():
            last_v = net(torch.from_numpy(obs_chw)[None].to(DEVICE), last_extra)[1].squeeze().item()

        # Convert buffers to tensors
        obs_arr = np.array(obs_buf, dtype=np.float32)
        extra_arr = np.array(extra_buf, dtype=np.float32)
        acts = np.array(act_buf, dtype=np.int64)
        old_logp = np.array(logp_buf, dtype=np.float32)
        rews = np.array(rew_buf, dtype=np.float32)
        vals = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.bool_)

        adv, ret = compute_gae(rews, vals, dones, next_value=last_v, gamma=GAMMA, lam=lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        obs_t = torch.from_numpy(obs_arr).to(DEVICE)
        extra_t = torch.from_numpy(extra_arr).to(DEVICE)
        acts_t = torch.from_numpy(acts).to(DEVICE)
        old_logp_t = torch.from_numpy(old_logp).to(DEVICE)
        adv_t = torch.from_numpy(adv).to(DEVICE)
        ret_t = torch.from_numpy(ret).to(DEVICE)

        for _ in range(epochs):
            for b_obs, b_extra, b_act, b_oldlogp, b_adv, b_ret in minibatches(
                obs_t, extra_t, acts_t, old_logp_t, adv_t, ret_t, batch_size=batch_size, shuffle=True
            ):
                logits, value = net(b_obs, b_extra)
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

        # Checkpoint
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
