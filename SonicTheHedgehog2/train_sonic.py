# main_sonic_train.py
"""
PPO Training Pipeline for Sonic 2
Includes extra scalar features: lives, screen_x, screen_y, screen_x_end
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo

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
    """Compute advantages and returns using GAE."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_next * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns

def minibatches(*arrays, batch_size=64, shuffle=True):
    """Yield minibatches from input arrays."""
    n = arrays[0].shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, n, batch_size):
        j = idx[s:s+batch_size]
        yield [a[j] for a in arrays]

def entropy_coef_schedule(total_steps_done):
    """Linearly decay entropy coefficient 0.05 → 0.01 over 200k steps."""
    progress = min(total_steps_done / 200_000.0, 1.0)
    return 0.05 - 0.04 * progress

# ---------------------------
# Training
# ---------------------------
def main():
    record_next_episode = False
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Environment ----
    env = make_env(render='rgb_array', record_video=True)
    obs0, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    obs_chw = to_chw(obs0)
    obs_channels = obs_chw.shape[0]
    num_actions = env.action_space.n

    # ---- Network & Optimizer ----
    net = ActorCriticCNNExtra(
        obs_shape=(obs_channels, IMG_SIZE, IMG_SIZE),
        num_actions=num_actions,
        extra_state_dim=4
    ).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # ---- Checkpoint loading ----
    model_path = os.path.join(ckpt_dir, "sonic_ppo_latest.pt")
    global_steps = 0
    if os.path.exists(model_path):
        print("\033[93m🔄 Loading existing model...\033[0m")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        net.load_state_dict(checkpoint["model"])
        global_steps = checkpoint.get("steps", 0)

    # ---- PPO hyperparameters ----
    rollout_steps = 2048
    epochs = 4
    batch_size = 64
    lam = 0.95
    grad_clip = 0.5

    episode = 0
    ep_return = 0.0

    print(f"✅ Training starts | obs_shape={(obs_channels, IMG_SIZE, IMG_SIZE)} | actions={num_actions}")

    # ---------------------------
    # Main training loop
    # ---------------------------
    while global_steps < TOTAL_TIMESTEPS:
        obs_buf, extra_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], []

        # -------- Rollout collection --------
        for _ in range(rollout_steps):
            obs_t = torch.from_numpy(obs_chw)[None].to(DEVICE)

            extra_features = np.array([
                info.get('lives', 3),
                info.get('screen_x', 0),
                info.get('screen_y', 0),
                info.get('screen_x_end', 10_000)
            ], dtype=np.float32)
            extra_t = torch.from_numpy(extra_features[None]).to(DEVICE)

            with torch.no_grad():
                logits, value = net(obs_t, extra_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            a = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)

            # Store rollout data
            obs_buf.append(obs_chw)
            extra_buf.append(extra_features)
            act_buf.append(a)
            logp_buf.append(float(logprob.item()))
            rew_buf.append(float(reward))
            val_buf.append(float(value.squeeze().item()))
            done_buf.append(done)

            # Advance
            obs_chw = to_chw(next_obs)
            ep_return += reward
            global_steps += 1

            if done:
                episode += 1
                print(f"[ep {episode:04d}] return={ep_return:.2f} | steps={global_steps:,}")

                # Record next episode if at save step
                if global_steps // SAVE_FREQ != (global_steps - 1) // SAVE_FREQ:
                    record_next_episode = True

                out = env.reset()
                obs_chw, info = out if isinstance(out, tuple) else (out, {})
                obs_chw = to_chw(obs_chw)
                ep_return = 0.0

                if record_next_episode:
                    video_dir = "videos"
                    os.makedirs(video_dir, exist_ok=True)
                    env = RecordVideo(
                        env, video_folder=video_dir,
                        episode_trigger=lambda e: True,
                        name_prefix=f"sonic_ep{episode:04d}"
                    )
                    record_next_episode = False

            if global_steps >= TOTAL_TIMESTEPS:
                break

        # -------- Compute advantages / returns --------
        with torch.no_grad():
            o_last = torch.from_numpy(obs_chw)[None].to(DEVICE)
            e_last = torch.from_numpy(np.array([
                info.get('lives', 3),
                info.get('screen_x', 0),
                info.get('screen_y', 0),
                info.get('screen_x_end', 10_000)
            ], dtype=np.float32)[None]).to(DEVICE)
            _, next_value_t = net(o_last, e_last)
            next_value = float(next_value_t.squeeze().item())

        obs_arr   = np.array(obs_buf, dtype=np.float32)
        extra_arr = np.array(extra_buf, dtype=np.float32)
        act_arr   = np.array(act_buf, dtype=np.int64)
        logp_arr  = np.array(logp_buf, dtype=np.float32)
        rew_arr   = np.array(rew_buf, dtype=np.float32)
        val_arr   = np.array(val_buf, dtype=np.float32)
        done_arr  = np.array(done_buf, dtype=np.bool_)

        adv_arr, ret_arr = compute_gae(
            rewards=rew_arr, values=val_arr, dones=done_arr,
            next_value=next_value, gamma=GAMMA, lam=lam
        )

        # Normalize advantage
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # -------- PPO update ---------
        obs_t   = torch.from_numpy(obs_arr).to(DEVICE)
        extra_t = torch.from_numpy(extra_arr).to(DEVICE)
        act_t   = torch.from_numpy(act_arr).to(DEVICE)
        oldlogp_t = torch.from_numpy(logp_arr).to(DEVICE)
        ret_t   = torch.from_numpy(ret_arr).to(DEVICE)
        val_t   = torch.from_numpy(val_arr).to(DEVICE)
        adv_t   = torch.from_numpy(adv_arr).to(DEVICE)

        for _ in range(epochs):
            for mb_obs, mb_extra, mb_act, mb_oldlogp, mb_ret, mb_val, mb_adv in minibatches(
                obs_t, extra_t, act_t, oldlogp_t, ret_t, val_t, adv_t,
                batch_size=batch_size, shuffle=True
            ):
                logits, value = net(mb_obs, mb_extra)
                dist = Categorical(logits=logits)

                # policy loss
                new_logp = dist.log_prob(mb_act)
                ratio = torch.exp(new_logp - mb_oldlogp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # value loss (clipped)
                value_clipped = mb_val + (value.squeeze() - mb_val).clamp(-CLIP_RANGE, CLIP_RANGE)
                v_loss_unclipped = (value.squeeze() - mb_ret) ** 2
                v_loss_clipped = (value_clipped - mb_ret) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # entropy
                entropy = dist.entropy().mean()
                ent_coef = entropy_coef_schedule(global_steps)
                loss = policy_loss + value_loss - ent_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()

        # -------- Save checkpoint --------
        if global_steps // SAVE_FREQ != (global_steps - rollout_steps) // SAVE_FREQ:
            ckpt = os.path.join(ckpt_dir, f"sonic_ppo_{global_steps // 1000}k.pt")
            torch.save({"model": net.state_dict(), "steps": global_steps}, ckpt)
            torch.save({"model": net.state_dict(), "steps": global_steps},
                       os.path.join(ckpt_dir, "sonic_ppo_latest.pt"))
            print(f"💾 saved {ckpt}")

    print("✅ Training finished")

if __name__ == "__main__":
    main()
