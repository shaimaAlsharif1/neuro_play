# train_sonic.py
import os
import time
import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter  # optional (won't error if not installed)
from collections import deque

from environment_sonic import make_env
from network_sonic import ActorCriticCNNExtra as Net
from agent_sonic import PPOAgent
from config_sonic import (
    IMG_SIZE,
    MAX_EPISODE_STEPS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SAVE_EVERY_STEPS = 50_000
CHECKPOINT_DIR = "checkpoints"

def save_checkpoint(path, net, optimizer, global_steps, update, extra_info=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_steps": global_steps,
        "update": update,
        "device": str(DEVICE),
        "extra_info": extra_info or {},
    }, path)
    print(f"💾 Saved checkpoint to {path}")

# ---------- helpers ----------

def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)

@torch.no_grad()
def build_extra(info, prev_sx):
    """Make the 4-dim side input: progress, rings_norm, dx_norm, lives_norm."""
    sx = int(info.get("screen_x", info.get("x", 0)))
    end_x = max(int(info.get("screen_x_end", 0)), 10_000)
    rings = int(info.get("rings", 0))
    lives = int(info.get("lives", 3))

    progress = sx / float(end_x)
    dx = np.clip(sx - prev_sx, -8, 8) / 8.0
    rings_n = min(rings, 100) / 100.0
    lives_n = (lives - 1) / 2.0  # {1..3} -> {0..1}

    return np.array([progress, rings_n, dx, lives_n], dtype=np.float32), sx


def compute_gae(rewards, values, dones, gamma=0.997, lam=0.95, last_value=0.0):
    """
    rewards: (T,)
    values:  (T,)
    dones:   (T,)  done at step t (True means episode ended after reward[t])
    returns: (T,), advantages: (T,)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    # normalize advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return returns.astype(np.float32), adv.astype(np.float32)


# ---------- main training ----------

def main():
    # --- env ---
    env = make_env(render=True, record_video=True)  # videos/sonic_ep-*.mp4
    obs, info = env.reset()
    # obs shape -> ensure channel-first (1,H,W)
    if obs.ndim == 2:
        obs = np.expand_dims(obs, axis=0)
    elif obs.shape[0] != 1:
        # GrayResizeWrapper(keep_dim=False) might give (H,W); keep it (1,H,W)
        obs = np.expand_dims(obs[0], axis=0)

    prev_sx = int(info.get("screen_x", info.get("x", 0)))
    extra, prev_sx = build_extra(info, prev_sx)

    # --- net / agent ---
    net = Net(obs_shape=(1, IMG_SIZE, IMG_SIZE), num_actions=5, extra_state_dim=4).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)  # warm start; you can raise later
    agent = PPOAgent(net, optimizer, clip_range=0.2, grad_clip=0.5, entropy_coef=0.02)

    # --- logs ---
    writer = None
    try:
        writer = SummaryWriter("logs")
    except Exception:
        pass

    # --- hyperparams ---
    steps_per_update = 2048
    epochs = 4
    batch_size = 64
    gamma = 0.997
    gae_lambda = 0.95
    total_updates = 2000

    global_steps = 0
    ep_return, ep_steps = 0.0, 0
    ep = 0

    print(f"✅ Training starts | obs_shape={obs.shape} | actions=5")

    for update in range(total_updates):
        # rollout storage
        obs_buf   = np.zeros((steps_per_update, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        extra_buf = np.zeros((steps_per_update, 4), dtype=np.float32)
        act_buf   = np.zeros((steps_per_update,), dtype=np.int64)
        logp_buf  = np.zeros((steps_per_update,), dtype=np.float32)
        val_buf   = np.zeros((steps_per_update,), dtype=np.float32)
        rew_buf   = np.zeros((steps_per_update,), dtype=np.float32)
        done_buf  = np.zeros((steps_per_update,), dtype=np.float32)

        for t in range(steps_per_update):
            obs_buf[t] = obs.astype(np.float32)
            extra_buf[t] = extra

            # act
            obs_t = to_tensor(obs[None, ...])
            extra_t = to_tensor(extra[None, ...])
            action, logprob, value = agent.act(obs_t, extra_t)

            action_i = int(action.item())
            logp_buf[t] = float(logprob.item())
            val_buf[t] = float(value.item())
            act_buf[t] = action_i

            # step
            next_obs, reward, terminated, truncated, info = env.step(action_i)
            done = terminated or truncated
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)

            ep_return += float(reward)
            ep_steps += 1
            global_steps += 1

            if global_steps % SAVE_EVERY_STEPS == 0:
                    ckpt_path = os.path.join(
                        CHECKPOINT_DIR, f"sonic_step-{global_steps}.pt"
                    )
                    save_checkpoint(
                        ckpt_path, net, optimizer, global_steps, update,
                        extra_info={"episode": ep}
                    )

            # next obs/extra
            if next_obs.ndim == 2:
                next_obs = np.expand_dims(next_obs, axis=0)
            elif next_obs.shape[0] != 1:
                next_obs = np.expand_dims(next_obs[0], axis=0)

            extra, prev_sx = build_extra(info, prev_sx)
            obs = next_obs

            if done:
                if writer:
                    writer.add_scalar("episode/return", ep_return, ep)
                    writer.add_scalar("episode/steps", ep_steps, ep)
                print(f"[ep {ep:04d}] return={ep_return:.2f} | steps={ep_steps}")
                ep += 1
                ep_return, ep_steps = 0.0, 0
                # reset
                obs, info = env.reset()
                if obs.ndim == 2:
                    obs = np.expand_dims(obs, axis=0)
                elif obs.shape[0] != 1:
                    obs = np.expand_dims(obs[0], axis=0)
                prev_sx = int(info.get("screen_x", info.get("x", 0)))
                extra, prev_sx = build_extra(info, prev_sx)

        # bootstrap value for GAE
        with torch.no_grad():
            v_boot = agent.net(to_tensor(obs[None, ...]), to_tensor(extra[None, ...]))[1].item()

        # compute returns & advantages
        ret_buf, adv_buf = compute_gae(rew_buf, val_buf, done_buf, gamma=gamma, lam=gae_lambda, last_value=v_boot)

        # tensors
        obs_t   = to_tensor(obs_buf)
        extra_t = to_tensor(extra_buf)
        act_t   = torch.as_tensor(act_buf, dtype=torch.long, device=DEVICE)
        oldlogp_t = to_tensor(logp_buf)
        ret_t   = to_tensor(ret_buf)
        val_t   = to_tensor(val_buf)
        adv_t   = to_tensor(adv_buf)

        # single PPO update (no duplicate optimization elsewhere)
        agent.update(
            obs_t, extra_t, act_t, oldlogp_t, ret_t, val_t, adv_t,
            epochs=epochs, batch_size=batch_size,
            entropy_coef=lambda s: 0.02,  # fixed or schedule
            global_steps=global_steps
        )

        # (optional) log scalars
        if writer:
            writer.add_scalar("update/mean_return", ret_buf.mean(), update)
            writer.add_scalar("update/advantage_std", adv_buf.std(), update)

    env.close()
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
