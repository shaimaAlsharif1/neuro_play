import argparse
import numpy as np
import torch
from torch.distributions import Categorical

from environment_sonic import make_env
from network_sonic import ActorCriticCNNExtra
from config_sonic import IMG_SIZE, DEVICE

# --- helpers ---
def preprocess(obs_np):
    # to (C,H,W), float32 in [0,1]; match training
    if obs_np.ndim == 2:
        chw = obs_np[None, :, :]
    elif obs_np.ndim == 3 and obs_np.shape[-1] == 1:
        chw = np.transpose(obs_np, (2, 0, 1))
    else:
        chw = np.mean(obs_np, axis=-1, keepdims=True).transpose(2, 0, 1)
    return chw.astype(np.float32)

def extra_from_info(info):
    # MUST match training order & scale
    return np.array([
        float(info.get("lives", 3)),
        float(info.get("screen_x", 0)),
        float(info.get("screen_y", 0)),
        float(info.get("screen_x_end", 1)),
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sonic_ppo_600k.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--record", action="store_true",
                    help="save mp4 videos to ./videos_play/")
    args = ap.parse_args()

    # Create env (render=True for local viewing)
    env = make_env(render=True)

    # (Optional) record video files
    if args.record:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, video_folder="videos_play",
                          episode_trigger=lambda e: True)

    # Reset & infer shapes
    out = env.reset()
    obs, info = out if isinstance(out, tuple) else (out, {})
    obs_shape = (1, IMG_SIZE, IMG_SIZE)
    num_actions = env.action_space.n

    # Build the SAME net used in training (extra_state_dim=4)
    net = ActorCriticCNNExtra(
        obs_shape=obs_shape,
        num_actions=num_actions,
        extra_state_dim=4
    ).to(DEVICE)

    ckpt = torch.load(args.model, map_location=DEVICE)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"✅ Loaded model from {args.model}")
    print("🕹️ Starting play session...")

    ep = 0
    episode_reward = 0.0
    while ep < args.episodes:
        x = torch.from_numpy(preprocess(obs))[None].to(DEVICE)          # (1,C,H,W)
        extra = torch.from_numpy(extra_from_info(info))[None].to(DEVICE) # (1,4)

        with torch.no_grad():
            logits, v = net(x, extra)          # <-- pass extra_state here
            dist = Categorical(logits=logits)
            action = torch.argmax(dist.probs, dim=-1).item()  # greedy

        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += float(r)

        if done:
            print(f"🏁 Episode {ep+1} finished | total_reward={episode_reward:.2f}")
            out = env.reset()
            obs, info = out if isinstance(out, tuple) else (out, {})
            episode_reward = 0.0
            ep += 1

    env.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()
