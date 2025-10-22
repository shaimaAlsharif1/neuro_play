import gymnasium as gym

def make_env(render: bool = False):
    """
    Headless for training unless render=True (human window).
    """
    render_mode = "human" if render else None
    return gym.make("LunarLander-v3", render_mode=render_mode)

def make_env_human():
    """
    On-screen window for evaluation (requires local GUI).
    """
    return gym.make("LunarLander-v3", render_mode="human")

def make_env_rgb():
    """
    RGB array for RecordVideo (required for MP4 recording).
    """
    return gym.make("LunarLander-v3", render_mode="rgb_array")


# Optional helper for quick manual random rollout (debug)
def random_rollout(episodes: int = 3, render: bool = True, seed: int = 1):
    env = make_env(render=render)
    try:
        for ep in range(1, episodes + 1):
            s, _ = env.reset(seed=seed + ep)
            done = tr = False
            total = 0.0
            while not (done or tr):
                a = env.action_space.sample()
                s, r, done, tr, _ = env.step(a)
                total += r
            print(f"[random] ep {ep}/{episodes} return={total:.2f}")
    finally:
        env.close()
