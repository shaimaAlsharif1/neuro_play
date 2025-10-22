
import gymnasium as gym

def make_env(render: bool):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return env

# runs the game randomly, no training and learning
def random_rollout(episodes: int = 5, render: bool = False):
    env = make_env(render)
    try:
        for ep in range(1, episodes + 1):
            s, _ = env.reset()
            done, tr = False, False
            total = 0.0
            while not (done or tr):
                a = env.action_space.sample()
                ns, r, done, tr, _ = env.step(a)
                total += r
                s = ns
            print(f"[Random] Episode {ep}/{episodes} | Return: {total:.2f}")
    finally:
        env.close()
