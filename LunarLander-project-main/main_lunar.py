import argparse
import torch
import os
from environment import random_rollout
from train import train_dqn, evaluate, DQNAgent, DQNConfig, make_env

def parse_args():
    p = argparse.ArgumentParser(description="LunarLander-v2 random or DQN training")
    p.add_argument("--random", action="store_true", help="Run random agent for a few episodes")
    p.add_argument("--episodes", type=int, default=5, help="Episodes for random run")
    p.add_argument("--render", action="store_true", help="Render during random/eval runs")
    p.add_argument("--train", action="store_true", help="Train DQN")
    p.add_argument("--train_steps", type=int, default=50_000, help="Training steps")
    p.add_argument("--eval_episodes", type=int, default=5, help="Evaluation episodes after training")
    p.add_argument("--save_video", action="store_true", help="Record MP4(s) to ./videos during eval")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--run_trained", action="store_true", help="Run the trained model only (no retraining)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.random:
        random_rollout(episodes=args.episodes, render=args.render, seed=args.seed)

    elif args.train:
        train_dqn(train_steps=args.train_steps,
                  eval_episodes=args.eval_episodes,
                  render_eval=args.render,
                  save_video=args.save_video,
                  seed=args.seed)

    elif args.run_trained:
        # ✅ Load and run the trained model
        model_path = "LunarLander-project-main/dqn_lunarlander.pth"

        env = make_env(render=False)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = DQNConfig()
        agent = DQNAgent(obs_dim, act_dim, device, cfg)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")

        print(f"\033[92mLoading trained model from {model_path}\033[0m")
        agent.load(model_path)

        evaluate(agent,
                 eval_episodes=args.eval_episodes,
                 render_window=args.render,
                 save_video=args.save_video,
                 seed=args.seed)

    else:
        print("Select a mode: --random, --train, or --run_trained (see --help)")


if __name__ == "__main__":
    main()
