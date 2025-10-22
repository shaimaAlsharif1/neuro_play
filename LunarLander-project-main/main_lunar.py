

#import and set up
import argparse
from train import train_dqn
from environment import random_rollout




# for command line arguments
def parse_args():
    p = argparse.ArgumentParser(description="LunarLander-v2 random or DQN training")
    p.add_argument("--random", action="store_true", help="Run random agent for a few episodes")
    p.add_argument("--episodes", type=int, default=5, help="Episodes for random run")
    p.add_argument("--render", action="store_true", help="Render during random/eval runs")
    p.add_argument("--train", action="store_true", help="Train DQN")
    p.add_argument("--train_steps", type=int, default=50_000, help="Training steps")
    p.add_argument("--eval_episodes", type=int, default=5, help="Evaluation episodes after training")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    return p.parse_args()

# the main function
def main():
    args = parse_args()
    if args.random:
        random_rollout(episodes=args.episodes, render=args.render)
    elif args.train:
        train_dqn(train_steps=args.train_steps, eval_episodes=args.eval_episodes, render_eval=args.render, seed=args.seed)
    else:
        print("Select a mode: --random or --train (see --help)")


if __name__ == "__main__":
    main()
