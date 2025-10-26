import argparse
import os

# --- Constants for Breakout ---
DEFAULT_ENV_ID = "BreakoutNoFrameskip-v4"
DEFAULT_TRAIN_STEPS = 5_000_000  # Atari training is typically in millions of steps
DEFAULT_RANDOM_EPISODES = 5
DEFAULT_EVAL_EPISODES = 10

def parse_args():
    p = argparse.ArgumentParser(description="Breakout-v4/NoFrameskip-v4 random or DQN training")

    # --- Environment Selection ---
    p.add_argument(
        "--env",
        type=str,
        default=DEFAULT_ENV_ID,
        help="Atari environment ID (e.g., BreakoutNoFrameskip-v4 or ALE/Breakout-v5)",
    )

    # --- Random Rollout Mode ---
    p.add_argument("--random", action="store_true", help="Run random agent for a few episodes (for verification)")
    p.add_argument("--episodes", type=int, default=DEFAULT_RANDOM_EPISODES, help="Episodes for random run")
    p.add_argument("--render", action="store_true", help="Render the game during random/eval runs")

    # --- Training Mode ---
    p.add_argument("--train", action="store_true", help="Train DQN agent")
    p.add_argument("--train_steps", type=int, default=DEFAULT_TRAIN_STEPS, help="Total training steps (e.g., 5,000,000 for Atari)")
    p.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES, help="Evaluation episodes after training")
    p.add_argument("--save_video", action="store_true", help="Record MP4(s) to ./videos during eval")

    # --- Global Settings ---
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return p.parse_args()

def main():
    args = parse_args()

    # Set the environment variable for reproducibility outside of gym's seed()
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.random:
        # Note: You'll need to define the random_rollout function to accept the 'env' argument
        print(f"Running random rollout on {args.env} for {args.episodes} episodes...")
        # Assume random_rollout function is defined elsewhere
        # random_rollout(env_id=args.env, episodes=args.episodes, render=args.render, seed=args.seed)
        # Placeholder for actual function call

    elif args.train:
        print(f"Starting DQN training on {args.env} for {args.train_steps} steps...")
        # Assume train_dqn function is defined elsewhere
        # train_dqn(env_id=args.env,
        #           train_steps=args.train_steps,
        #           eval_episodes=args.eval_episodes,
        #           render_eval=args.render,
        #           save_video=args.save_video,
        #           seed=args.seed)
        # Placeholder for actual function call

    else:
        print("Error: Select a mode: --random or --train (see --help)")

if __name__ == "__main__":
    main()
