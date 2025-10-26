
# config_sonic.py
"""
Global configuration file for Sonic RL project.
Defines environment, training, and model hyperparameters.
"""

# ---------------------- Environment settings ----------------------
import torch


GAME_ID = "SonicTheHedgehog2-Genesis"
STATE = "EmeraldHillZone.Act1"

FRAME_SKIP = 4            # Number of frames to skip per action
FRAME_STACK = 4           # Number of frames stacked for CNN input
IMG_SIZE = 84             # Resized frame width/height
MAX_EPISODE_STEPS = 4500  # Max frames per episode (~75s)

# ---------------------- Action space ----------------------
DISCRETE_ACTIONS = [
    [],                   # 0: No-op
    ['RIGHT'],            # 1
    ['RIGHT', 'A'],       # 2
    ['RIGHT', 'B'],       # 3
    ['LEFT'],             # 4
    ['LEFT', 'A'],        # 5
    ['DOWN'],             # 6
    ['UP'],               # 7
    ['A'],                # 8
    ['B'],                # 9
    # ['RIGHT', 'C'],
]

# ---------------------- PPO hyperparameters ----------------------
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
CLIP_RANGE = 0.2
TOTAL_TIMESTEPS = 5_000_000
SAVE_FREQ = 50_000

# ---------------------- Device ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def print_config():
    """Display all hyperparameters for logging"""
    print("="*50)
    print("CONFIGURATION")
    print("="*50)
    print(f"Game: {GAME_ID}")
    print(f"State: {STATE}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Frame stack: {FRAME_STACK}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"Device: {DEVICE}")
    print("="*50)
