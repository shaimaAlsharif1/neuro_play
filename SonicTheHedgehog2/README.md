# Sonic the Hedgehog 2 RL Agent

## Overview
Training a PPO agent to play Sonic 2 (Emerald Hill Zone Act 1)

## Quick Start
1. Install: `pip install retro torch gymnasium opencv-python pandas`
2. Train: `python scripts/train.py`
3. Test: `python scripts/test.py`

## Architecture
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: CNN encoder + Actor-Critic heads
- **Observations**: 84x84 grayscale, 4-frame stack
- **Actions**: 10 discrete (RIGHT, RIGHT+JUMP, etc.)

## Reward Shaping
- Forward progress: +0.1 per 100 pixels
- Ring loss: -0.3
- Life loss: -1.0
- Level completion: +1.0
- Idle penalty: -0.01
- Jump spam penalty: -0.1




note i want to make it like this
sonic-rl-project/
├── README.md                 # Project overview
├── requirements.txt          # Dependencies
├── configs/
│   └── config.py
├── src/
│   ├── agent.py
│   ├── network.py
│   ├── environment.py
│   ├── discretizer.py
│   ├── wrappers/
│   │   ├── resetstate.py
│   │   └── utils.py
├── scripts/
│   ├── train.py
│   └── test.py
├── checkpoints/              # Saved models
└── logs/                     # Episode logs
