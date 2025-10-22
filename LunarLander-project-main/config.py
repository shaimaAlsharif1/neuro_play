from dataclasses import dataclass

@dataclass
class DQNConfig:
    # Discount & optimization
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    max_grad_norm: float = 10.0

    # Exploration (epsilon-greedy)
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    eps_decay_steps: int = 30_000  # how quickly epsilon decreases

    # Training control
    train_after: int = 5_000       # start learning after this many env steps
    train_every: int = 4           # learn every N steps
    target_update_every: int = 1_000
