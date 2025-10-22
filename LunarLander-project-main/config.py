from dataclasses import dataclass

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    eps_decay_steps: int = 30_000
    target_update_every: int = 1_000
    train_after: int = 5_000
    train_every: int = 4
    max_grad_norm: float = 10.0
