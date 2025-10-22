
from dataclasses import dataclass

# stores all the hyperparameters,
@dataclass
class DQNConfig:
    gamma: float = 0.99 #discount factor
    lr: float = 1e-3 #learning rate
    batch_size: int = 128
    buffer_size: int = 100_000
    start_epsilon: float = 1.0 # explorationi rate
    end_epsilon: float = 0.05
    eps_decay_steps: int = 30_000 # how queckly epsilon decreases
    target_update_every: int = 1_000
    train_after: int = 5_000  #start training after this steps
    train_every: int = 4
    max_grad_norm: float = 10.0
