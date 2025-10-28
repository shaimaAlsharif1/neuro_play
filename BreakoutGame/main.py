import os
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from dqn import DQNBreakout
from model import AtariNet

os.environ['KMP_duplicate_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# environment = DQNBreakout(device=device,render_mode = 'rgb_array')
environment = DQNBreakout(device=device,render_mode = 'human')

model = AtariNet(nb_actions=4)

model.load_model()

state = environment.reset()

print(model.forward(state))

# for _ in range(100):
#     action = environment.action_space.sample()
#     state, reward, terminated, truncated , info = environment.step(action)
    