import os
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from dqn import DQNBreakout
from model import AtariNet
from agent import Agent

os.environ['KMP_duplicate_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device,render_mode = 'rgb_array')
# environment = DQNBreakout(device=device,render_mode = 'human')

model = AtariNet(nb_actions=4)

model.to(device)

model.load_model()

agent = Agent(model= model,
              device= device,
              epsilon=1.0,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.00001,
              memory_capacity=10000,
              batch_size=64)


agent.train(env=environment, epochs=200000)

# state = environment.reset()

# action_probs = model.forward(state).detach()

# print(f"{action_probs}, {torch.argmax(action_probs, dim=-1, keepdim=True)}")

# print(model.forward(state))

# # for _ in range(100):
# #     action = environment.action_space.sample()
# #     state, reward, terminated, truncated , info = environment.step(action)
