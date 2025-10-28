import collections
import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import ale_py

class DQNBreakout(gym.Wrapper):
    def __init__(self,render_mode='rgb_array',repeat=4,device='cpu'):
        env = gym.make('ALE/Breakout-v5',render_mode=render_mode)

        super(DQNBreakout,self).__init__(env)

        self.image_shape = (84,84)
        self.repeat = repeat
        self.device = device
        self.lives = env.unwrapped.ale.lives()
        self.frame_buffer = []

    def step(self,action):
        total_reward = 0
        done = False

        for i in range(self.repeat):
            observation, reward,terminated, truncated,info = self.env.step(action)

            total_reward += reward

            current_lives = info['lives']

            if current_lives < self.lives:
                total_reward -= 1
                self.lives = current_lives
                # print(f"lives: {self.lives} total rewards: { total_reward}")


            self.frame_buffer.append(observation)

            if done:
                break
        # max_frame = np.max(self.frame_buffer[-2:],axis=0)
        # # max_frame = max_frame.to(self.device)
        # max_frame = self.process_observation(max_frame)
        # # max_frame = max_frame.to(self.device)

        # # total_reward = torch.tensor(total_reward).view(1,-1).float()
        # total_reward = torch.tensor([[total_reward]], dtype=torch.float32, device=self.device)
        # # total_reward = total_reward.to(self.device)

        # # done = torch.tensor(done).view(1,-1)
        # # done = done.to(self.device)
        # done_tensor = torch.tensor([[done]], dtype=torch.float32, device=self.device)

        # return max_frame, total_reward, done_tensor , info
        frames = torch.stack([self.process_observation(f) for f in self.frame_buffer[-2:]], dim=0)  # [2,1,H,W]
        max_frame = torch.max(frames, dim=0)[0]  # [1,H,W]

        total_reward = torch.tensor([[total_reward]], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[done]], dtype=torch.float32, device=self.device)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self):
        self.frame_buffer = []

        observation, info = self.env.reset()

        self.lives = self.env.unwrapped.ale.lives()
        observation = self.process_observation(observation)
        return observation, info


    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        # img = img.unsqeeze(0)
        # img = img.unsqeeze(0)
        img = img / 255.0
        img = img.to(self.device)


        return img
