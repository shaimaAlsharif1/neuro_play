import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import copy
import time



class RepalyMemory:


    def __init__(self, capacity, device= "cpu"):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0



    def insert(self, transition):

        # transition = [item.to("cpu") for item in transition]

        # if len(self.memory) < self.capacity:
        #     self.memory.append(transition)

        # else:
        #     self.memory.remove(self.memory[0])
        #     self.memory.append(transition)
        processed = []
        for item in transition:
            if isinstance(item, tuple):   # handle (obs, info)
                item = item[0]
            if isinstance(item, np.ndarray):  # convert to tensor
                item = torch.from_numpy(item).float()
            elif isinstance(item, (int, float, bool)):  # scalar
                item = torch.tensor(item, dtype=torch.float32)
            if hasattr(item, "to"):  # move to CPU safely
                item = item.to("cpu")
            processed.append(item)

        if len(self.memory) < self.capacity:
            self.memory.append(processed)
        else:
            self.memory[self.position] = processed
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)

        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, model, device="cpu", epsilon=1.0, min_epsilon=0.1,
                nb_warmup = 10000, nb_actions=None, memory_capacity=10000, batch_size=32, learning_rate= 0.00025):
        self.memory = RepalyMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        # self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / nb_warmup
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.AdamW(model.parameters(), lr= learning_rate)

        print(f"Starting epsilon is {self.epsilon}")
        print(f"Epsilon decay is {self.epsilon_decay}")


    def get_action(self, state):
        # if torch.rand(1) < self.epsilon:
        #     return torch.randint(self.nb_actions, (1, 1))

        # else:
        #     av = self.model(state).detach()
        #     return torch.argmax(av, dim = 1, keepdim=True)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nb_actions)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.model(state_t)
                return int(torch.argmax(q_values, dim=1).item())



    def train(self, env, epochs):
        stats = {"Returns": [], "AvgReturns": [], "EpsilonCheckpoint": []}

        #plotter = LivePlot()

        for epoch in range(1, epochs + 1):
            state, _ = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b,  done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    # target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    target_b = reward_b + (1 - done_b.float()) * self.gamma * next_qsa_b
                    loss= F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return += float(reward)#.item()

            stats["Returns"].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0 :
                self.model.save_model()
                print("model saved")

                average_returns = np.mean(stats["Returns"][-100:])

                stats["AvgReturns"].append(average_returns)
                stats["EpsilonCheckpoint"].append(self.epsilon)

                if (len(stats["Returns"])) > 100 :
                    print(f'Epoch: {epoch} - Average Return: {np.mean(stats["Returns"][-100:])} - Epsilon: {self.epsilon}')

                else :
                    print(f"Epoch: {epoch} - Average Return: {np.mean(stats['Returns'][-1:])} - Epsilon: {self.epsilon}")

            if epoch % 100 == 0 :
                self.target_model.load_state_dict(self.model.state_dict())
                # plotter.update_plot(stats)


            if epoch % 1000 == 0 :
                self.model.save_model(f"models/model_iter_{epoch}.pt")

        return stats


    def evaluate(self, env):

        for epoch in range(1, 3):
            state, _ = env.reset()

            done = False

            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
