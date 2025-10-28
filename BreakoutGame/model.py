import torch
import torch.nn as nn
import os

class AtariNet(nn.Module):
    def __init__(self, nb_actions=4):
        super(AtariNet,self).__init__()

        self.relu = nn.ReLU()
        self.convl = nn.Conv2d(1,32,kernel_size=(8,8), stride=(4,4))
        self.convl = nn.Conv2d(32,64,kernel_size=(4,4), stride=(2,2))
        self.convl = nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=.2)
        self.action_value1 = nn.Linear(3136,1024)
        self.action_value2 = nn.Linear(1024,1024)
        self.action_value3 = nn.Linear(1024,nb_actions)

        self.state_value1 = nn.Linear(3134,1024)
        self.state_value2 = nn.Linear(1024,1024)
        self.state_value3 = nn.Linear(1024,1)
    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.convl(x))
        x = self.relu(self.convl(x))
        x = self.relu(self.convl(x))
        x = self.flatten(x)

        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))
        state_value = self.dropout(state_value)

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)
        output = state_value + (action_value - action_value.mean())

        return output
    
    def save_model(self, weights_filename='models/latest.pt'):
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(self.state_dict(), weights_filename)

    def load_model(self, weights_filenames='models/lates.pt'):
        try:
            self.load_state_dict(torch.load(weights_filenames))
            print(f"loaded weights at {weights_filenames}")
        except:
            print(f"No weights file available at {weights_filenames}")