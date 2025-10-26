import torch.nn as nn
import torch.nn.functional as F

# Define the number of actions for Breakout (up, down, left, right or similar set)
# Assuming 4 actions from the original Keras code.
NUM_ACTIONS = 4

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture based on the DeepMind paper for Atari.
    Input state is a stack of 4 grayscale frames, with shape (N, 4, 84, 84).
    """
    def __init__(self, num_actions=NUM_ACTIONS):
        super(DQN, self).__init__()
        self.num_actions = num_actions

        # The input is (N, 4, 84, 84) - PyTorch uses channels-first (C, H, W)
        
        # Conv1: Input 4 channels, Output 32 channels, Kernel 8x8, Stride 4
        # Output shape: (N, 32, 20, 20) -> floor((84 - 8)/4) + 1 = 20
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        
        # Conv2: Input 32 channels, Output 64 channels, Kernel 4x4, Stride 2
        # Output shape: (N, 64, 9, 9) -> floor((20 - 4)/2) + 1 = 9
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        
        # Conv3: Input 64 channels, Output 64 channels, Kernel 3x3, Stride 1
        # Output shape: (N, 64, 7, 7) -> floor((9 - 3)/1) + 1 = 7
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions for the Flatten layer
        self.flatten_size = 64 * 7 * 7  # 3136
        
        # Fully Connected Layers
        # Dense 1 (FC1): Input 3136, Output 512, ReLU activation
        self.fc1 = nn.Linear(self.flatten_size, 512)
        
        # Dense 2 (FC2 - Output Layer): Input 512, Output num_actions, Linear activation
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        # Apply ReLU activation after each convolution
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output: (N, 64, 7, 7) -> (N, 3136)
        # -1 infers the batch size
        x = x.view(-1, self.flatten_size)
        
        # Apply ReLU activation after the first fully connected layer
        x = F.relu(self.fc1(x))
        
        # Output layer (Q-values - linear activation is default/desired here)
        q_values = self.fc2(x)
        
        return q_values

def create_q_model():
    """Helper function to instantiate the DQN model."""
    return DQN(num_actions=NUM_ACTIONS)