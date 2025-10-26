import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 4

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture based on the DeepMind paper for Atari.
    The input state is typically a stack of 4 grayscale frames, with shape (4, 84, 84).
    """
    def __init__(self, num_actions=NUM_ACTIONS):
        super(DQN, self).__init__()
        self.num_actions = num_actions

        # 1. Convolutions on the frames on the screen
        # Input shape: (N, 4, 84, 84) -> (batch_size, channels, height, width)
        
        # Conv1: Input 4 channels, Output 32 channels, Kernel 8x8, Stride 4
        # Output shape: (N, 32, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        
        # Conv2: Input 32 channels, Output 64 channels, Kernel 4x4, Stride 2
        # Output shape: (N, 64, 9, 9)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        
        # Conv3: Input 64 channels, Output 64 channels, Kernel 3x3, Stride 1
        # Output shape: (N, 64, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions for the Flatten layer
        # 64 channels * 7 * 7 = 3136
        self.flatten_size = 64 * 7 * 7  
        
        # 2. Fully Connected Layers
        # Dense 1: Input 3136, Output 512
        self.fc1 = nn.Linear(self.flatten_size, 512)
        
        # Dense 2: Input 512, Output num_actions (Q-values)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        # Apply ReLU activation after each convolution
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output of the convolutional layers
        x = x.view(-1, self.flatten_size)
        
        # Apply ReLU activation after the first fully connected layer
        x = F.relu(self.fc1(x))
        
        # Output layer (Q-values, uses linear activation by default in PyTorch Linear layer)
        q_values = self.fc2(x)
        
        return q_values

def create_q_model():
    """Helper function to instantiate the DQN model."""
    return DQN(num_actions=NUM_ACTIONS)

# Instantiate the models, equivalent to the Keras notebook setup
model = create_q_model()
model_target = create_q_model()

# Optional: Print the model structure to verify
# print(model)

# Optional: Example of how to use the models for prediction
# if __name__ == '__main__':
#     dummy_input = torch.randn(1, 4, 84, 84)  # Batch size 1, 4 channels, 84x84
#     q_values = model(dummy_input)
#     print(f"Output Q-values shape: {q_values.shape}") # Should be torch.Size([1, 4])