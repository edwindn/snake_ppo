import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class CNNNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        print(f'Initialized network with action dim {out_dim}')

        # Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten()
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim, in_dim)
            flattened_size = self.encoder(dummy).shape[1]

        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Actor head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )


    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.encoder(x)
        out = self.fc(features)
        return self.head(out)

    

class CNNActorCritic(nn.Module):
    def __init__(self, in_dim, action_dim, hidden_dim=256):
        super().__init__()
        print(f'Initialized network with action dim {action_dim}')

        # Convolutional Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten()
        )

        # Calculate flattened size (for 64x64 input)
        # flattened_size = 128 * (in_dim // 4) * (in_dim // 4)  # 16x16 after two stride=2
        
        # Dynamically determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_dim, in_dim)
            flattened_size = self.encoder(dummy).shape[1]

        # Shared fully connected layer
        self.fc_shared = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.encoder(x)
        shared = self.fc_shared(features)
        policy = self.actor(shared)
        value = self.critic(shared)
        return policy, value

    def get_value(self, x):
        features = self.encoder(x)
        shared = self.fc_shared(features)
        return self.critic(shared)
    



class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    

class Agent(nn.Module):
    def __init__(self, envs, input_dim, hidden_dim=64):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), self.critic(x) # also return entropy?
    

if __name__ == '__main__':
    input = torch.randn(1, 10, 10)
    actor = CNNNetwork(100, 3)
    output = actor(input)
    print(output.shape)
