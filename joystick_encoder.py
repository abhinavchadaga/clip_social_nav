import torch
from torch import nn


class JoyStickEncoder(nn.Module):
    def __init__(self) -> None:
        super(JoyStickEncoder, self).__init__()
        # define linear layers
        self.fc1 = nn.Linear(0, 0)
        self.fc2 = nn.Linear(0, 0)
        self.fc3 = nn.Linear(0, 0)

        # activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        pass
