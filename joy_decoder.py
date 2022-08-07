import torch
from torch import nn
import pytorch_lightning as pl


class JoystickDecoder(pl.LightningModule):

    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super().__init__()
        # fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)

        # activation
        self.relu = nn.ReLU()
        # batch norm
        self.batch_norm1 = nn.BatchNorm1d(hidden1)
        self.batch_norm2 = nn.BatchNorm1d(hidden2)

    def forward(self, x):
        """ forward pass 

        Args: 
            x (Tensor): output of pre-trained lidar encoder. (batch_size, feature_size)

        Return:
            Tensor: joystick input prediction (batch_size, joystick_length, 3)
        """

        # input to hidden1
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        # hidden1 to hidden2
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu()

        # hidden2 to output
        x = self.fc3()
        # reshape to match joystick output
        x = torch.unfold(dim=1, size=3, step=3)
        return x

