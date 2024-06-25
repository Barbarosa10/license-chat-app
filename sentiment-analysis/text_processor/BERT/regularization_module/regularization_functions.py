import torch

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, x, training=True):
        if not training:
            return x

        mask = (torch.rand(x.shape, device=x.device) > self.dropout_rate).float()
        return x * mask / (1 - self.dropout_rate)

class ReLU:
    def forward(self, x):
        return torch.relu(x)