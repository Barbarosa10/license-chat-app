import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)

        loss = -torch.sum(log_probs[range(targets.size(0)), targets]) / targets.size(0)
        return loss