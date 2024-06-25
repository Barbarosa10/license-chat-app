import torch
import torch.nn as nn

class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, inputs, targets):
        selected_log_probs = inputs[range(targets.size(0)), targets]
        loss = -torch.sum(selected_log_probs) / targets.size(0)
        
        return loss