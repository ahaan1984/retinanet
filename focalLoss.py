import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha:float = 1.0, gamma:float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        CE = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE)
        FL = -self.alpha*(1-pt)**self.gamma*torch.log(pt)
        return FL
