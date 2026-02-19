import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits
        # targets: (B, H, W) class indices
        
        # Convert inputs to probs
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Flatten
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - Tversky
