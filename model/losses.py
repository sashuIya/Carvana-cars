from torch import nn
from torch.nn import functional as F


class Loss:
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def __call__(self, y_pred, y_true):
        loss = self.nll_loss(y_pred, y_true)
        if self.dice_weight:
            eps = 1e-6
            intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
            union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1) + eps

            loss += 1.0 - (2 * intersection / union).mean()
        return loss
