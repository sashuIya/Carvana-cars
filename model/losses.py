from torch import nn
from torch.nn import functional as F

def dice_loss(y_pred, y_true):
    smooth = 1.

    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    intersection = (y_pred_flat * y_true_flat).sum()

    return 1 - (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)

def bce_dice_loss(y_pred, y_true):
    return F.binary_cross_entropy(y_pred, y_true) + dice_loss(y_pred, y_true)
