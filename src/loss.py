import torch.nn as nn
import numpy as np

# batch dice loss
def dice_loss(y_pred, y_true, smooth=1e-5):
    return 1 - (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = np.array([i ** 2. for i in range(4, 0, -1)][::-1])
        self.weights /= self.weights.sum()

    def forward(self, y_pred, y_true, smooth=1e-5):
        loss = 0
        # forr each deep supervision level
        for w, y in zip(self.weights, y_pred):
            y = nn.functional.interpolate(y, size=y_true[0][0].shape, mode='nearest')  # fill the image
            for c in range(len(y_true)):  # channel
                y_c = y[:, c, :, :, :]
                loss += w * nn.functional.binary_cross_entropy_with_logits(y_c, y_true[c], reduction='mean')
                loss += w * dice_loss(y_c, y_true[c], smooth)
            
        return loss