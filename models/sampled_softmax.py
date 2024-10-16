import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxBatchBCELoss(nn.Module):
    def __init__(self):
        super(SoftmaxBatchBCELoss, self).__init__()

    def forward(self, logits, targets):
        # Применяем softmax по всему батчу
        softmax_logits = F.softmax(logits, dim=0)
        
        # Вычисляем BCE с логитами после применения softmax
        loss = F.binary_cross_entropy(softmax_logits, targets)
        return loss

# # Пример использования
# criterion = SoftmaxBatchBCELoss()
# logits = torch.randn(10, 1)  # Пример батча с логитами
# targets = torch.randint(0, 2, (10, 1)).float()  # Бинарные целевые значения
# loss = criterion(logits, targets)
