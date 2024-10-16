import torch
import torch.nn as nn
    

class DotProduct(nn.Module):
    def __init__(self):
        super(DotProduct, self).__init__()

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return torch.sum(user_emb * item_emb, dim=1)