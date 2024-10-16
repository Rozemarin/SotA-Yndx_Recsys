import torch
import torch.nn as nn
    

class DotProductWithTemperature(nn.Module):
    def __init__(self, temperature=20):
        super(DotProductWithTemperature, self).__init__()
        self.temperature = temperature

    def forward(self, user_emb, item_emb):
        dot_product = torch.sum(user_emb * item_emb, dim=-1)
        return dot_product / self.temperature