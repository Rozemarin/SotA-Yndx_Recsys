import torch
import torch.nn as nn


class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims=[128], dropout=0.2, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(x)
    