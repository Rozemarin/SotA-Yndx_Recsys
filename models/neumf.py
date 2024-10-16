import torch


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if necessary

        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square

        if self.reduce_sum:
            ix = torch.sum(ix, dim=0, keepdim=True)  # Keep dim for consistency with linear part
            ix = ix.unsqueeze(0)
        return 0.5 * ix 


class MLP_nmf(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropouts):
        super().__init__()
        layers = []
        for i, dim in enumerate(mlp_dims):
            layers.append(torch.nn.Linear(input_dim, dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropouts[i]))
            input_dim = dim
        layers.append(torch.nn.Linear(mlp_dims[-1], 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class NeuralFactorizationMachineModel(torch.nn.Module):
    def __init__(self, user_embed_dim, item_embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.fm = FactorizationMachine(reduce_sum=False)
        self.linear = torch.nn.Linear(user_embed_dim + item_embed_dim, 1)
        self.mlp = MLP_nmf(user_embed_dim + item_embed_dim, mlp_dims, dropouts)

    def forward(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=1) 
        # Linear output
        linear_output = self.linear(x) 
        # FM output
        fm_output = self.fm(x).reshape(-1, 1)
        # MLP_nmf output
        mlp_output = self.mlp(x)  # Shape: [batch_size, 1]
        # Combine outputs
        output = linear_output + fm_output + mlp_output
        return output.squeeze(1)  # Remove extra dimension if necessary