import torch


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        # Ensure x has at least two dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if necessary
        
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        
        # If reducing sum, output shape will be (batch_size, 1)
        if self.reduce_sum:
            ix = torch.sum(ix, dim=0, keepdim=True)  # Keep dim for consistency with linear part
            ix = ix.unsqueeze(0)
        return 0.5 * ix 


class FactorizationMachineModel(torch.nn.Module):
    def __init__(self, user_embed_dim, item_embed_dim):
        super().__init__()
        self.user_embed_dim = user_embed_dim
        self.item_embed_dim = item_embed_dim
        self.linear = torch.nn.Linear(user_embed_dim + item_embed_dim, 1)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=1)
        linear_output = self.linear(x)
        fm_output = self.fm(x)
        output = linear_output + fm_output  
        return output.squeeze(1)
