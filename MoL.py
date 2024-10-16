import torch
import torch.nn as nn


class EmbeddingSubEncoder(nn.Module):
    def __init__(self, layer_sizes):
        super(EmbeddingSubEncoder, self).__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  
                layers.append(nn.ReLU())
                
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EmbeddingEncoder(nn.Module):
    def __init__(self, num_encoders=8, layer_sizes=[256, 64, 8]):
        super(EmbeddingEncoder, self).__init__()
        
        self.encoders = nn.ModuleList(
            [EmbeddingSubEncoder(layer_sizes) for _ in range(num_encoders)]
        )

    def forward(self, x):
        outputs = [encoder(x) for encoder in self.encoders]
        
        return torch.stack(outputs, dim=1)  


class LogitsMoL(nn.Module):
    def __init__(self):
        super(LogitsMoL, self).__init__()

    def forward(self, embedding1, embedding2):
        batch_size = embedding1.size(0)
        logits = torch.matmul(embedding1, embedding2.transpose(1, 2))  # [B, M, N]
        logits_vectorized = logits.view(batch_size, -1)
        
        return logits_vectorized


class MoLGatingFN(nn.Module):
    def __init__(self, layer_sizes):
        super(MoLGatingFN, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2: 
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, user_embedding, item_embedding, logits_vectorized):
        combined_input = torch.cat((user_embedding, item_embedding, logits_vectorized), dim=-1)
        x = self.network(combined_input)
        x = self.softmax(x) 
        
        return x
    
    
class MoLCombinedModel(nn.Module):
    def __init__(self, user_embedding_encoder, item_embedding_encoder, mol_gating_fn, logits_mol):
        super(MoLCombinedModel, self).__init__()
        self.user_embedding_encoder = user_embedding_encoder
        self.item_embedding_encoder = item_embedding_encoder
        self.mol_gating_fn = mol_gating_fn
        self.logits_mol = logits_mol

    def forward(self, user_emb, item_emb):
        user_enc_embs = self.user_embedding_encoder(user_emb)  
        item_enc_embs = self.item_embedding_encoder(item_emb)
        mol_logits = self.logits_mol(user_enc_embs, item_enc_embs) 
        gating_output = self.mol_gating_fn(user_emb, item_emb, mol_logits)  
        scalar_product = (gating_output * mol_logits).sum(dim=-1)
        # active = torch.sigmoid(scalar_product)
        
        return scalar_product
        # return active


