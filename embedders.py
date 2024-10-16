from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import torch
from models import build_sasrec_model, data_to_sequences, batch_generator


class BaseEmbedder(ABC):
    def __init__(self, algorithm_class, algorithm_kwargs=None, freeze_embeddings=True, device='cpu'):
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs if algorithm_kwargs else {}
        self.algorithm = None
        self.freeze_emebddings = freeze_embeddings
        self.device = device
        self.user_embeddings = None
        self.item_embeddings = None

    @abstractmethod
    def fit(self, train_df, explicit=False):
        pass

    @staticmethod
    def make_implicit(ratings):
        return (ratings > 0).astype(int)

    def get_user_embeddings(self, user_ids):
        if self.user_embeddings is None:
            raise ValueError("User embeddings have not been computed. Run `fit()` first.")
        return self.user_embeddings(user_ids)

    def get_item_embeddings(self, item_ids):
        if self.item_embeddings is None:
            raise ValueError("Item embeddings have not been computed. Run `fit()` first.")
        return self.item_embeddings(item_ids)


class MatrixUserItemEmbedder(BaseEmbedder):
    def fit(self, train_df, explicit=False):
        user_idx = train_df['user_id']
        item_idx = train_df['item_id']

        if explicit:
            ratings = train_df['rating'].values
        else:
            ratings = self.make_implicit(train_df['rating'].values)
            
        interaction_matrix = csr_matrix((ratings, (user_idx, item_idx)))

        self.algorithm = self.algorithm_class(**self.algorithm_kwargs)

        latent_matrix = self.algorithm.fit_transform(interaction_matrix)
        self.user_embeddings = nn.Embedding.from_pretrained(torch.tensor(latent_matrix, dtype=torch.float32), freeze=self.freeze_emebddings).to(self.device)
        self.item_embeddings = nn.Embedding.from_pretrained(torch.tensor(self.algorithm.components_.T, dtype=torch.float32), freeze=self.freeze_emebddings).to(self.device)

class SASRecUserItemEmbedder(BaseEmbedder):
    def __init__(self, algorithm_class = None, 
                        algorithm_kwargs=
                         dict(
                            num_epochs = 20,
                            maxlen = 200,
                            hidden_units = 64,
                            dropout_rate = 0.4,
                            num_blocks = 1,
                            num_heads = 1,
                            batch_size = 64,
                            sampler_seed = 99,
                            manual_seed = 111,
                            learning_rate = 1e-3,
                            l2_emb = 0,
                         ),
                         device='cpu'
                ):
        super().__init__(algorithm_class, algorithm_kwargs, device=device)
    
    def fit(self, train_df, explicit=False):
        self.data_description = dict(
            users = 'user_id',
            items = 'item_id',
            order = 'timestamp',
            n_users = len(np.unique(train_df['user_id'])),
            n_items = len(np.unique(train_df['item_id']))
        )
        # fit
        self.model, self.losses = build_sasrec_model(self.algorithm_kwargs, train_df, self.data_description)
        # item embeddings
        self.item_embeddings = self.model.item_emb
        self.item_embeddings.requires_grad = False
        # user embeddings
        training_seqs = data_to_sequences(train_df, self.data_description)
        sampler = batch_generator(
            training_seqs,
            batch_size = self.algorithm_kwargs['batch_size'],
            maxlen = self.algorithm_kwargs['maxlen'],
            pad_token = self.data_description['n_items']
        )
        dim = self.algorithm_kwargs['hidden_units']
        
        user_embeddings_list = []
        
        for batch in sampler:
            _, *seq_data = batch
            seq, pos = (torch.LongTensor(seq_data[i]).to(self.device) for i in range(len(seq_data)))
        
            with torch.no_grad():
                log_feats = self.model.log2feats(seq)
                final_feat = log_feats[:, -1, :]  # Extract the last feature for each sequence
                user_embeddings_list.append(final_feat)
    
        user_embeddings = torch.cat(user_embeddings_list, dim=0)
        
        self.user_embeddings = nn.Embedding.from_pretrained(user_embeddings, freeze=self.freeze_emebddings).to(self.device)
