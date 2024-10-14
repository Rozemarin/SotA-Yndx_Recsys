from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch.nn as nn
import torch

from sklearn.decomposition import TruncatedSVD
from SASRec.SASRec_utils import build_sasrec_model, data_to_sequences, batch_generator


class BaseEmbedder(ABC):
    def __init__(self, algorithm_class, algorithm_kwargs=None):
        """
        Initializes the recommender with the algorithm class and its arguments.

        Args:
        algorithm_class: The class of the algorithm (e.g., TruncatedSVD, NMF, etc.)
        algorithm_kwargs: A dictionary of keyword arguments to pass to the algorithm's constructor
        """
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs if algorithm_kwargs else {}
        self.algorithm = None
        self.user_embeddings = None
        self.item_embeddings = None

    @abstractmethod
    def fit(self, train_df, explicit=False):
        """
        Abstract method to train the model.

        Args:
        train_df: A DataFrame containing user_id, item_id, and rating columns.
        explicit: Boolean, whether to treat the ratings as explicit or convert to binary (implicit).
        """
        pass

    @staticmethod
    def make_implicit(ratings):
        return (ratings > 0).astype(int)

    def get_user_embeddings(self, user_ids):
        """
        Returns the embeddings for the given list of user_ids.

        Args:
        user_ids: A list of user IDs.

        Returns:
        The corresponding user embeddings.
        """
        if self.user_embeddings is None:
            raise ValueError("User embeddings have not been computed. Run `fit()` first.")
        return self.user_embeddings[user_ids]

    def get_item_embeddings(self, item_ids):
        """
        Returns the embeddings for the given list of item_ids.

        Args:
        item_ids: A list of item IDs.

        Returns:
        The corresponding item embeddings.
        """
        if self.item_embeddings is None:
            raise ValueError("Item embeddings have not been computed. Run `fit()` first.")
        return self.item_embeddings[item_ids]

    @abstractmethod
    def predict(self, user_ids, item_ids):  # сюда надо передавать все item_ids
        """
        Predicts the interaction score (dot product) between user embeddings and item embeddings.

        Args:
        user_ids: A list of user IDs to predict for.
        item_ids: A list of item IDs to predict for.

        Returns:
        A dict {user_id: [list of predicted items sorted from best to worst]}
        """
        pass


class MatrixUserItemEmbedder(BaseEmbedder):
    def __init__(self, algorithm_class, algorithm_kwargs=None):
        super().__init__(algorithm_class, algorithm_kwargs)
        self.user_mapping = None
        self.item_mapping = None
        self.user_inverse_mapping = None
        self.item_inverse_mapping = None

    def fit(self, train_df, explicit=True):
        """
        Trains the model using the passed algorithm class.

        Args:
        train_df: A DataFrame containing user_id, item_id, and rating columns.
        explicit: Boolean, whether to treat the ratings as explicit or convert to binary (implicit).
        """
        self.user_mapping = {uid: i for i, uid in enumerate(train_df['user_id'].unique())}
        self.item_mapping = {iid: i for i, iid in enumerate(train_df['item_id'].unique())}
        
        self.user_inverse_mapping = {i: uid for uid, i in self.user_mapping.items()}
        self.item_inverse_mapping = {i: iid for iid, i in self.item_mapping.items()}

        user_idx = train_df['user_id'].map(self.user_mapping)
        item_idx = train_df['item_id'].map(self.item_mapping)

        if explicit:
            ratings = train_df['rating'].values
        else:
            ratings = self.make_implicit(train_df['rating'].values)

        interaction_matrix = csr_matrix((ratings, (user_idx, item_idx)))

        self.algorithm = self.algorithm_class(**self.algorithm_kwargs)
        latent_matrix = self.algorithm.fit_transform(interaction_matrix)
        self.user_embeddings = latent_matrix
        self.item_embeddings = self.algorithm.components_.T

    def get_user_embeddings(self, user_ids):
        """
        Returns the embeddings for the given list of user_ids.

        Args:
        user_ids: A list of user IDs.

        Returns:
        The corresponding user embeddings.
        """
        if self.user_embeddings is None:
            raise ValueError("User embeddings have not been computed. Run `fit()` first.")
        user_indices = [self.user_mapping[uid] for uid in user_ids]
        return self.user_embeddings[user_indices]

    def get_item_embeddings(self, item_ids):
        """
        Returns the embeddings for the given list of item_ids.

        Args:
        item_ids: A list of item IDs.

        Returns:
        The corresponding item embeddings.
        """
        if self.item_embeddings is None:
            raise ValueError("Item embeddings have not been computed. Run `fit()` first.")
        item_indices = [self.item_mapping[iid] for iid in item_ids]
        return self.item_embeddings[item_indices]

    def predict(self, user_ids, item_ids):
        """
        Predicts the interaction score (dot product) between user embeddings and item embeddings.

        Args:
        user_ids: A list of user IDs to predict for.
        item_ids: A list of item IDs to predict for.

        Returns:
        A dict {user_id: [list of predicted items sorted from best to worst]}
        """
        user_embs = self.get_user_embeddings(user_ids)
        item_embs = self.get_item_embeddings(item_ids)

        scores = np.dot(user_embs, item_embs.T)

        predictions = {}
        for i, user_id in enumerate(user_ids):
            user_scores = scores[i]
            sorted_items = sorted(zip(item_ids, user_scores), key=lambda x: x[1], reverse=True)
            predictions[user_id] = [item_id for item_id, score in sorted_items]

        return predictions

    
class SASRecUserItemEmbedding(BaseEmbedder):
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
                         )
                ):
        """
        Initializes the recommender with the algorithm class and its arguments.

        Args:
        algorithm_class: The class of the algorithm (e.g., TruncatedSVD, NMF, etc.)
        algorithm_kwargs: A dictionary of keyword arguments to pass to the algorithm's constructor
        """
        super().__init__(algorithm_class, algorithm_kwargs)
        # 
        self.computed_user_embs = set()

    def fit(self, train_df, explicit=True):
        """
        Abstract method to train the model.
        
        Args:
        train_df: A DataFrame containing user_id, item_id, and rating columns.
        explicit: Boolean, whether to treat the ratings as explicit or convert to binary (implicit).
        """
        self.user_mapping = {uid: i for i, uid in enumerate(train_df['user_id'].unique())}
        self.item_mapping = {iid: i for i, iid in enumerate(train_df['item_id'].unique())}
        
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
        self.item_embeddings = self.model.item_emb.weight.detach().numpy()
        # user embeddings
        training_seqs = data_to_sequences(train_df, self.data_description)
        sampler = batch_generator(
            training_seqs,
            batch_size = self.algorithm_kwargs['batch_size'],
            maxlen = self.algorithm_kwargs['maxlen'],
            pad_token = self.data_description['n_items']
        )
        dim = self.algorithm_kwargs['hidden_units']
        user_embeddings = np.zeros(shape=(0, dim))
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
        for batch in sampler:
            _, *seq_data = batch
            # convert batch data into torch tensors
            seq, pos = (torch.LongTensor(np.array(x)).to(device) for x in seq_data)
            with torch.no_grad():
                log_feats = self.model.log2feats(seq)
                final_feat = log_feats[:, -1, :]
                user_embeddings = np.append(user_embeddings, final_feat, axis=0)
        self.user_embeddings = user_embeddings
        
    def get_user_embeddings(self, user_ids):
        """
        Returns the embeddings for the given list of user_ids.

        Args:
        user_ids: A list of user IDs.

        Returns:
        The corresponding user embeddings.
        """
        if self.user_embeddings is None:
            raise ValueError("User embeddings have not been computed. Run `fit()` first.")
        user_indices = [self.user_mapping[uid] for uid in user_ids]
        return self.user_embeddings[user_indices]

    def get_item_embeddings(self, item_ids):
        """
        Returns the embeddings for the given list of item_ids.

        Args:
        item_ids: A list of item IDs.

        Returns:
        The corresponding item embeddings.
        """
        if self.item_embeddings is None:
            raise ValueError("Item embeddings have not been computed. Run `fit()` first.")
        item_indices = [self.item_mapping[iid] for iid in item_ids]
        return self.item_embeddings[item_indices]

    def predict(self, user_ids, item_ids):
        pass


if __name__ == "__main__":
    
    data = {
        'user_id': [1, 1, 2, 2, 3, 3, 3],
        'item_id': [1, 2, 1, 3, 2, 3, 4],
        'rating': [5, 4, 4, 5, 3, 2, 4],
        'timestamp': [1112486027, 1112484676, 1112484819, 1112484727, 1112484580, 1112485527, 1112486399]
    }

    df = pd.DataFrame(data)
    embedder = SASRecUserItemEmbedding()
    embedder.fit(df, explicit=True)
    
    print(embedder.get_user_embeddings([2]))
    print(embedder.get_item_embeddings([2]))
    user_ids = [1, 2, 3]
    all_items = df['item_id'].unique()

    predictions = embedder.predict(user_ids, all_items)