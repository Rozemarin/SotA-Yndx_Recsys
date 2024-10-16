import torch
import random
import pandas as pd
import math


class TrainDataloader:
    def __init__(self, df: pd.DataFrame, user_embeddings: torch.nn.Embedding, 
                 item_embeddings: torch.nn.Embedding, item_ids_whitelist: set[int]=None, 
                 batch_size: int=64, num_neg: int=100, exclude_seen: bool=True, device: str='cpu'):
        self.df = df
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.item_ids_whitelist = item_ids_whitelist if item_ids_whitelist is not None else set(range(item_embeddings.num_embeddings))
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.exclude_seen = exclude_seen
        self.device = device
        self.user_item_interactions = self._build_user_item_interactions() if exclude_seen else None
        self.reset_iterator()

    def _build_user_item_interactions(self):
        user_item_dict = {}
        for _, row in self.df.iterrows():
            user_item_dict.setdefault(row.user_id, set()).add(row.item_id)
        return user_item_dict

    def reset_iterator(self):
        self.current_index = 0

    def __len__(self):
        # Calculate the total number of interactions and divide by batch size
        total_interactions = len(self.df) * (1 + self.num_neg)  # 1 positive + num_neg negatives per user
        return math.ceil(total_interactions / self.batch_size)

    def update_whitelist(self, new_whitelist: set[int]):
        if new_whitelist is None:
            self.item_ids_whitelist = set(range(item_embeddings.num_embeddings))
        else:
            self.item_ids_whitelist = new_whitelist
        self.reset_iterator()

    def __iter__(self):
        self.reset_iterator()  # Reset at the start of each new iteration
        batch = [[], [], [], [], []]
        len_batch = 0

        for i, row in self.df.iterrows():
            if row.item_id not in self.item_ids_whitelist:
                continue

            user_id = torch.tensor(row.user_id).to(self.device)
            user_emb = self.user_embeddings(user_id)

            pos_item_id = row.item_id

            if self.exclude_seen:
                interacted_items = self.user_item_interactions[row.user_id]
                # Exclude already seen items from the whitelist for negative sampling
                negative_sampling_pool = self.item_ids_whitelist - interacted_items
            else:
                negative_sampling_pool = self.item_ids_whitelist

            if len(negative_sampling_pool) < self.num_neg:
                neg_item_ids = list(negative_sampling_pool)
            else:
                neg_item_ids = random.sample(negative_sampling_pool, self.num_neg)

            item_ids = torch.tensor([pos_item_id] + list(neg_item_ids)).to(self.device)
            item_embs = self.item_embeddings(item_ids)

            labels = torch.tensor([1] + [0] * self.num_neg, dtype=torch.float32).to(self.device)

            for j, label in enumerate(labels):
                batch[0].append(user_id)
                batch[1].append(user_emb)
                batch[2].append(item_ids[j])
                batch[3].append(item_embs[j])
                batch[4].append(label)
                len_batch += 1

                if len_batch == self.batch_size:
                    yield (torch.stack(batch[0]),
                           torch.stack(batch[1]),
                           torch.stack(batch[2]),
                           torch.stack(batch[3]),
                           torch.stack(batch[4]))
                    batch = [[], [], [], [], []]
                    len_batch = 0

        if len_batch > 0:
            yield (torch.stack(batch[0]),
                   torch.stack(batch[1]),
                   torch.stack(batch[2]),
                   torch.stack(batch[3]),
                   torch.stack(batch[4]))


class TestDataloader:
    def __init__(self, df: pd.DataFrame, user_embeddings: torch.nn.Embedding, 
                 item_embeddings: torch.nn.Embedding, item_ids_whitelist: set[int]=None, 
                 batch_size: int=64, device: str='cpu'):
        self.df = df
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.item_ids_whitelist = item_ids_whitelist if item_ids_whitelist is not None else set(range(item_embeddings.num_embeddings))
        self.batch_size = batch_size
        self.device = device
        self.reset_iterator()

    def reset_iterator(self):
        self.current_index = 0

    def __len__(self):
        # Calculate the total number of interactions and divide by batch size
        total_interactions = len(self.df[['user_id']].drop_duplicates()) * len(self.item_ids_whitelist)  # all items for each user
        return math.ceil(total_interactions / self.batch_size)

    def update_whitelist(self, new_whitelist: set[int]):
        if new_whitelist is None:
            self.item_ids_whitelist = set(range(item_embeddings.num_embeddings))
        else:
            self.item_ids_whitelist = new_whitelist
        self.reset_iterator()

    def __iter__(self):
        self.reset_iterator()  # Reset at the start of each new iteration
        batch = [[], [], [], []]
        len_batch = 0

        for i, row in self.df[['user_id']].drop_duplicates().iterrows():
            user_id = torch.tensor(row.user_id).to(self.device)
            user_emb = self.user_embeddings(user_id)
            item_ids = torch.tensor(list(self.item_ids_whitelist), dtype=torch.int32).to(self.device)
            item_embs = self.item_embeddings(item_ids)

            for j, item_id in enumerate(item_ids):
                batch[0].append(user_id)
                batch[1].append(user_emb)
                batch[2].append(item_id)
                batch[3].append(item_embs[j])
                len_batch += 1

                if len_batch == self.batch_size:
                    # print(batch)
                    yield (torch.stack(batch[0]),
                           torch.stack(batch[1]),
                           torch.stack(batch[2]),
                           torch.stack(batch[3]))
                    batch = [[], [], [], []]
                    len_batch = 0

        if len_batch > 0:
            yield (torch.stack(batch[0]),
                   torch.stack(batch[1]),
                   torch.stack(batch[2]),
                   torch.stack(batch[3]))