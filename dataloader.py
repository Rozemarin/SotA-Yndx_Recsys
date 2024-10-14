from torch.utils.data import Dataset, DataLoader
import numpy as np


class TrainInteractionDataset(Dataset):
    def __init__(self, df, embedder, all_items=None, num_negatives=10):
        self.df = df
        self.embedder = embedder
        self.num_negatives = num_negatives
        self.user_pos_items = self.df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.all_items = all_items if all_items else set(self.df['item_id'].unique())

    def __len__(self):
        return len(self.df) * (1 + self.num_negatives)  # Общее количество примеров

    def _sample_negatives(self, user_id):
        pos_items = self.user_pos_items[user_id]
        neg_items = list(self.all_items - pos_items)
        neg_samples = np.random.choice(neg_items, size=self.num_negatives, replace=False)
        return neg_samples

    def __getitem__(self, idx):
        # Вычисляем индекс пользователя и типа взаимодействия
        user_index = idx // (1 + self.num_negatives)
        is_positive = idx % (1 + self.num_negatives) == 0

        user_id, pos_item_id = self.df.iloc[user_index][['user_id', 'item_id']].values

        if is_positive:
            # Положительный пример
            label = 1
            item_id = pos_item_id
        else:
            # Отрицательный пример
            label = 0
            neg_item_ids = self._sample_negatives(user_id)
            item_id = neg_item_ids[idx % (1 + self.num_negatives) - 1]

        user_embed = self.embedder.get_user_embeddings([user_id]).astype(np.float32).flatten()  # Shape: (embed_size,)
        item_embed = self.embedder.get_item_embeddings([item_id]).astype(np.float32).flatten()  # Shape: (embed_size,)

        return user_id, user_embed, item_id, item_embed, label


class TestInteractionDataset(Dataset):
    def __init__(self, df, embedder, user_ids, item_ids):
        self.df = df
        self.embedder = embedder
        self.user_ids = user_ids
        self.item_ids = item_ids

        self.pairs = [(user_id, item_id) for user_id in self.user_ids for item_id in self.item_ids]
        self.interactions = {(row['user_id'], row['item_id']): row['rating'] for _, row in df.iterrows()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_id, item_id = self.pairs[idx]

        user_embed = self.embedder.get_user_embeddings([user_id]).astype(np.float32).flatten()  # Shape: (embed_size,)
        item_embed = self.embedder.get_item_embeddings([item_id]).astype(np.float32).flatten()  # Shape: (embed_size,)

        label = self.interactions.get((user_id, item_id), 0)

        return user_id, user_embed, item_id, item_embed, label



def create_train_dataloader(df, embedder, num_negatives=10, batch_size=32):
    dataset = TrainInteractionDataset(df, embedder, num_negatives=num_negatives)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_test_dataloader(df, embedder, user_ids, item_ids, batch_size=32):
    dataset = TestInteractionDataset(df, embedder, user_ids, item_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)