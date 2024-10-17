import torch
import numpy as np

class HIndexer(nn.Module):
    def __init__(self, user_encoder, item_encoder, lamb, k):
        """
        :param user_encoder: Кодировщик пользователя (EmbeddingSubEncoder).
        :param item_encoder: Кодировщик товаров (EmbeddingSubEncoder).
        :param lamb: Количество случайных выборок для подсчета порога.
        :param k: Количество элементов для выбора.
        """
        super(HIndexer, self).__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.lamb = lamb
        self.k = k

    def forward(self, user_vector, item_vectors):
        """
        Применение h_indexer для одного пользователя и множества товаров.
        
        :param user_vector: Эмбеддинг пользователя.
        :param item_vectors: Эмбеддинги товаров.
        :return: Список индексов наиболее релевантных товаров.
        """
        # Кодируем эмбеддинги
        encoded_user = self.user_encoder(user_vector)  # [D]
        encoded_items = self.item_encoder(item_vectors)  # [N, D]
        
        # Преобразуем к numpy для использования h_indexer
        encoded_user_np = encoded_user.detach().cpu().numpy()
        encoded_items_np = encoded_items.detach().cpu().numpy()

        # Применение h_indexer
        relevant_indices = h_indexer(encoded_items_np, encoded_user_np, self.lamb, self.k)
        
        return relevant_indices

# Пример функции h_indexer
def h_indexer(data, query, lamb, k):
    rand_indices = np.random.choice(data.shape[0], lamb, replace=False)
    sampled_similarities = np.dot(data[rand_indices], query)
    threshold_idx = min(k * lamb // data.shape[0], lamb - 1)
    t = np.partition(sampled_similarities, threshold_idx)[threshold_idx]
    indices = [index for index, x in enumerate(data) if np.dot(x, query) > t]

    return indices
