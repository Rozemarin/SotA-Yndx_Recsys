import torch
import numpy as np
from tqdm.notebook import tqdm

class Evaluator:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def hit_rate(recommended_items, ground_truth_items, n):
        """Calculate hit rate at top-N recommendations for a single user."""
        top_n_recommendations = recommended_items[:n]
        hits = len(set(top_n_recommendations) & set(ground_truth_items))
        return 1 if hits > 0 else 0

    @staticmethod
    def recall(recommended_items, ground_truth_items, n):
        """Calculate recall at top-N recommendations for a single user."""
        top_n_recommendations = recommended_items[:n]
        hits = len(set(top_n_recommendations) & set(ground_truth_items))
        total_relevant = len(ground_truth_items)
        if total_relevant == 0:
            return 0
        return hits / total_relevant

    @staticmethod
    def mrr(recommended_items, ground_truth_items, n):
        """Calculate Mean Reciprocal Rank at top-N recommendations for a single user."""
        top_n_recommendations = recommended_items[:n]
        for rank, item in enumerate(top_n_recommendations, start=1):
            if item in ground_truth_items:
                return 1 / rank
        return 0

    def count_metrics(self, init_recommended_items, init_ground_truth_items, cutoffs):
        hit_rates = {n: [] for n in cutoffs}
        recall_scores = {n: [] for n in cutoffs}
        mrr_scores = {n: [] for n in cutoffs}

        for recommended_items, ground_truth_items in zip(init_recommended_items, init_ground_truth_items):
            for cutoff in cutoffs:
                hit_rates[cutoff].append(self.hit_rate(recommended_items, ground_truth_items, cutoff))
                recall_scores[cutoff].append(self.recall(recommended_items, ground_truth_items, cutoff))
                mrr_scores[cutoff].append(self.mrr(recommended_items, ground_truth_items, cutoff))

        avg_hit_rates = {f'hit_rate_at_{n}': np.mean(hit_rates[n]) for n in cutoffs}
        avg_recall_scores = {f'recall_at_{n}': np.mean(recall_scores[n]) for n in cutoffs}
        avg_mrr_scores = {f'mrr_at_{n}': np.mean(mrr_scores[n]) for n in cutoffs}

        return {
            "avg_hit_rates": avg_hit_rates,
            "avg_recall_scores": avg_recall_scores,
            "avg_mrr_scores": avg_mrr_scores
        }

    def evaluate(self, model, dataloader, cutoffs=[10, 50, 200, 500]):
        # Assume they go in the same order because of the initialization process
        ground_truth_items = {}  # user_id: [item_ids]
        recommended_items = {}  # user_id: [(item_id, score)]

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for user_id, user_embed, item_id, item_embed, label in tqdm(dataloader):
                user_embed, item_embed = (user_embed.to(self.device),
                                          item_embed.to(self.device))

                predictions = model(user_embed, item_embed).squeeze()

                for i in range(len(user_id)):
                    ui = int(user_id[i].cpu().numpy())
                    ii = int(item_id[i].cpu().numpy())
                    score = predictions[i].item()

                    if ui not in ground_truth_items:
                        ground_truth_items[ui] = []
                    if label == 1:
                        ground_truth_items[ui].append(ii)

                    if ui not in recommended_items:
                        recommended_items[ui] = []
                    recommended_items[ui].append((ii, score))

        for ui in recommended_items:
            recommended_items[ui].sort(key=lambda x: x[1], reverse=True)

        recommended_items = {ui: [item[0] for item in items] for ui, items in recommended_items.items()}
        metrics = self.count_metrics(recommended_items.values(), ground_truth_items.values(), cutoffs)
        return recommended_items, metrics