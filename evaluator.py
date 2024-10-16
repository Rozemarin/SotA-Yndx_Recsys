from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from tqdm.notebook import tqdm


@dataclass
class EvaluationMetrics:
    avg_hit_rates: Dict[str, float]
    avg_recall_scores: Dict[str, float]
    avg_mrr_scores: Dict[str, float]

        
class Evaluator:
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def evaluate(self, model: torch.nn.Module, dataloader, cutoffs: List[int] = [10, 50, 200, 500]) -> Tuple[List[int], EvaluationMetrics]:
        model = model.to(self.device)
        model.eval()  # Set model to evaluation mode

        all_user_ids = []
        all_item_ids = []
        all_scores = []

        # Collect predictions and ground truths
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                user_id, user_emb, item_id, item_emb = batch
            
                # Forward pass to get scores for each item
                scores = model(user_emb, item_emb)  # [batch_size, num_items]

                all_user_ids.extend(user_id.cpu().numpy())
                all_item_ids.extend(item_id.cpu().numpy())
                all_scores.extend(scores.cpu().numpy().flatten())

        df = pd.DataFrame({'user_id': all_user_ids, 'item_id': all_item_ids, 'score': all_scores}).sort_values('score', ascending=False)
        recommended_items = df.groupby('user_id')['item_id'].agg(list)

        g_df = dataloader.df[dataloader.df['user_id'].isin(df['user_id'])]
        ground_truth_items = g_df.groupby('user_id')['item_id'].agg(list)  
        
        # Return the calculated metrics
        return recommended_items, self.count_metrics(recommended_items.values, ground_truth_items.values, cutoffs)

    @staticmethod
    def hit_rate(recommended_items: List[int], ground_truth_items: List[int], n: int) -> int:
        """Calculate hit rate at top-N recommendations for a single user."""
        top_n_recommendations = recommended_items[:n]
        hits = len(set(top_n_recommendations) & set(ground_truth_items))
        return 1 if hits > 0 else 0

    @staticmethod
    def recall(recommended_items: List[int], ground_truth_items: List[int], n: int) -> float:
        """Calculate recall at top-N recommendations for a single user."""
        top_n_recommendations = recommended_items[:n]
        hits = len(set(top_n_recommendations) & set(ground_truth_items))
        total_relevant = n
        if total_relevant == 0:
            return 0
        return hits / total_relevant

    @staticmethod
    def mrr(recommended_items: List[int], ground_truth_items: List[int], n: int) -> float:
        """Calculate Mean Reciprocal Rank at top-N recommendations for a single user."""
        top_n_recommendations = recommended_items[:n]
        for rank, item in enumerate(top_n_recommendations, start=1):
            if item in ground_truth_items:
                return 1 / rank
        return 0

    def count_metrics(
        self,
        init_recommended_items: List[List[int]],
        init_ground_truth_items: List[List[int]],
        cutoffs: List[int]
    ) -> EvaluationMetrics:
        """
        Count the hit rate, recall, and MRR metrics across all users.
        """
        hit_rates = {n: [] for n in cutoffs}
        recall_scores = {n: [] for n in cutoffs}
        mrr_scores = {n: [] for n in cutoffs}

        # Loop through each user's recommendations and ground truth items
        for recommended_items, ground_truth_items in zip(init_recommended_items, init_ground_truth_items):
            for cutoff in cutoffs:
                hit_rates[cutoff].append(self.hit_rate(recommended_items, ground_truth_items, cutoff))
                recall_scores[cutoff].append(self.recall(recommended_items, ground_truth_items, cutoff))
                mrr_scores[cutoff].append(self.mrr(recommended_items, ground_truth_items, cutoff))

        # Calculate the average metrics across all users
        avg_hit_rates = {f'hit_rate_at_{n}': np.mean(hit_rates[n]) for n in cutoffs}
        avg_recall_scores = {f'recall_at_{n}': np.mean(recall_scores[n]) for n in cutoffs}
        avg_mrr_scores = {f'mrr_at_{n}': np.mean(mrr_scores[n]) for n in cutoffs}

        return EvaluationMetrics(
            avg_hit_rates=avg_hit_rates,
            avg_recall_scores=avg_recall_scores,
            avg_mrr_scores=avg_mrr_scores
        )