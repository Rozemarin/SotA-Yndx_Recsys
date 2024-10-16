import torch
import torch.nn as nn
import torch.optim as optim
from dataset_prep import get_train_val_test_from_dataset
from sklearn.decomposition import TruncatedSVD
from embedders import MatrixUserItemEmbedder
from dataloader import TrainDataloader, TestDataloader
from trainer import Trainer
from evaluator import Evaluator
from models import MLP


device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_epochs = 3
embedding_vector_len = 64
batch_size = 512

trainer = Trainer(print_info_after_iters=3000, plot_losses=True)
evaluator = Evaluator(device)

train_df, val_df, test_df = get_train_val_test_from_dataset(dataset_name='ml-1m', 
                                                            pcore=5, 
                                                            max_seq_len=200, 
                                                            train_val_test_split_method='timestamp', 
                                                            train_val_test_ratio=[0.7, 0.2, 0.1], 
                                                            random_state=42)
embedder = MatrixUserItemEmbedder(
    algorithm_class=TruncatedSVD, 
    algorithm_kwargs={'n_components': embedding_vector_len, 'random_state': 42}, 
    device=device
)
embedder.fit(train_df, explicit=False)
user_embeddings = embedder.user_embeddings
item_embeddings = embedder.item_embeddings

train_loader = TrainDataloader(train_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, num_neg=100, device=device)
val_loader = TestDataloader(val_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, device=device)
test_loader = TestDataloader(test_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, device=device)

models = [MLP(input_dim=2 * embedding_vector_len), MLP(input_dim=2 * embedding_vector_len), MLP(input_dim=2 * embedding_vector_len)]
top_ks = [10000, 1000, 100]  # до скольки обрезаем айтемы после каждой модели
assert len(models) == len(top_ks)

for global_epoch in range(global_epochs):
    for j, model in enumerate(models):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        trainer.train(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=1
        )
        
        recs, metrics = evaluator.evaluate(model, val_loader)
        print(f'Global epoch {global_epoch+1}/{global_epochs}, model {j+1}/{len(models)} (val):', metrics, '\n')
        
        items = set()
        for lst in recs:
            items.update(lst[:top_ks[j]])
        
        train_loader.update_whitelist(items)
        val_loader.update_whitelist(items)
        test_loader.update_whitelist(items)
        
    train_loader.update_whitelist(None)
    val_loader.update_whitelist(None)
    test_loader.update_whitelist(None)
    
recs, metrics = evaluator.evaluate(models[-1], test_loader)
print(f'Test:', metrics) 