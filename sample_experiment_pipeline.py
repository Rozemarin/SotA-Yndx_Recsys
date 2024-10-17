import torch
import torch.nn as nn
import torch.optim as optim
from dataset_prep import get_train_val_test_from_dataset
from sklearn.decomposition import TruncatedSVD
from embedders import MatrixUserItemEmbedder
from dataloader import TrainDataloader, TestDataloader
from trainer import Trainer
from evaluator import Evaluator
from models.fm import FactorizationMachineModel
from models.neumf import NeuralFactorizationMachineModel
from models.mlp import MLP


device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_epochs = 5
embedding_vector_len = 64
batch_size = 128

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

train_loader = TrainDataloader(train_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, num_neg=127, device=device)
train_loader_for_metrics = TestDataloader(train_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, device=device)
val_loader_for_metrics = TestDataloader(val_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, device=device)
test_loader_for_metrics = TestDataloader(test_df, user_embeddings, item_embeddings, item_ids_whitelist=None, batch_size=batch_size, device=device)

print('----------SVD+FactorizationMachineModel(embedding_vector_len, embedding_vector_len)+BCEWL----------')

model = FactorizationMachineModel(embedding_vector_len, embedding_vector_len)
for global_epoch in range(global_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer.train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=1
    )
    
    recs, metrics = evaluator.evaluate(model, train_loader_for_metrics)
    print(f'Global epoch {global_epoch+1}/{global_epochs} Train:', metrics, '\n')
    
    recs, metrics = evaluator.evaluate(model, val_loader_for_metrics)
    print(f'Global epoch {global_epoch+1}/{global_epochs} Validation:', metrics, '\n')
    
recs, metrics = evaluator.evaluate(model, test_loader_for_metrics)
print(f'Test:', metrics) 

print('----------SVD+NeuralFactorizationMachineModel(embedding_vector_len, embedding_vector_len, mlp_dims=(256, 64), dropouts=(0.1, 0.1))+BCEWL----------')

model = NeuralFactorizationMachineModel(embedding_vector_len, embedding_vector_len, mlp_dims=(256, 64), dropouts=(0.1, 0.1)) 
for global_epoch in range(global_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer.train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=1
    )
    
    recs, metrics = evaluator.evaluate(model, train_loader_for_metrics)
    print(f'Global epoch {global_epoch+1}/{global_epochs} Train:', metrics, '\n')
    
    recs, metrics = evaluator.evaluate(model, val_loader_for_metrics)
    print(f'Global epoch {global_epoch+1}/{global_epochs} Validation:', metrics, '\n')
    
recs, metrics = evaluator.evaluate(model, test_loader_for_metrics)
print(f'Test:', metrics) 

print('----------SVD+MLP(input_dim=embedding_vector_len*2, embed_dims=[512], dropout=0.1)+BCEWL----------')

model = MLP(input_dim=embedding_vector_len*2, embed_dims=[512], dropout=0.1)
for global_epoch in range(global_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    trainer.train(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=1
    )
    
    recs, metrics = evaluator.evaluate(model, train_loader_for_metrics)
    print(f'Global epoch {global_epoch+1}/{global_epochs} Train:', metrics, '\n')
    
    recs, metrics = evaluator.evaluate(model, val_loader_for_metrics)
    print(f'Global epoch {global_epoch+1}/{global_epochs} Validation:', metrics, '\n')
    
recs, metrics = evaluator.evaluate(model, test_loader_for_metrics)
print(f'Test:', metrics) 
