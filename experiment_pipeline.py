import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from dataset_prep import *
from embedders import *
from dataloader import * 
from models.mlp import MLP
from trainer import *
from evaluator import *
from topk_cutter import *


if __name__ == '__main__':
    prep = PrepareMovielens(dataset='ml-1m')
    prep.download_dataset()
    prep.preprocess_ratings_df(pcore=220, max_seq_len=200)

    # Step 2: Split the dataset into train, validation, and test sets
    train_df, val_df, test_df = prep.train_test_split(method='timestamp', random_state=42)

    # Step 3: Initialize and fit the embedding model
    embedder = MatrixUserItemEmbedder(
        algorithm_class=TruncatedSVD,
        algorithm_kwargs={'n_components': 20, 'random_state': 42}
    )
    embedder.fit(train_df, explicit=False)

    # Step 4: dataloaders
    train_dataloader = create_train_dataloader(train_df, embedder, num_negatives=3, batch_size=32)
    all_items = train_dataloader.dataset.all_items
    val_dataloader = create_test_dataloader(val_df, embedder, val_df.user_id.unique(), all_items)
    test_dataloader = create_test_dataloader(test_df, embedder, test_df.user_id.unique(), all_items)

    model1 = MLP(40)
    model2 = MLP(40)

    criterion = nn.BCELoss()  # sigmoid считается в трейне
    optimizer_lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(optimizer=optim.Adam, optimizer_lr=optimizer_lr, criterion=criterion, device=device)
    
    model1, _, _ = trainer.train(model1, train_dataloader, epochs=1, plot_losses=True)
    evaluator = Evaluator(device=device)
    recommended_items, metrics = evaluator.evaluate(model1, val_dataloader)
    print(metrics)
    cutter_1000 = Cutter(1000)

    train_df = cutter_1000.cut_df(train_df, recommended_items.values())
    
    train_dataloader = create_train_dataloader(train_df, embedder, num_negatives=3, batch_size=32)
    all_items = train_dataloader.dataset.all_items
    val_dataloader = create_test_dataloader(val_df, embedder, val_df.user_id.unique(), all_items)
    test_dataloader = create_test_dataloader(test_df, embedder, test_df.user_id.unique(), all_items)

    model2, _, _ = trainer.train(model2, train_dataloader, epochs=1, plot_losses=True)

    recommended_items, metrics = evaluator.evaluate(model2, val_dataloader)
    print(metrics)
    