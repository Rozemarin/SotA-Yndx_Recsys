import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    def __init__(self, print_info_after_iters: Optional[int] = 5000, plot_losses: bool = True, device: torch.device = None):
        self.print_info_after_iters = print_info_after_iters
        self.plot_losses = plot_losses
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, 
              model: nn.Module,
              dataloader,
              optimizer: optim.Optimizer,
              criterion: nn.Module,
              scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
              epochs: int = 3) -> nn.Module:
        
        model.to(self.device)
        model.train()  # Set the model to training mode
        
        # Store training history
        batch_losses = []
        batch_accuracies = []
        iter_losses = []
        iter_accuracies = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            running_loss, running_corrects, total_samples = 0.0, 0, 0

            for i, batch in enumerate(tqdm(dataloader)):
                user_id, user_emb, item_id, item_emb, labels = batch
                optimizer.zero_grad()  # Reset gradients

                # Forward pass
                outputs = model(user_emb, item_emb)
                loss = criterion(outputs.squeeze(), labels)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                # Track statistics
                preds = torch.round(torch.sigmoid(outputs))  # Assuming binary classification, threshold = 0.5
                corrects = torch.sum(preds.flatten() == labels).item()
                batch_loss = loss.item()
                batch_accuracy = corrects / len(labels)

                batch_losses.append(batch_loss)
                batch_accuracies.append(batch_accuracy)

                running_loss += batch_loss * len(labels)
                running_corrects += corrects
                total_samples += len(labels)

                if self.print_info_after_iters and (i + 1) % self.print_info_after_iters == 0:
                    avg_loss = running_loss / total_samples
                    avg_acc = running_corrects / total_samples
                    print(f"Iteration {i+1} | Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_acc:.4f}")
                    
                    # Store info after the specified number of iterations
                    iter_losses.append(avg_loss)
                    iter_accuracies.append(avg_acc)
                    running_loss, running_corrects, total_samples = 0.0, 0, 0  # Reset for the next iteration batch

        if self.plot_losses:
            self._plot_training_curves(batch_losses, iter_losses, batch_accuracies, iter_accuracies)

        return model

    def _plot_training_curves(self, 
                          batch_losses: List[float], 
                          iter_losses: List[float], 
                          batch_accuracies: List[float], 
                          iter_accuracies: List[float]) -> None:
        # Plot training losses and accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot losses
        ax1.plot(batch_losses, label="Batch losses", color="blue", alpha=0.5)
        if self.print_info_after_iters:
            ax1.plot(range(self.print_info_after_iters, len(batch_losses), self.print_info_after_iters), iter_losses, 
                     label="Averaged losses", color="red", linewidth=2)
        ax1.set_title("Loss during training")
        ax1.set_xlabel("Batches")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Plot accuracies
        ax2.plot(batch_accuracies, label="Batch accuracies", color="blue", alpha=0.5)
        if self.print_info_after_iters:
            ax2.plot(range(self.print_info_after_iters, len(batch_accuracies), self.print_info_after_iters), iter_accuracies, 
                     label="Averaged accuracies", color="red", linewidth=2)
        ax2.set_title("Accuracy during training")
        ax2.set_xlabel("Batches")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.show()