import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class Trainer:
    def __init__(self, optimizer, optimizer_lr, criterion, device, log_interval=1000):
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval

    def train(self, model, train_dataloader, epochs=1, plot_losses=True):
        model = model.to(self.device)
        model.train()

        optimizer = self.optimizer(model.parameters(), lr=self.optimizer_lr)

        train_losses = []  # List to store average loss per epoch
        train_accuracies = []  # List to store accuracy per epoch
        iter_losses = []  # Track losses per iteration (batch)
        iter_accuracies = []  # Track accuracies per iteration (batch)

        for epoch in range(epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (user_id, user_embed, item_id, item_embed, labels) in enumerate(progress_bar):
                user_embed, item_embed, labels = (user_embed.to(self.device),
                                                  item_embed.to(self.device),
                                                  labels.float().to(self.device))
                optimizer.zero_grad()

                # Forward pass
                predictions = model(user_embed, item_embed).squeeze()
                loss = self.criterion(predictions, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.item()
                iter_losses.append(loss.item())

                # Calculate accuracy (binary classification example, adjust for your use case)
                predicted_classes = (torch.sigmoid(predictions) > 0.5).float()  # Assumes binary classification
                correct_predictions += (predicted_classes == labels).sum().item()
                total_predictions += labels.size(0)
                accuracy = correct_predictions / total_predictions

                iter_accuracies.append(accuracy)

                # Update progress bar with current loss and accuracy
                progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

                # Logging every few batches
                if (batch_idx + 1) % self.log_interval == 0:
                    avg_loss = running_loss / self.log_interval
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Avg Loss over last {self.log_interval} batches: {avg_loss:.4f}")
                    running_loss = 0.0

            # At the end of the epoch, log the average loss and accuracy
            avg_epoch_loss = sum(iter_losses[-len(train_dataloader):]) / len(train_dataloader)
            train_losses.append(avg_epoch_loss)
            epoch_accuracy = correct_predictions / total_predictions
            train_accuracies.append(epoch_accuracy)
            print(f"Epoch {epoch+1} completed: Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Plot metrics if plot_losses is True
        if plot_losses:
            self.plot_metrics(iter_losses, train_losses, iter_accuracies, train_accuracies)

        return model, train_losses, train_accuracies

    def plot_metrics(self, iter_losses, epoch_losses, iter_accuracies, epoch_accuracies):
        """Plots losses and accuracies side by side."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(range(1, len(iter_losses) + 1), iter_losses, label='Iteration Loss', color='blue')
        axs[0].plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Epoch Loss', color='red')
        axs[0].set_xlabel('Iterations / Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training Loss Over Iterations and Epochs')
        axs[0].legend()

        axs[1].plot(range(1, len(iter_accuracies) + 1), iter_accuracies, label='Iteration Accuracy', color='blue')
        axs[1].plot(range(1, len(epoch_accuracies) + 1), epoch_accuracies, label='Epoch Accuracy', color='red')
        axs[1].set_xlabel('Iterations / Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Training Accuracy Over Iterations and Epochs')
        axs[1].legend()

        plt.tight_layout()
        plt.show()