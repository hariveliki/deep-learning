import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from get_data import get_data
from cnn import CNN
from train_eval import train


def k_fold_cross_validation(k=5, epochs=10, batch_size=64, seed=42):
    """
    Perform k-fold cross validation and return performance statistics.

    Args:
        k (int): Number of folds
        epochs (int): Number of training epochs per fold
        batch_size (int): Batch size for training
        seed (int): Random seed for reproducibility

    Returns:
        dict: Statistics including mean and std of accuracy/loss
    """
    # Get full dataset
    train_loader, _ = get_data(batch_size=batch_size, subset_size=5000, seed=seed)

    # Convert DataLoader to dataset
    full_dataset = train_loader.dataset

    # Calculate fold size
    fold_size = len(full_dataset) // k

    # Store results for each fold
    fold_results = []

    for fold in range(k):
        print(f"Training fold {fold+1}/{k}")

        # Create train/validation split for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        train_indices = list(range(0, val_start)) + list(
            range(val_end, len(full_dataset))
        )
        val_indices = list(range(val_start, val_end))

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )

        confs = [
            ("C", {"kernel": 3, "channels": 16, "batch_norm": True}),
            ("C", {"kernel": 3, "channels": 32, "batch_norm": True}),
            ("P", {"kernel": 2}),
            ("C", {"kernel": 3, "channels": 64, "batch_norm": True}),
            ("C", {"kernel": 3, "channels": 64, "batch_norm": True}),
            ("P", {"kernel": 2}),
            ("C", {"kernel": 3, "channels": 128, "batch_norm": True}),
            ("C", {"kernel": 3, "channels": 128, "batch_norm": True}),
            ("P", {"kernel": 2}),
            ("L", {"units": 500, "batch_norm": True, "dropout": 0.3}),
            ("L", {"units": 200, "batch_norm": True, "dropout": 0.1}),
        ]
        model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train model
        history = train(
            model,
            epochs=epochs,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Store final validation metrics
        fold_results.append(
            {
                "val_loss": history["valid_loss"][-1],
                "val_accuracy": history["valid_accuracy"][-1],
            }
        )

    # Calculate statistics
    val_losses = [r["val_loss"] for r in fold_results]
    val_accuracies = [r["val_accuracy"] for r in fold_results]

    stats = {
        "mean_val_loss": np.mean(val_losses),
        "std_val_loss": np.std(val_losses),
        "mean_val_accuracy": np.mean(val_accuracies),
        "std_val_accuracy": np.std(val_accuracies),
        "fold_results": fold_results,
    }

    return stats


# Run k-fold cross validation
stats = k_fold_cross_validation(k=5, epochs=10)

# Print results with confidence intervals
print("\n Confidence Results:")
print(f"Validation Loss: {stats['mean_val_loss']:.4f} ± {stats['std_val_loss']:.4f}")
print(
    f"Validation Accuracy: {stats['mean_val_accuracy']:.2f}% ± {stats['std_val_accuracy']:.2f}%"
)

# Plot results
plt.figure(figsize=(12, 5))

# Create two subplots
plt.subplot(1, 2, 1)

# Plot accuracy results
fold_numbers = range(1, len(stats["fold_results"]) + 1)
accuracies = [r["val_accuracy"] for r in stats["fold_results"]]
plt.plot(fold_numbers, accuracies, "bo-", label="Fold Accuracy")

# Plot accuracy mean and confidence interval
mean_acc = stats["mean_val_accuracy"]
std_acc = stats["std_val_accuracy"]
plt.axhline(y=mean_acc, color="r", linestyle="--", label="Mean Accuracy")
plt.fill_between(
    fold_numbers,
    mean_acc - std_acc,
    mean_acc + std_acc,
    color="r",
    alpha=0.2,
    label="±1 std dev",
)

plt.xlabel("Fold")
plt.ylabel("Validation Accuracy (%)")
plt.title("Cross Validation Accuracy")
plt.legend()
plt.grid(True)

# Plot loss results
plt.subplot(1, 2, 2)
losses = [r["val_loss"] for r in stats["fold_results"]]
plt.plot(fold_numbers, losses, "go-", label="Fold Loss")

# Plot loss mean and confidence interval
mean_loss = stats["mean_val_loss"]
std_loss = stats["std_val_loss"]
plt.axhline(y=mean_loss, color="r", linestyle="--", label="Mean Loss")
plt.fill_between(
    fold_numbers,
    mean_loss - std_loss,
    mean_loss + std_loss,
    color="r",
    alpha=0.2,
    label="±1 std dev",
)

plt.xlabel("Fold")
plt.ylabel("Validation Loss")
plt.title("Cross Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
