import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from get_data import get_data
from cnn import CNN
from l1_l2 import train
from torchsummary import summary
from plot_confidence import plot_confidence
from plot_train import plot_train
from sklearn.model_selection import StratifiedKFold

K = 5
BATCH_SIZE = 64
EPOCHS = 10
SEED = 42


def k_fold_cross_validation(k=5, epochs=10, batch_size=64, seed=42):
    train_loader, _ = get_data(batch_size=batch_size, seed=seed)
    full_dataset = train_loader.dataset
    
    # Get all targets for stratification
    targets = [y for _, y in full_dataset]
    
    # Create stratified k-fold with custom split
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    
    # Generate indices for custom split (e.g., 90% train, 10% validation)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        # Convert numpy indices to regular Python integers and ensure they're integers
        train_idx = [int(i) for i in train_idx]
        val_idx = [int(i) for i in val_idx]
        
        # Use smaller validation set (ensure all operations maintain integer indices)
        val_idx = val_idx[:len(val_idx)]
        train_idx = [int(i) for i in np.concatenate([train_idx, val_idx[len(val_idx):]])]  # Convert to integers explicitly
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )
        print("Training size:", len(train_loader.dataset))
        print("Validation size:", len(val_loader.dataset))
        confs = [
            ("C", {"kernel": 3, "channels": 16, "padding": 1, "batch_norm": True}),
            ("C", {"kernel": 3, "channels": 32, "padding": 1, "batch_norm": True}),
            ("P", {"kernel": 2}),
            ("C", {"kernel": 3, "channels": 64, "padding": 1, "batch_norm": True}),
            ("C", {"kernel": 3, "channels": 64, "padding": 1, "batch_norm": True}),
            ("P", {"kernel": 2}),
            ("C", {"kernel": 3, "channels": 128, "padding": 1, "batch_norm": True}),
            ("C", {"kernel": 3, "channels": 128, "padding": 1, "batch_norm": True}),
            ("P", {"kernel": 2}),
            ("L", {"units": 500, "dropout": 0.4}),
            ("L", {"units": 200, "dropout": 0.1}),
        ]
        model = CNN(
            dim=64,
            num_classes=200,
            confs=confs,
            in_channels=3,
            weight_init="kaiming",
        )
        summary(model, (3, 64, 64))
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        history = train(
            model,
            epochs=epochs,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        # save plot after each fold with fold number
        plot_train(history, fold=fold+1)
        fold_results.append(
            {
                "val_loss": history["valid_loss"][-1],
                "val_accuracy": history["valid_accuracy"][-1],
            }
        )
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


stats = k_fold_cross_validation(k=K, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED)
print("\n Confidence Results:")
print(f"Validation Loss: {stats['mean_val_loss']:.4f} ± {stats['std_val_loss']:.4f}")
print(
    f"Validation Accuracy: {stats['mean_val_accuracy']:.2f}% ± {stats['std_val_accuracy']:.2f}%"
)
plot_confidence(stats)
