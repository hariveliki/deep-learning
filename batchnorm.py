import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_data
from cnn import CNN


def train(
    model,
    epochs,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device=None,
    l1_lambda=0,
    l2_lambda=0,
):
    loss_train_hist = []
    loss_valid_hist = []
    train_accuracy_hist = []
    valid_accuracy_hist = []

    for epoch in range(epochs):
        print(
            f"|---------------------------| Start Epoch {epoch}: |---------------------------|"
        )

        # Training phase
        model.train()
        loss_train = 0
        total = 0
        correct = 0

        for imgs, labels in train_loader:
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            if isinstance(labels, int):
                labels = torch.tensor([labels])
            if device is not None:
                imgs = imgs.to(device)
                labels = labels.to(device)

            # Forward pass
            labels = labels.long()
            logits = model(imgs)
            loss = criterion(logits, labels)

            # Add L1 regularization
            if l1_lambda > 0:
                l1_reg = torch.tensor(0.0, requires_grad=True)
                for param in model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1)
                loss = loss + l1_lambda * l1_reg

            # Add L2 regularization
            if l2_lambda > 0:
                l2_reg = torch.tensor(0.0, requires_grad=True)
                for param in model.parameters():
                    l2_reg = l2_reg + torch.norm(param, 2)
                loss = loss + l2_lambda * l2_reg

            loss_train += loss.item()
            preds = nn.functional.softmax(logits, dim=1)
            predicted = preds.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train /= len(train_loader)
        train_accuracy = 100 * correct / total

        # Evaluation phase
        model.eval()
        loss_valid = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(0)
                if isinstance(labels, int):
                    labels = torch.tensor([labels])
                if device is not None:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                labels = labels.long()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss_valid += loss.item()
                preds = nn.functional.softmax(logits, dim=1)
                predicted = preds.argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss_valid /= len(valid_loader)
        valid_accuracy = 100 * correct / total

        # Store metrics
        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)
        train_accuracy_hist.append(train_accuracy)
        valid_accuracy_hist.append(valid_accuracy)

        # Print metrics
        print(f"Train Loss: {loss_train:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Valid Loss: {loss_valid:.4f} | Valid Accuracy: {valid_accuracy:.2f}%")

    return {
        "train_loss": loss_train_hist,
        "valid_loss": loss_valid_hist,
        "train_accuracy": train_accuracy_hist,
        "valid_accuracy": valid_accuracy_hist,
    }


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, valid_loader = get_data(batch_size=64, subset_size=5000, seed=42)
    criterion = nn.CrossEntropyLoss()
    results = {}
    batch_confs = [
        {"name": "Batchnorm", "batch_norm": True},
        {"name": "No Batchnorm", "batch_norm": False},
    ]
    for conf in batch_confs:
        confs = [
            ("C", {"kernel": 3, "channels": 16, "batch_norm": conf["batch_norm"]}),
            ("C", {"kernel": 3, "channels": 32, "batch_norm": conf["batch_norm"]}),
            ("P", {"kernel": 2}),
            ("C", {"kernel": 3, "channels": 64, "batch_norm": conf["batch_norm"]}),
            ("C", {"kernel": 3, "channels": 64, "batch_norm": conf["batch_norm"]}),
            ("P", {"kernel": 2}),
            ("C", {"kernel": 3, "channels": 128, "batch_norm": conf["batch_norm"]}),
            ("C", {"kernel": 3, "channels": 128, "batch_norm": conf["batch_norm"]}),
            ("P", {"kernel": 2}),
            ("L", {"units": 500, "batch_norm": conf["batch_norm"]}),
            ("L", {"units": 200, "batch_norm": conf["batch_norm"]}),
        ]
        model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        model.to(device)
        history = train(
            model,
            epochs=5,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            l1_lambda=0,
            l2_lambda=0,
        )
        results[conf["name"]] = history
    print(results)
