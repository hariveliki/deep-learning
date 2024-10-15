import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_base import CNN
from utils import get_data


def train(model, epochs, train_loader, device, criterion, optimizer):
    loss_train_hist = []
    train_accuracy_hist = []
    print("Training length: ", len(train_loader))
    for epoch in range(epochs):
        print(
            f"|---------------------------| Start Epoch {epoch}: |---------------------------|"
        )
        loss_train = 0
        total = 0
        correct = 0
        model.train()
        for idx, (imgs, labels) in enumerate(train_loader):
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            if isinstance(labels, int):
                labels = torch.tensor([labels])
            # imgs = imgs.to(device)
            # labels = labels.to(device)
            # Forward pass
            labels = labels.long()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss_train += loss.item()
            preds = nn.functional.softmax(logits, dim=1)
            predicted = preds.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10_000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        loss_train /= len(train_loader)
        loss_train_hist.append(loss_train)
        train_accuracy = 100 * correct / total
        train_accuracy_hist.append(train_accuracy)
        print(f"Train Loss: {loss_train}")
        print(f"Train Accuracy: {train_accuracy}")
    return loss_train_hist, train_accuracy_hist


if __name__ == "__main__":
    confs = [
        ("C", {"kernel": 3, "channels": 16}),
        ("P", {"kernel": 2}),
        ("C", {"kernel": 3, "channels": 32}),
        ("P", {"kernel": 2}),
        ("L", {"units": 500}),
        ("L", {"units": 500}),
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
    # model.to(device)
    train_loader, valid_loader = get_data(batch_size=100, subset_size=100, seed=42)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_train, train_accuracy = train(
        model,
        epochs=200,
        train_loader=train_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
    )
    plt.plot(loss_train)
    plt.show()
