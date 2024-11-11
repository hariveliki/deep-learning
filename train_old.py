import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_base import CNN
from utils import get_data


def train(model, epochs, train_loader, criterion, optimizer, device=None):
    loss_train_hist = []
    train_accuracy_hist = []
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
            if device is not None:
                imgs = imgs.to(device)
                labels = labels.to(device)
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
        loss_train /= len(train_loader)
        loss_train_hist.append(loss_train)
        train_accuracy = 100 * correct / total
        train_accuracy_hist.append(train_accuracy)
        print(f"Train Loss: {loss_train}")
        print(f"Train Accuracy: {train_accuracy}")
    return loss_train_hist, train_accuracy_hist


def eval(model, valid_loader, criterion, device=None):
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
    return loss_valid, valid_accuracy