import torch
import torch.nn as nn
from cnn import CNN
from torch.utils.data import DataLoader
from typing import List, Tuple, Union
from datasets import load_dataset
from tiny_dataset import TinyDataset


# def get_dim_after_conv(dim: int, conv_ksize: int, conv_stride=1, conv_padding=0) -> int:
#     return (dim - conv_ksize + 2 * conv_padding) // conv_stride + 1


# def get_dim_after_pool(
#     dim: int, pool_kernel_size: int, pool_stride=None, pool_padding=0
# ) -> int:
#     if pool_stride is None:
#         pool_stride = pool_kernel_size
#     return (dim - pool_kernel_size + 2 * pool_padding) // pool_stride + 1


# def get_dim_after_conv_and_pool(dim_init: int, confs: List[Tuple[str, dict]]):
#     dims = []
#     for n, (layer, conf) in enumerate(confs):
#         if n == 0 and layer == "C":
#             dim = get_dim_after_conv(
#                 dim=dim_init,
#                 conv_ksize=conf["kernel"],
#                 conv_stride=conf.get("stride", 1),
#                 conv_padding=conf.get("padding", 0),
#             )
#             dims.append(dim)
#         elif n != 0 and layer == "C":
#             dim = get_dim_after_conv(
#                 dim=dim,
#                 conv_ksize=conf["kernel"],
#                 conv_stride=conf.get("stride", 1),
#                 conv_padding=conf.get("padding", 0),
#             )
#             dims.append(dim)
#         elif n != 0 and layer == "P":
#             dim = get_dim_after_pool(dim=dim, pool_kernel_size=conf["kernel"])
#             dims.append(dim)
#     return dims[-1]


def get_data(batch_size: Union[int, None] = None):
    train_dataset = load_dataset("Maysee/tiny-imagenet", split="train")
    val_dataset = load_dataset("Maysee/tiny-imagenet", split="valid")
    train_data = TinyDataset(train_dataset)
    val_data = TinyDataset(val_dataset)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def train(model, epochs, train_loader, device, criterion, optimizer):
    for epoch in range(epochs):
        print(
            f"|---------------------------| Start Epoch {epoch}: |---------------------------|"
        )
        loss_train = 0
        total = 0
        correct = 0
        model.train()
        for (imgs, labels) in train_loader:
            print(imgs.shape)
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
        loss_train /= len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Train Loss: {loss_train}")
        print(f"Train Accuracy: {train_accuracy}")
        break
    return loss_train, train_accuracy


if __name__ == "__main__":
    confs = [
        ("C", {"kernel": 3, "channels": 16}),
        ("P", {"kernel": 2}),
        ("C", {"kernel": 3, "channels": 32}),
        ("P", {"kernel": 2}),
        ("L", {"units": 500}),
        ("L", {"units": 500}),
    ]
    model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
    train_loader, _ = get_data(batch_size=32)
    for (imgs, labels) in train_loader:
        print(imgs.shape)
        break
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    train(
        model,
        epochs=1,
        train_loader=train_loader,
        device=None,
        criterion=criterion,
        optimizer=optimizer,
    )
