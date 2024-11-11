#!/usr/bin/env python
# coding: utf-8

# # Problem Description
# Tiny ImageNet contains 100'000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images, 50 validation images and 50 test images.

# # Metrics
# For the evaluation of the model, we will use accuracy as our metric. It is straightforward and defined as follows:
# $$ \text{Accuracy} = \frac{\text{correct classifications}}{\text{all classifications}} $$
# 
# However, accuracy has a disadvantage for multiclass classification problems, as it does not consider class imbalances. If our model is biased towards one class, and that class has the highest occurrence, accuracy may fail to reflect this bias. In our case, since the dataset does not have class imbalances, accuracy should be sufficient for our evaluation.
# 
# To estimate the error in the chosen metric, we could also consider using an alternative metric like the F1 Score, which penalizes false predictions rather than just summarizing the correct ones.
# 

# # Base Architecture
# - The base model consists of two convolutional layers for feature extraction and two pooling layers to reduce the spatial dimension of the image. Two fully connected layer ensure enough parameters. The goal is to train with a single sample or batch and to show that it works as well as in the next step to find a proper learning rate and batch size.

import torch
import torch.nn as nn
from torchsummary import summary
import utils
from typing import List, Tuple, Dict


class CNN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        confs: List[Tuple[str, Dict]],
        in_channels: int,
        weight_init=None,
    ):
        super(CNN, self).__init__()
        self.net = nn.ModuleList()
        self.weight_init = weight_init

        # Split configurations
        linear_idxs = [idx for idx, (layer, _) in enumerate(confs) if layer == "L"]
        linear_start = linear_idxs[0]
        convolution_conf = confs[:linear_start]
        linear_conf = confs[linear_start:]

        # Process convolution layers
        current_channels = in_channels
        for layer, conf in convolution_conf:
            if layer == "C":
                print(f"Creating Conv2d: in_channels={current_channels}, out_channels={conf['channels']}")
                self.net.append(
                    nn.Conv2d(
                        current_channels,
                        out_channels=conf["channels"],
                        kernel_size=conf["kernel"],
                        stride=conf.get("stride", 1),
                        padding=conf.get("padding", 0),
                    )
                )
                self.net.append(nn.ReLU())
                if conf.get("batch_norm", False):
                    self.net.append(nn.BatchNorm2d(conf["channels"]))
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))
                current_channels = conf["channels"]
            elif layer == "P":
                self.net.append(nn.MaxPool2d(kernel_size=conf["kernel"]))

        # Calculate dimensions after convolutions
        self.dim = utils.get_dim_after_conv_and_pool(dim_init=dim, confs=convolution_conf)
        
        # Process linear layers
        for idx, (layer, conf) in enumerate(linear_conf):
            if idx == 0:
                self.net.append(nn.Flatten())
                self.net.append(
                    nn.Linear(self.dim * self.dim * current_channels, conf["units"])
                )
                self.net.append(nn.ReLU())
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))
            elif idx == len(linear_conf) - 1:
                self.net.append(nn.Linear(conf["units"], num_classes))
            else:
                self.net.append(nn.Linear(conf["units"], conf["units"]))
                self.net.append(nn.ReLU())
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))

        if self.weight_init is not None:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.weight_init == "random":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x comes in as [batch_size, height, width, channels] or [batch_size, channels, height, width]
        if x.shape[1] != 3:  # If channels are not in the correct position
            # print(f"Input shape before permute: {x.shape}")
            x = x.permute(0, 3, 1, 2)  # Change from (N,H,W,C) to (N,C,H,W)
            # print(f"Input shape after permute: {x.shape}")
        
        # Now x should be [batch_size, channels, height, width]
        assert x.shape[1] == 3, f"Expected 3 channels in dimension 1, got shape {x.shape}"

        for layer in self.net:
            x = layer(x)

        return x


confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
]


x = torch.rand(10, 64, 64, 3)
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
model(x)
summary(model, (64, 64, 3))


train_loader, valid_loader = utils.get_data(batch_size=None)
print(train_loader, valid_loader)


# ## Discussion

# # SGD, Tuning of Learning Rate and Batch Size
# - Stochastic Gradient Descent (SGD) is an optimization method. Unlike traditional gradient descent, where we train on the entire dataset, SGD updates model parameters using a small batch of the data. This enables faster learning, say faster convergence to local (global) minima, and is therefore more efficient.
# - The learning rate controls how much the model's weights are adjusted in response to the error at each step of training. A small learning rate increases the training time, while a large learning rate speeds up training but can cause the model to overshoot and fail to converge.
# - The batch size refers to the number of training examples used for updating the model's weights. The key idea is that using smaller groups to update the weights will also allow the model to generalize well, as the batch acts as a proxy for the whole dataset.
# 
# First, we'll conduct an overfitting test by training on a small dataset of 100 samples, expecting the loss to decrease and accuracy to increase. Then, we'll search for the highest learning rate that still allows convergence. Finally, we'll determine a batch size that is memory efficient and balances representation of the data statistics with minimizing noise.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn import CNN
from utils import get_data


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


def train(model, epochs, train_loader, valid_loader, criterion, optimizer, device=None):
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
        valid_loss, valid_accuracy = eval(model, valid_loader, criterion, device)

        # Store metrics
        loss_train_hist.append(loss_train)
        loss_valid_hist.append(valid_loss)
        train_accuracy_hist.append(train_accuracy)
        valid_accuracy_hist.append(valid_accuracy)

        # Print metrics
        print(f"Train Loss: {loss_train:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_accuracy:.2f}%")

    return {
        "train_loss": loss_train_hist,
        "valid_loss": loss_valid_hist,
        "train_accuracy": train_accuracy_hist,
        "valid_accuracy": valid_accuracy_hist,
    }


confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
]
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
train_loader, valid_loader = get_data(batch_size=100, subset_size=100, seed=42)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_train, train_accuracy = train(
    model,
    epochs=200,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=None,
)
# Plot training loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_train)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()


# TODO evaluate the model on the validation set
confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
]
device = torch.device("mps" if torch.backends.mps.is_available() else None)
criterion = nn.CrossEntropyLoss()
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [32, 64]
results = {}
epochs = 10
for lr in learning_rates:
    print(f"Learning rate: {lr}")
    for bs in batch_sizes:
        print(f"Batch size: {bs}")
        train_loader, valid_loader = get_data(batch_size=bs, seed=42)
        model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
        if device is not None:
            model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss_train, train_accuracy = train(
            model,
            epochs=epochs,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        results[(lr, bs)] = (loss_train, train_accuracy)


plt.figure(figsize=(20, 30))  # Increased figure size to accommodate more subplots
for i, ((lr, bs), (loss, accuracy)) in enumerate(results.items()):
    # Plot Training Loss
    plt.subplot(len(results), 2, 2*i + 1)
    plt.plot(loss)
    plt.title(f'Training Loss (LR={lr}, BS={bs})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot Training Accuracy
    plt.subplot(len(results), 2, 2*i + 2)
    plt.plot(accuracy)
    plt.title(f'Training Accuracy (LR={lr}, BS={bs})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()


# ## Discussion
# Our overfitting test with SGD and 100 samples (full batch) shows that the algorithm works. It convergences as expected and the accuracy increases up to 100% and shows clearly overfitting.
# 
# For the right learning rate and batch size, there are two candidates which show a proper convergence and a high increase in accuracy:
# 
#   - LR=0.001 and BS=32
# 
#     Enables more frequent gradient updates and to capture the statistics of a smaller batch size. In doing so, we hope to reduce the risk of overfitting on the training data.
#     
#   - LR=0.01 and BS=64
#   
#     Enables faster model training because a higher learning rate results in larger steps during weight updates, say speeding up convergence. A larger batch size reduces the number of gradient updates per epoch, which can make training more efficient, though it may also affect the model’s ability to generalize.

# # SGD, Weight Initialization, Model Complexity, Convolution Settings
# - `TODO` Explain what you will do here

# ## Weight Initialization `TODO`
# 1. **Kaiming Initialization**:
#    - Weights are drawn from a normal distribution.
#    - Variance is scaled by $\frac{2}{n}$, where $n$ is the number of input units in the layer.
#    - This scaling helps prevent signal attenuation (very small signals) or signal explosion (very large signals) in deeper layers.
# 
# 2. **Random Uniform Initialization**:
#    - Weights are drawn uniformly from a fixed range, typically between \((-0.1, 0.1)\).
#    - This can lead to issues in deeper networks as signals may attenuate or explode due to lack of variance scaling based on the layer's depth or size.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import get_data

# Define a deeper network configuration to show the effect of weight initialization
confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 64}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
]

device = torch.device("mps" if torch.backends.mps.is_available() else None)
train_loader, valid_loader = get_data(batch_size=64, subset_size=20000, seed=42)
criterion = nn.CrossEntropyLoss()

results = {
    "kaiming_loss": [],
    "random_loss": [],
    "kaiming_accuracy": [],
    "random_accuracy": [],
}

# Train with smaller learning rate and more epochs
for init_type in ["kaiming", "random"]:
    model = CNN(
        dim=64, 
        num_classes=200, 
        confs=confs, 
        in_channels=3, 
        weight_init=init_type
    )
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    loss, accuracy = train(
        model,
        epochs=5,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    
    results[f"{init_type}_loss"] = loss
    results[f"{init_type}_accuracy"] = accuracy

plt.figure(figsize=(10, 5))
# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(results["kaiming_loss"], label="Kaiming")
plt.plot(results["random_loss"], label="Random")
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(results["kaiming_accuracy"], label="Kaiming")
plt.plot(results["random_accuracy"], label="Random")
plt.legend()
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()


# ### Discussion

# ## Model Complexity
# We will examine 4 different model variants of different complexity, by using different number of layers, units or filters per layer, and pay attention to the learning curve to ensure the training is stable.

# ### Model Variant 1
# `TODO` explain architecture
# 
# 3x3 conv, 16
# 3x3 conv, 32  
# pool, 2/    
# 
# fc 500  
# fc 500 

from train_eval import train

confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
]
device = torch.device("mps" if torch.backends.mps.is_available() else None)
criterion = nn.CrossEntropyLoss()
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
summary(model, (3, 64, 64))


model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loader, valid_loader = get_data(batch_size=64, seed=42)
results = train(
    model,
    epochs=5,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)


import importlib
import plot_train
importlib.reload(plot_train)
from plot_train import plot_train
plot_train(results)


# ### Model Variant 2
# `TOOD` explain architecture (Add more layers, reduce fully connected)
# 
# 3x3 conv, 16  
# 3x3 conv, 32  
# pool, 2/   
# 3x3 conv, 64  
# pool, 2/  
# 
# fc 500  
# fc 400 

from torchsummary import summary
import importlib
import cnn
importlib.reload(cnn)
from cnn import CNN
confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 64}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 400}),
]
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
summary(model, (64, 64, 3))


import torch
import torch.optim as optim
import torch.nn as nn
from utils import get_data
import importlib
import train_eval
importlib.reload(train_eval)
from train_eval import train

device = torch.device("mps" if torch.backends.mps.is_available() else None)
# model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader, valid_loader = get_data(batch_size=64, subset_size=1000, seed=42)
results = train(
    model,
    epochs=5,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=None,
)


import importlib
import plot_train
importlib.reload(plot_train)
from plot_train import plot_train
plot_train(results)


# ### Model Variant 3
# `TODO` Add consecutive cnn layer, reduce fc
# 
# 3x3 conv, 16  
# 3x3 conv, 32  
# pool, 2/  
# 3x3 conv, 64  
# 3x3 conv, 64  
# pool, 2/  
# 
# fc 500  
# fc 300 

from torchsummary import summary
import importlib
import cnn
importlib.reload(cnn)
from cnn import CNN
confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("C", {"kernel": 3, "channels": 64}),
    ("C", {"kernel": 3, "channels": 64}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 300}),
]
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
summary(model, (64, 64, 3))


device = torch.device("mps" if torch.backends.mps.is_available() else None)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader, valid_loader = get_data(batch_size=64, seed=42)
results = train(
    model,
    epochs=5,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)


import importlib
import plot_train
importlib.reload(plot_train)
from plot_train import plot_train
plot_train(results)


# ### Model Variant 4
# Add more cnn and consecutive, reduce fc  
# 
# 3x3 conv, 16  
# 3x3 conv, 32  
# pool, 2/  
# 3x3 conv, 64  
# 3x3 conv, 64  
# pool, 2/  
# 3x3 conv, 128  
# 3x3 conv, 128  
# pool, 2/  
# 
# fc 500  
# fc 200




# ### Discussion
# - Variant 1
# - Variant 2
# - Variant 3
# - Variant 4

# # Regularization
# - Briefly describe what the goal of regularization methods in general is

# ## L1/L2
# - Explain

def my_code():
    pass


# ## Dropout
# - Explain

def my_code():
    pass


# ## Discussion
# - To what extent is this goal achieved in the given case?

# # Batchnorm (without REG, with SGD)
# - Evaluate whether Batchnorm is useful. Describe what the idea of BN is, what it is supposed to help.

def my_code():
    pass


# ## Discussion

# # Adam
# - Explain

# ## Without BN, without REG
# - Explain

def my_code():
    pass


# ## Without BN, with REG
# - Explain

def my_code():
    pass


# ## Discussion

# # Transfer Learning
# - Explain

def my_code():
    pass


# ## Discussion

# # Conclusion
