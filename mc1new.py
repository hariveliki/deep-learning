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
# We will examine four different model variants with varying levels of complexity by adjusting the number of layers, units, or filters per layer. Additionally, we will monitor the learning curves to ensure that the training process remains stable.
# 
# I hypothesize that Tiny ImageNet will benefit from a deeper architecture—specifically, increasing the number of convolutional layers. This enhancement should enable the model to learn more complex patterns, and be get better in differentiation between 200 classes.
# 
# For the four proposed model variants, I will progressively increase the number of convolutional layers while decreasing the number of units in each dense layer. This strategy should mitigate overfitting by reducing the total number of parameters.

# ### Model Variant 1
# 
# **Architecture:**
# - **Convolutional layers:**
#   - 3x3 conv, 16 channels
#   - 3x3 conv, 32 channels
#   - Pooling layer with kernel size 2
# 
# - **Fully connected layers:**
#   - fc 500 units
#   - fc 500 units
# 
# **Evaluation:**
# - This architecture includes two convolutional and two fully connected (linear) layers.
# - The convolutional layers are effective for capturing spatial patterns within the input images.
# - The fully connected layers, with 500 units each, significantly increase the number of parameter, which can raises the risk of overfitting.
# 
# **Future considerations:**
# - Increase the number of convolutional layers.
# - Reduce the number of units in the fully connected layers.

from train_eval import train

confs = [
    ("C", {"kernel": 3, "channels": 16}),
    ("C", {"kernel": 3, "channels": 32}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 500}),
]
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
summary(model, (3, 64, 64))


device = torch.device("mps" if torch.backends.mps.is_available() else None)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loader, valid_loader = get_data(batch_size=64, seed=42)
results = train(
    model,
    epochs=20,
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
# 
# **Architecture Changes**:
# - **Convolutional Layers**:
#   - Initial layers: 3x3 conv, 16 and 3x3 conv, 32.
#   - Additional layer added: 3x3 conv, 64.
# - **Pooling**:
#   - Two pooling layers with a stride of 2.
#   - Spatial dimension reduced to 14x14.
# - **Fully Connected Layers**:
#   - Reduced units in the fully connected layer from 500 to 400.
# 
# **Parameter Impact**:
# - The added convolutional layer with 64 filters (3x3 kernel) increases parameters by **18,400**.
# - Reducing the fully connected layer units from 500 to 400 cuts linear parameters by more than half.
# 
# **Performance Expectation**:
# - **Feature Extraction**: Enhanced by the added convolutional layer, increasing the depth of representation.
# - **Overfitting Reduction**: Simplifying the fully connected layer may reduce overfitting.

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
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader, valid_loader = get_data(batch_size=64, seed=42)
results = train(
    model,
    epochs=20,
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


# ### Model Variant 3
# 
# **Architecture**:
# 
# - **Convolution Layers**:
#   - 3x3 conv, 16 channels
#   - 3x3 conv, 32 channels
#   - Pooling layer, stride 2
#   - 3x3 conv, 64 channels
#   - 3x3 conv, 64 channels
#   - Pooling layer, stride 2
# 
# - **Fully Connected Layers**:
#   - fc 500
#   - fc 300
# 
# - **Parameter Increase**: Increased parameters in the convolutional layers from **18,400** to **36,900**, by adding consecutive convolutional layer.
# - **Output Shape Comparison**: Output shape remains close to the previous architecture, adjusting from **(64, 14, 14)** to **(64, 13, 13)**.
# - **Overfitting Consideration**: The depth could improve generalization.

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
    epochs=20,
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
# 
# Architecture:
# 
# - **Convolutional Layers**:
#   - **First Stage**: Two 3x3 convolutional layers with 16 and 32 channels, followed by a pooling layer (stride 2) to reduce spatial dimensions.
#   - **Second Stage**: Two 3x3 convolutional layers with 64 channels, followed by a pooling layer (stride 2).
#   - **Third Stage**: Two 3x3 convolutional layers with 128 channels, followed by a pooling layer (stride 2).
# - **Fully Connected Layers**:
#   - First fully connected layer with 500 units.
#   - Second fully connected layer with 200 units.
# 
# - **Feature Extraction**: The model applies three stages of convolutional layers with increasing channel depth (16, 32, 64, and 128), interspersed with pooling layers to downsample spatial dimensions.
# - **Gradual Detail Extraction**: Multiple convolutional layers before each pooling step help extract detailed features progressively while reducing spatial dimensions.
# - **Output Dimensions**: Starting with an input size of (3, 64, 64) for channels, height, and width, the model outputs a final feature map of (128, 4, 4), representing a dense, high-level feature representation.
# - **Parameter Count**: Approximately 1.4 million parameters, which is the lowest so far.
# - **Performance Expectation**: With this setup, the model should show the best performance so far by extracting the most patterns and saving them to the feature space.
# 

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
    ("C", {"kernel": 3, "channels": 128}),
    ("C", {"kernel": 3, "channels": 128}),
    ("P", {"kernel": 2}),
    ("L", {"units": 500}),
    ("L", {"units": 200}),
]
model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
summary(model, (64, 64, 3))


import torch
import torch.optim as optim
import torch.nn as nn
from utils import get_data
from train_eval import train

device = torch.device("mps" if torch.backends.mps.is_available() else None)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader, valid_loader = get_data(batch_size=64, seed=42)
results = train(
    model,
    epochs=20,
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


# ### Discussion
# 
# - Comparing the training and validation loss/accuracy plots from the first to the fourth model variant reveals:
#   - The gap between training and validation loss narrows.
#   - The same trend is observed for training and validation accuracy.
#   - While the validation loss increased after 5 epochs in the first model, it only rises after 15 epochs in the last model.
#   - Although accuracy on the training set reached 100% for the first three variants, the last layer achieves a maximum accuracy of 35% after 20 epochs.
#   - The validation accuracy increases alongside the training accuracy, achieving the narrowest gap, but still remains between 15% and 20%.
# 
# The steady improvements from the first to the last variant are due to:
#   - Increasing the number of convolutional layers, which expands the feature space and extracts more patterns from the images.
#     - This increases the overall parameters only slightly compared to fully connected layers.
#   - Reducing the units in the fully connected layers, decreasing the total parameters from 14.4 million to 1 million. This simplification prevents the model from overfitting.

# # Regularization
# 
# Regularization helps mitigate overfitting and enhances a model’s ability to generalize to unseen data. There are several ways to achieve regularization, two of which are:
# 
# **L1 and L2 Regularization**  
# With L1 and L2 regularization, we penalize the model's weights by adding a regularization term to the loss function. This approach reduces the model's reliance on large weights.
# 
# - **L1 Regularization**: Adds the absolute value of each weight to the loss function. This encourages sparsity, often driving some weights to zero. The regularized loss function \( L \) with L1 regularization is:
# 
#   $$
#   L = L_0 + \lambda \sum_{i} |w_i|
#   $$
# 
#   where \( L_0 \) is the original loss, \( \lambda \) is the regularization strength, and \( w_i \) are the model weights.
# 
# - **L2 Regularization**: Adds the square of each weight to the loss function, discouraging large weights but not forcing them to zero, which helps smooth the learned patterns. The regularized loss function with L2 regularization is:
# 
#   $$
#   L = L_0 + \lambda \sum_{i} w_i^2
#   $$
# 
# **Dropout**  
# In dropout, we randomly set a proportion \( p \) of units (neurons) to zero for each training update. By disabling certain connections, dropout prevents over-reliance on specific paths.

# ## L1/L2 Regularization
# 
# We will evaluate both L1 and L2 regularization methods by conducting multiple training runs with different regularization strengths (λ). Each method will be compared against a baseline model without regularization.
# 
# **Expectation**
# 
# **Training Loss:**
# - Models with regularization should exhibit higher training loss
# - This is expected because we're adding penalty terms to the loss function:
#   - L1: λ∑|w| (sum of absolute weights)
#   - L2: λ∑w² (sum of squared weights)
# - The higher loss indicates that we're preventing the model from overfitting to the training data
# 
# **Validation Loss:**
# - Models with regularization should demonstrate:
#   - Lower validation loss in the long run
#   - Better generalization to unseen data
#   - Delayed onset of overfitting
# - Without regularization:
#   - Validation loss typically starts increasing early, i.e. shows earlier signs of overfitting
# - With regularization:
#   - Validation loss should remain stable for longer
#   - The gap between training and validation loss should be smaller
# 
# We'll test this by comparing the following configurations:
# - No regularization (baseline)
# - L1 with λ ∈ {0.0001, 0.001, 0.01}
# - L2 with λ ∈ {0.0001, 0.001, 0.01}

from torchsummary import summary
import importlib
import cnn
import l1_l2
importlib.reload(cnn)
importlib.reload(l1_l2)
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


import torch
import torch.optim as optim
import torch.nn as nn
import l1_l2
importlib.reload(l1_l2)
from l1_l2 import train
from utils import get_data

device = torch.device("mps" if torch.backends.mps.is_available() else None)
train_loader, valid_loader = get_data(batch_size=64, seed=42)
criterion = nn.CrossEntropyLoss()
reg_configs = [
    {
        "name": "No Regularization",
        "l1_lambda": 0,
        "l2_lambda": 0,
    },
    {
        "name": "l1 0.0001",
        "l1_lambda": 0.0001,
        "l2_lambda": 0,
    },
    {
        "name": "l1 0.001",
        "l1_lambda": 0.001,
        "l2_lambda": 0,
    },
    {
        "name": "l1 0.01",
        "l1_lambda": 0.01,
        "l2_lambda": 0,
    },
]
results = {}
for reg_config in reg_configs:
    print(reg_config["name"])
    model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    history = train(
        model,
        epochs=10,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        l1_lambda=reg_config["l1_lambda"],
        l2_lambda=reg_config["l2_lambda"],
    )
    results[reg_config["name"]] = history
    print("\n")


import importlib
import plot_reg
importlib.reload(plot_reg)
from plot_reg import plot_regularization_results
plot_regularization_results(results)


import torch
import torch.optim as optim
import torch.nn as nn
import l1_l2

importlib.reload(l1_l2)
from l1_l2 import train
from utils import get_data

device = torch.device("mps" if torch.backends.mps.is_available() else None)
train_loader, valid_loader = get_data(batch_size=64, seed=42)
criterion = nn.CrossEntropyLoss()
reg_configs = [
    {
        "name": "No Regularization",
        "l1_lambda": 0,
        "l2_lambda": 0,
    },
    {
        "name": "l2 0.0001",
        "l1_lambda": 0,
        "l2_lambda": 0.0001,
    },
    {
        "name": "l2 0.001",
        "l1_lambda": 0,
        "l2_lambda": 0.001,
    },
    {
        "name": "l2 0.01",
        "l1_lambda": 0,
        "l2_lambda": 0.01,
    },
]
results = {}
for reg_config in reg_configs:
    print(reg_config["name"])
    model = CNN(dim=64, num_classes=200, confs=confs, in_channels=3)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    history = train(
        model,
        epochs=10,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        l1_lambda=reg_config["l1_lambda"],
        l2_lambda=reg_config["l2_lambda"],
    )
    results[reg_config["name"]] = history
    print("\n")


import importlib
import plot_reg
importlib.reload(plot_reg)
from plot_reg import plot_regularization_results
plot_regularization_results(results)


# ### Discussion
# 
# 1. **Impact of High Regularization**
# 
#    - When the regularization strength is too high (λ ∈ {0.001, 0.01}):
#      - The model shows little to no improvement during training
#      - Both training and validation losses remain nearly constant
#      - The plateaued loss indicates that the strong regularization leads to a suboptimal solution, where the model is underfitting
# 
# 2. **Benefits of Appropriate Regularization**
#    - With correctly chosen regularization strength, e.g. (λ = 0.0001):
#      - The model shows better performance on the validation set, while the unregularized model's validation loss increases over time (indicating overfitting)
# 

# ## Dropout
# - Explain




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
