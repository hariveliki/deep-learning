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
        self.dim = utils.get_dim_after_conv_and_pool(
            dim_init=dim, confs=convolution_conf
        )

        # Process linear layers
        current_units = self.dim * self.dim * current_channels  # Track input units
        for idx, (layer, conf) in enumerate(linear_conf):
            if idx == 0:
                self.net.append(nn.Flatten())
                self.net.append(nn.Linear(current_units, conf["units"]))
                current_units = conf["units"]  # Update current units
                self.net.append(nn.ReLU())
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))
            elif idx == len(linear_conf) - 1:
                self.net.append(nn.Linear(current_units, num_classes))
            else:
                self.net.append(nn.Linear(current_units, conf["units"]))
                current_units = conf["units"]  # Update current units
                self.net.append(nn.ReLU())
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))

        if self.weight_init is not None:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif self.weight_init == "random":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] != 3:  # If channels are not in the correct position
            x = x.permute(0, 3, 1, 2)  # Change from (N,H,W,C) to (N,C,H,W)
        assert (
            x.shape[1] == 3
        ), f"Expected 3 channels in dimension 1, got shape {x.shape}"

        for layer in self.net:
            x = layer(x)

        return x
