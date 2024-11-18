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

        linear_idxs = [idx for idx, (layer, _) in enumerate(confs) if layer == "L"]
        linear_start = linear_idxs[0]
        convolution_conf = confs[:linear_start]
        linear_conf = confs[linear_start:]

        current_channels = in_channels
        for layer, conf in convolution_conf:
            if layer == "C":
                self.net.append(
                    nn.Conv2d(
                        current_channels,
                        out_channels=conf["channels"],
                        kernel_size=conf.get("kernel", 3),
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

        self.dim = utils.get_dim_after_conv_and_pool(
            dim_init=dim, confs=convolution_conf
        )

        current_units = self.dim * self.dim * current_channels
        for idx, (layer, conf) in enumerate(linear_conf):
            if idx == 0:
                self.net.append(nn.Flatten())
                self.net.append(nn.Linear(current_units, conf["units"]))
                current_units = conf["units"]
                self.net.append(nn.ReLU())
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))
            elif idx == len(linear_conf) - 1:
                self.net.append(nn.Linear(current_units, num_classes))
            else:
                self.net.append(nn.Linear(current_units, conf["units"]))
                current_units = conf["units"]
                if conf.get("batch_norm", False):
                    self.net.append(nn.BatchNorm1d(conf["units"]))
                self.net.append(nn.ReLU())
                if conf.get("dropout", 0):
                    self.net.append(nn.Dropout(conf["dropout"]))

        if self.weight_init is not None:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if self.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif self.weight_init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif self.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif self.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif self.weight_init == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif self.weight_init == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif self.weight_init == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
                elif self.weight_init == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=1.0)
                elif self.weight_init == "constant":
                    nn.init.constant_(m.weight, 0.0)
                elif self.weight_init == "ones":
                    nn.init.ones_(m.weight)
                elif self.weight_init == "zeros":
                    nn.init.zeros_(m.weight)
                elif self.weight_init == "sparse":
                    nn.init.sparse_(m.weight, sparsity=0.1)
                else:
                    raise ValueError(f"Unknown weight initialization method: {self.weight_init}")

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        assert (
            x.shape[1] == 3
        ), f"Expected 3 channels in dimension 1, got shape {x.shape}"

        for layer in self.net:
            x = layer(x)

        return x
