import torch.nn as nn
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

        linear_idxs = [idx for idx, (layer, _) in enumerate(confs) if layer == "L"]
        linear_start = linear_idxs[0]
        convolution_conf = confs[:linear_start]
        linear_conf = confs[linear_start:]
        for layer, conf in convolution_conf:
            if layer == "C":
                self.net.append(
                    nn.Conv2d(
                        in_channels,
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
                in_channels = conf["channels"]
            elif layer == "P":
                self.net.append(nn.MaxPool2d(kernel_size=conf["kernel"]))
            else:
                raise NotImplementedError(f"Layer {layer} not implemented")

        self.dim = self.get_dim_after_conv_and_pool(dim_init=dim, confs=confs)
        for idx, (layer, conf) in enumerate(linear_conf):
            if idx == 0:
                self.net.append(nn.Flatten())
                self.net.append(
                    nn.Linear(self.dim * self.dim * in_channels, conf["units"])
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

    def forward(self, x):
        N, C, H, W = x.shape
        # x = x.permute(
        #     0, 3, 1, 2
        # )  # Adjust (batch_size, H, W, C) to (batch_size, C, H, W)
        assert x.shape == (N, C, H, W)

        for layer in self.net:
            x = layer(x)

        return x

    def get_dim_after_conv(
        self, dim: int, conv_ksize: int, conv_stride=1, conv_padding=0
    ) -> int:
        return (dim - conv_ksize + 2 * conv_padding) // conv_stride + 1

    def get_dim_after_pool(
        self, dim: int, pool_kernel_size: int, pool_stride=None, pool_padding=0
    ) -> int:
        if pool_stride is None:
            pool_stride = pool_kernel_size
        return (dim - pool_kernel_size + 2 * pool_padding) // pool_stride + 1

    def get_dim_after_conv_and_pool(self, dim_init: int, confs: List[Tuple[str, dict]]):
        dims = []
        for n, (layer, conf) in enumerate(confs):
            if n == 0 and layer == "C":
                dim = self.get_dim_after_conv(
                    dim=dim_init,
                    conv_ksize=conf["kernel"],
                    conv_stride=conf.get("stride", 1),
                    conv_padding=conf.get("padding", 0),
                )
                dims.append(dim)
            elif n != 0 and layer == "C":
                dim = self.get_dim_after_conv(
                    dim=dim,
                    conv_ksize=conf["kernel"],
                    conv_stride=conf.get("stride", 1),
                    conv_padding=conf.get("padding", 0),
                )
                dims.append(dim)
            elif n != 0 and layer == "P":
                dim = self.get_dim_after_pool(dim=dim, pool_kernel_size=conf["kernel"])
                dims.append(dim)
        return dims[-1]
