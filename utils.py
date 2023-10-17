from typing import List

def get_dim_after_conv(
    dim: int,
    conv_ksize: int,
    conv_stride=1,
    conv_padding=0
) -> int:
    return (dim - conv_ksize + 2 * conv_padding) // conv_stride + 1

def get_dim_after_pool(
    dim: int,
    pool_kernel_size: int,
    pool_stride=None,
    pool_padding=0
) -> int:
    if pool_stride is None:
        pool_stride = pool_kernel_size
    return (dim - pool_kernel_size + 2 * pool_padding) // pool_stride + 1


def get_dim_after_conv_and_pool(
        dim_init: int, layers: list, confs: List[dict]
):
    dims = []
    for n, (layer, conf) in enumerate(zip(layers, confs)):
        if n == 0 and layer == "C":
            dim = get_dim_after_conv(
                dim=dim_init,
                conv_ksize=conf["kernel"],
                conv_stride=conf["stride"],
                conv_padding=conf["padding"]
            )
            dims.append(dim)
        elif n != 0 and layer == "C":
            dim = get_dim_after_conv(
                dim=dim,
                conv_ksize=conf["kernel"],
                conv_stride=conf["stride"],
                conv_padding=conf["padding"]
            )
            dims.append(dim)
        elif n != 0 and layer == "P":
            dim = get_dim_after_pool(
                dim=dim,
                pool_kernel_size=conf["kernel"]
            )
            dims.append(dim)
    return dims[-1]
