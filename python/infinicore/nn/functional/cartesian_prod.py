import infinicore


def cartesian_prod(*tensors, out=None):
    if out is not None:
        raise NotImplementedError("cartesian_prod does not support out= yet")

    if not infinicore.use_ntops or tensors[0].device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "cartesian_prod is only available via ntops on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.cartesian_prod(*tensors)
