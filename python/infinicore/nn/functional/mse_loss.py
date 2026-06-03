import infinicore


def mse_loss(input, target, reduction="mean", *, out=None):
    if out is not None:
        raise NotImplementedError("mse_loss does not support out= yet")

    if not infinicore.use_ntops or input.device.type not in ("cuda", "musa"):
        raise NotImplementedError(
            "mse_loss is only available via ntops on CUDA/MUSA devices"
        )

    return infinicore.ntops.torch.mse_loss(input, target, reduction)
